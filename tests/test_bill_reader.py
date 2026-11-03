"""The bill reader (SPEC §6.12): read it, then confirm it.

Offline: the validators, the in-memory image clean-up (EXIF and GPS dropped, size and magic bytes), the
read endpoint on a fake LLM (limits, nothing written to disk, nothing logged), and a live run planned
with a confirmed bill tariff, where the rule-based battery is priced with the same tariff."""

from __future__ import annotations

import base64
import builtins
import dataclasses
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.fakes_llm import reply
from tests.test_api_live import DAY, GOAL, SITE, SPLITS, Clock, build, wait
from tests.test_counterfactual import OUTAGE, a4_llm
from voltproof.api.app import CONTENT_SECURITY_POLICY, create_app
from voltproof.api.live import Limits
from voltproof.baselines.rule_based import a0_setpoints
from voltproof.config import Settings
from voltproof.core.executor import execute
from voltproof.core.metrics import trajectory_total_cost
from voltproof.models import IntentSpec
from voltproof.vision import image as img
from voltproof.vision.extract import parse_reply, read_bill
from voltproof.vision.validate import HALF_HOUR, apply_bill, check_bill, period_names

CUR = SITE.currency  # the test site's currency (XTS, the ISO code reserved for testing)
AVERAGE = 7.5  # the fake day's average import price: 5.0 for 12 hours, 10.0 for 12 hours

TOU = {
    "currency": CUR,
    "tariff_kind": "time_of_use",
    "periods": [
        {"start_hour": 22, "end_hour": 6, "import_rate": 4.0},
        {"start_hour": 6, "end_hour": 18, "import_rate": 7.0},
        {"start_hour": 18, "end_hour": 22, "import_rate": 12.0},
    ],
    "export_rate": 2.0,
    "fixed_charge": 300.0,
}


def tou(**changes: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(TOU)), **changes}


# --- the validators -----------------------------------------------------------------------------------


def test_a_valid_time_of_use_bill_wraps_past_midnight() -> None:
    check = check_bill(TOU, CUR, AVERAGE)
    assert check["ok"] and check["errors"] == [] and check["warnings"] == []
    assert check["hourly"] == [4.0] * 6 + [7.0] * 12 + [12.0] * 4 + [4.0] * 2
    assert check["kind"] == "time_of_use" and check["export_rate"] == 2.0 and check["fixed_charge"] == 300.0
    assert period_names(check["hourly"]) == {"cheap": [0, 1, 2, 3, 4, 5, 22, 23], "peak": [18, 19, 20, 21], "normal": list(range(6, 18))}


def test_flat_and_short_kind_names_are_accepted() -> None:
    flat = check_bill({"currency": CUR.lower(), "tariff_kind": "flat", "periods": [{"start_hour": 0, "end_hour": 24, "import_rate": 6}], "export_rate": None}, CUR, AVERAGE)
    assert flat["ok"] and flat["hourly"] == [6.0] * 24 and flat["export_rate"] is None
    assert check_bill(tou(tariff_kind="tou"), CUR, AVERAGE)["kind"] == "time_of_use"


def test_a_flat_tariff_with_two_rates_blocks() -> None:
    check = check_bill(tou(tariff_kind="flat"), CUR, AVERAGE)
    assert not check["ok"] and any("A flat tariff has one rate" in e for e in check["errors"])


def test_the_currency_must_be_the_sites() -> None:
    check = check_bill(tou(currency="EUR"), CUR, AVERAGE)
    assert not check["ok"] and check["errors"] == [f"This bill is in EUR, but the site uses {CUR}. Pick the {CUR} site or another bill."]


@pytest.mark.parametrize("kind", ["tiered", "demand", "", None])
def test_other_tariff_kinds_are_not_supported(kind: Any) -> None:
    check = check_bill(tou(tariff_kind=kind), CUR, AVERAGE)
    assert not check["ok"] and "Only time-of-use and flat tariffs are supported." in check["errors"]


def test_gaps_and_overlaps_block_and_name_the_hours() -> None:
    gap = tou(periods=[{"start_hour": 0, "end_hour": 6, "import_rate": 4}, {"start_hour": 8, "end_hour": 24, "import_rate": 7}])
    assert check_bill(gap, CUR, AVERAGE)["errors"] == ["No rate for 6:00-8:00. The periods must cover the whole day."]
    overlap = tou(periods=[{"start_hour": 0, "end_hour": 12, "import_rate": 4}, {"start_hour": 10, "end_hour": 24, "import_rate": 7}])
    assert check_bill(overlap, CUR, AVERAGE)["errors"] == ["Two rates for 10:00-12:00. Each hour needs exactly one rate."]


@pytest.mark.parametrize("start, end", [(6.5, 18), ("06:30", "18:00"), (0, "17:30")])
def test_half_hour_slots_are_not_supported_yet(start: Any, end: Any) -> None:
    periods = [{"start_hour": start, "end_hour": end, "import_rate": 7}, {"start_hour": 18, "end_hour": 6, "import_rate": 4}]
    check = check_bill(tou(periods=periods), CUR, AVERAGE)
    assert not check["ok"] and any(HALF_HOUR in e for e in check["errors"])


def test_hh_mm_strings_on_the_hour_are_whole_hours() -> None:
    periods = [{"start_hour": "00:00", "end_hour": "12:00", "import_rate": "5.5"}, {"start_hour": "12", "end_hour": "24:00", "import_rate": 9}]
    check = check_bill(tou(periods=periods), CUR, AVERAGE)
    assert check["ok"] and check["hourly"] == [5.5] * 12 + [9.0] * 12


@pytest.mark.parametrize("rate", [0, -1, "abc", None, True])
def test_rates_must_be_above_zero(rate: Any) -> None:
    periods = [{"start_hour": 0, "end_hour": 12, "import_rate": rate}, {"start_hour": 12, "end_hour": 24, "import_rate": 9}]
    check = check_bill(tou(periods=periods), CUR, AVERAGE)
    assert not check["ok"] and "Row 1: the rate must be a number above zero." in check["errors"]


@pytest.mark.parametrize("rate, flagged", [(0.7, True), (0.75, False), (75.0, False), (76.0, True)])
def test_implausible_rates_are_flagged_not_blocked(rate: float, flagged: bool) -> None:
    periods = [{"start_hour": 0, "end_hour": 12, "import_rate": rate}, {"start_hour": 12, "end_hour": 24, "import_rate": 9}]
    check = check_bill(tou(periods=periods), CUR, AVERAGE)
    assert check["ok"]
    assert bool(check["warnings"]) is flagged
    if flagged:
        assert check["warnings"][0].startswith("Row 1: check this rate.")


def test_export_may_be_null_but_not_negative_and_unreadable_is_its_own_case() -> None:
    assert check_bill(tou(export_rate=None), CUR, AVERAGE)["ok"]
    assert check_bill(tou(export_rate=0), CUR, AVERAGE)["export_rate"] == 0.0
    assert not check_bill(tou(export_rate=-1), CUR, AVERAGE)["ok"]
    unreadable = check_bill({"unreadable": True}, CUR, AVERAGE)
    assert unreadable["unreadable"] and not unreadable["ok"] and len(unreadable["errors"]) == 1
    assert not check_bill(None, CUR, AVERAGE)["ok"] and not check_bill(tou(periods=[]), CUR, AVERAGE)["ok"]


def test_apply_bill_replaces_both_price_arrays_and_nothing_else() -> None:
    day, tariff = build(SITE, DAY)
    check = check_bill(TOU, CUR, AVERAGE)
    bill_day, bill_tariff = apply_bill(day, tariff, check)
    assert list(bill_day.import_price_per_kwh) == check["hourly"] == list(bill_tariff.import_price_per_kwh)
    assert list(bill_day.export_price_per_kwh) == [2.0] * 24
    assert bill_day.model_dump(exclude={"import_price_per_kwh", "export_price_per_kwh"}) == day.model_dump(exclude={"import_price_per_kwh", "export_price_per_kwh"})
    assert bill_tariff.currency == tariff.currency and bill_tariff.name == "Tariff from your bill"
    no_export, _ = apply_bill(day, tariff, check_bill(tou(export_rate=None), CUR, AVERAGE))
    assert list(no_export.export_price_per_kwh) == [0.0] * 24
    with pytest.raises(ValueError):
        apply_bill(day, tariff, check_bill(tou(currency="EUR"), CUR, AVERAGE))


def test_parse_reply_reads_strict_and_fenced_json() -> None:
    assert parse_reply('{"unreadable": true}') == ({"unreadable": True}, "strict")
    assert parse_reply('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert parse_reply("no json here") == (None, "invalid")


# --- the image ----------------------------------------------------------------------------------------


def png_bytes(size: tuple[int, int] = (40, 30), mode: str = "RGB") -> bytes:
    out = io.BytesIO()
    Image.new(mode, size, (200, 20, 20) if mode == "RGB" else (200, 20, 20, 128)).save(out, format="PNG")
    return out.getvalue()


def jpeg_with_gps() -> bytes:
    """A JPEG whose EXIF carries a camera make and a GPS position."""
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make
    exif[0x8825] = {1: "N", 2: (48.0, 8.0, 0.0), 3: "E", 4: (11.0, 34.0, 0.0)}  # GPSInfo
    out = io.BytesIO()
    Image.new("RGB", (64, 48), (10, 120, 200)).save(out, format="JPEG", exif=exif)
    return out.getvalue()


def test_a_jpeg_with_gps_exif_comes_out_without_any_metadata() -> None:
    raw = jpeg_with_gps()
    with Image.open(io.BytesIO(raw)) as before:
        assert before.getexif().get_ifd(0x8825) and before.getexif()[0x010F] == "SecretCam"  # the fixture really has GPS
    clean = img.sanitize_image(raw, "image/jpeg")
    with Image.open(io.BytesIO(clean)) as after:
        assert dict(after.getexif()) == {} and "exif" not in after.info and after.format == "JPEG"
    assert b"Exif" not in clean and b"SecretCam" not in clean


def test_large_images_are_scaled_to_1600px_and_transparency_lands_on_white() -> None:
    clean = img.sanitize_image(png_bytes((3200, 800)), "image/png")
    with Image.open(io.BytesIO(clean)) as out:
        assert out.size == (1600, 400) and out.mode == "RGB"
    with Image.open(io.BytesIO(img.sanitize_image(png_bytes((20, 20), "RGBA"), "image/png"))) as out:
        assert out.getpixel((5, 5))[0] > 200  # half-transparent red over white: light, not black


def test_size_and_magic_byte_limits() -> None:
    data = png_bytes()
    with pytest.raises(img.BillImageError, match="not a PNG or JPEG"):
        img.sanitize_image(data, "image/jpeg")  # declared JPEG, PNG bytes
    with pytest.raises(img.BillImageError, match="Only PNG and JPEG"):
        img.sanitize_image(b"GIF89a....", "image/gif")
    with pytest.raises(img.BillImageError, match="could not be opened"):
        img.sanitize_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")  # PNG magic, broken body
    with pytest.raises(img.BillImageError, match="larger than 2 MB"):
        img.sanitize_image(data + b"\x00" * img.MAX_BYTES, "image/png")
    with pytest.raises(img.BillImageError, match="larger than 2 MB"):
        img.decode_upload(base64.b64encode(b"\x00" * (img.MAX_BYTES + 1)).decode())
    with pytest.raises(img.BillImageError, match="could not be decoded"):
        img.decode_upload("not base64!")
    assert img.decode_upload(base64.b64encode(data).decode()) == data


def test_the_pixel_limit_refuses_a_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(img, "MAX_PIXELS", 100)
    with pytest.raises(img.BillImageError, match="too many pixels"):
        img.sanitize_image(png_bytes((20, 20)), "image/png")


# --- the read endpoint ----------------------------------------------------------------------------------


class BillLLM:
    """A fake model that 'reads' every bill as `answer` and records what it was sent."""

    def __init__(self, answer: Any = TOU) -> None:
        self.answer, self.calls = answer, []

    def chat(self, messages, tools=None, tool_choice=None, response_format=None, *, purpose, tags=None, cache=True):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, "purpose": purpose, "tags": dict(tags or {}), "cache": cache})
        return reply(json.dumps(self.answer), cost_usd=0.004)


def make_bill_app(settings: Settings, tmp_path: Path, llm: Any, *, live: bool = True, bill_limits: Limits | None = None, **overrides: Any) -> TestClient:
    s = dataclasses.replace(settings, live_mode=live, trusted_proxy_hops=1, **overrides)
    samples = tmp_path / "samples"
    samples.mkdir(exist_ok=True)
    (samples / "bill_1_tou_inr.png").write_bytes(png_bytes())  # the sample id is fixed; here it holds a test-currency bill
    (samples / "bill_1_tou_inr.json").write_text(json.dumps(TOU), encoding="utf-8")
    recorded = tmp_path / "results" / "vision_probe"
    recorded.mkdir(parents=True, exist_ok=True)
    (recorded / "results.json").write_text(json.dumps({"bill_1_tou_inr.png": {"answer": tou(export_rate=None), "latency_s": 9.5, "cost_usd": 0.0123}}), encoding="utf-8")
    app = create_app(settings=s, make_client=lambda _: llm, clock=Clock(), sites={SITE.site_id: SITE}, splits=SPLITS, day_builder=build, bill_limits=bill_limits, results_dir=tmp_path / "results", samples_dir=samples)
    return TestClient(app)


def upload(client: TestClient, data: bytes, mime: str = "image/jpeg", ip: str = "203.0.113.5"):
    body = {"site_id": SITE.site_id, "image_b64": base64.b64encode(data).decode(), "mime": mime}
    return client.post("/api/tariff-from-image", json=body, headers={"x-forwarded-for": f"198.51.100.1, {ip}"})


def test_samples_are_listed_and_served_and_nothing_else_is(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM())
    samples = client.get("/api/bill-samples").json()
    assert samples == [{"id": "bill_1_tou_inr", "site_id": SITE.site_id, "currency": CUR, "kind": "time_of_use", "label": f"Sample bill for {SITE.name} ({CUR}, time-of-use)"}]
    image = client.get("/api/bill-samples/bill_1_tou_inr.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.content.startswith(img.MAGIC["image/png"])
    assert client.get("/api/bill-samples/bill_2_flat_kes.png").status_code == 404  # its files are missing here
    assert client.get("/api/bill-samples/..%2Fsecret.png").status_code == 404


def test_live_off_a_sample_gives_its_recorded_read_and_an_upload_is_refused(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm, live=False)
    body = client.post("/api/tariff-from-image", json={"site_id": SITE.site_id, "sample_id": "bill_1_tou_inr"}).json()
    assert body["source"] == "recorded" and body["label"] == "Recorded read of the sample bill" and body["why"] == "live_mode_off"
    assert body["check"]["ok"] and body["check"]["export_rate"] is None and body["recorded_in"] == "results/vision_probe/results.json"
    assert body["cost_usd"] == 0.0123
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 503 and refused.json()["reason"] == "live_mode_off" and "Try a sample bill" in refused.json()["message"]
    assert llm.calls == []
    assert client.post("/api/tariff-from-image", json={"site_id": SITE.site_id, "sample_id": "nope"}).status_code == 422


def test_an_upload_is_cleaned_read_once_uncached_and_tagged(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm)
    response = upload(client, jpeg_with_gps())
    body = response.json()
    assert response.status_code == 200 and body["source"] == "live" and body["check"]["ok"] and body["cost_usd"] == 0.004
    (call,) = llm.calls
    assert call["purpose"] == "bill" and call["tags"] == {"demo": True, "purpose": "bill"} and call["cache"] is False
    sent = next(part for part in call["messages"][-1]["content"] if part.get("type") == "image_url")["image_url"]["url"]
    assert sent.startswith("data:image/jpeg;base64,")
    assert b"SecretCam" not in base64.b64decode(sent.split(",", 1)[1])  # the model gets the clean copy


def test_bad_uploads_are_refused_before_the_model(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm)
    assert upload(client, png_bytes(), mime="image/jpeg").status_code == 422
    assert upload(client, b"GIF89a" + b"\x00" * 50, mime="image/gif").json()["message"] == "Only PNG and JPEG images are accepted."
    big = client.post("/api/tariff-from-image", json={"site_id": SITE.site_id, "image_b64": "A" * 3_000_000, "mime": "image/png"})
    assert big.status_code == 422
    assert client.post("/api/tariff-from-image", json={"site_id": SITE.site_id}).status_code == 422
    assert llm.calls == []


def test_a_wrong_currency_reading_comes_back_blocked(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM(tou(currency="EUR")))
    check = upload(client, jpeg_with_gps()).json()["check"]
    assert not check["ok"] and "but the site uses" in check["errors"][0]


def test_bill_reads_have_their_own_per_visitor_and_hourly_limits(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM(), bill_reads_per_visitor=2, bill_reads_per_hour=3)
    data = jpeg_with_gps()
    assert [upload(client, data).status_code for _ in range(2)] == [200, 200]
    third = upload(client, data)
    assert third.status_code == 429 and third.json()["reason"] == "rate_limit" and third.json()["per_visitor_per_hour"] == 2 and "Retry-After" in third.headers
    assert upload(client, data, ip="203.0.113.9").status_code == 200
    everyone = upload(client, data, ip="203.0.113.10")
    assert everyone.status_code == 429 and everyone.json()["reason"] == "hourly_cap"
    status = client.get("/api/status").json()
    assert status["runs_left_this_hour"] == 10  # bill reads do not use up live runs


def test_bill_reads_stop_at_the_demo_budget(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm, demo_budget_usd_cap=0.01)
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 503 and refused.json()["reason"] == "budget" and llm.calls == []


def test_a_read_writes_nothing_to_disk_and_logs_neither_the_image_nor_its_hash(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM())
    data = jpeg_with_gps()
    writes: list[str] = []
    real_open, real_os_open, real_io_open = builtins.open, os.open, io.open

    def watch_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def watch_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND):
            writes.append(str(path))
        return real_os_open(path, flags, *args, **kwargs)

    def watch_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if any(flag in mode for flag in "wax+"):
            writes.append(str(file))
        return real_io_open(file, mode, *args, **kwargs)

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(builtins, "open", watch_open)
    monkeypatch.setattr(os, "open", watch_os_open)
    monkeypatch.setattr(io, "open", watch_io_open)
    before = sorted(p for p in tmp_path.rglob("*"))
    response = upload(client, data)
    monkeypatch.undo()
    assert response.status_code == 200
    assert writes == [] and sorted(p for p in tmp_path.rglob("*")) == before
    logged = "\n".join(r.getMessage() for r in caplog.records)
    clean = img.sanitize_image(data, "image/jpeg")
    for secret in (base64.b64encode(data).decode()[:40], base64.b64encode(clean).decode()[:40], hashlib.sha256(data).hexdigest(), hashlib.sha256(clean).hexdigest(), "203.0.113.5"):
        assert secret not in logged
    assert "bill read: ok=True unreadable=False" in logged


def test_the_check_endpoint_uses_the_same_validator(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM())
    tariff = {"kind": "time_of_use", "periods": TOU["periods"], "export_rate": 2.0}
    assert client.post("/api/tariff/check", json={"site_id": SITE.site_id, "tariff": tariff}).json() == check_bill(tou(fixed_charge=None), CUR, AVERAGE)
    bad = client.post("/api/tariff/check", json={"site_id": SITE.site_id, "tariff": {**tariff, "periods": TOU["periods"][:2]}}).json()
    assert bad["errors"] == ["No rate for 18:00-22:00. The periods must cover the whole day."]


def test_the_page_needs_no_new_csp_source() -> None:
    assert "img-src 'self' data:" in CONTENT_SECURITY_POLICY and "blob:" not in CONTENT_SECURITY_POLICY  # the preview is a data: URL


# --- a run planned with the bill's tariff -------------------------------------------------------------


def test_a_bill_tariff_run_prices_the_agent_and_the_rule_based_battery_with_the_bill(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, a4_llm())
    tariff = {"kind": "time_of_use", "periods": TOU["periods"], "export_rate": 2.0}
    started = client.post("/api/plan", json={"site": SITE.site_id, "day": DAY.isoformat(), "goal": GOAL, "tariff": tariff}, headers={"x-forwarded-for": "198.51.100.1, 203.0.113.5"})
    result = wait(client, started.json()["run_id"])
    view = result["view"]
    assert result["status"] == "done" and view["meta"]["tariff_source"] == "bill" and view["site"]["tariff_name"] == "Tariff from your bill"
    hourly = check_bill(TOU, CUR, AVERAGE)["hourly"]
    assert view["plan"]["import_price_per_kwh"] == hourly and view["plan"]["export_price_per_kwh"] == [2.0] * 24

    day, site_tariff = build(SITE, DAY)
    bill_day, _ = apply_bill(day, site_tariff, check_bill(TOU, CUR, AVERAGE))
    spec = IntentSpec.model_validate(OUTAGE)
    rule = execute(a0_setpoints(bill_day, SITE, "actual"), bill_day, SITE, spec.grid_outages)
    assert view["same_day"]["rule_based"]["realized_cost"] == round(trajectory_total_cost(rule, spec, bill_day, SITE).net_money, 2)
    site_rule = execute(a0_setpoints(day, SITE, "actual"), day, SITE, spec.grid_outages)
    assert view["same_day"]["rule_based"]["realized_cost"] != round(trajectory_total_cost(site_rule, spec, day, SITE).net_money, 2)  # the site's prices would differ


def test_an_invalid_bill_tariff_is_refused_without_using_a_run(settings: Settings, tmp_path: Path) -> None:
    llm = a4_llm()
    client = make_bill_app(settings, tmp_path, llm)
    tariff = {"kind": "time_of_use", "periods": TOU["periods"][:2], "export_rate": 2.0}
    refused = client.post("/api/plan", json={"site": SITE.site_id, "day": DAY.isoformat(), "goal": GOAL, "tariff": tariff}, headers={"x-forwarded-for": "198.51.100.1, 203.0.113.5"})
    assert refused.status_code == 422 and refused.json()["reason"] == "tariff_invalid"
    assert client.get("/api/status", headers={"x-forwarded-for": "203.0.113.5"}).json()["runs_left_this_hour"] == 10 and llm.calls == []
    wrong = client.post("/api/plan", json={"site": SITE.site_id, "day": DAY.isoformat(), "goal": GOAL, "tariff": {**tariff, "periods": TOU["periods"], "currency": "EUR"}})
    assert wrong.status_code == 422


def test_read_bill_returns_the_parsed_answer_and_cost() -> None:
    llm = BillLLM({"unreadable": True})
    out = read_bill(llm, png_bytes(), "image/png", tags={"demo": True, "purpose": "bill"})
    assert out["answer"] == {"unreadable": True} and out["json"] == "strict" and out["cost_usd"] == 0.004 and llm.calls[0]["cache"] is False


# --- Part D1: image safety ----------------------------------------------------------------------------


def png_declaring(width: int, height: int) -> bytes:
    """A valid PNG header that declares width x height, with one tiny compressed row: a few dozen bytes."""
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00" * 4)) + chunk(b"IEND", b"")


def test_a_tiny_file_declaring_a_huge_image_is_refused_fast_without_decoding_it() -> None:
    import time
    import tracemalloc

    bomb = png_declaring(60_000, 60_000)  # 3.6 gigapixels in under 100 bytes
    assert len(bomb) < 100
    tracemalloc.start()
    started = time.perf_counter()
    with pytest.raises(img.BillImageError, match="too many pixels"):
        img.sanitize_image(bomb, "image/png")
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 1.0 and peak < 5 * 1024 * 1024  # header only: no pixel buffer was allocated
    with pytest.raises(img.BillImageError, match="too many pixels"):
        img.sanitize_image(png_declaring(5001, 5000), "image/png")  # 25.005 megapixels: just over the cap


def test_pillows_own_bomb_guard_is_a_clean_refusal_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # Pillow's guard trips before ours
    with pytest.raises(img.BillImageError, match="too many pixels"):
        img.sanitize_image(png_bytes((40, 30)), "image/png")


def noisy_png(size: tuple[int, int]) -> bytes:
    import numpy as np

    out = io.BytesIO()
    Image.fromarray(np.random.default_rng(7).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(out, format="PNG")
    return out.getvalue()


def test_a_truncated_png_and_a_renamed_text_file_are_refused() -> None:
    whole = noisy_png((200, 150))
    with pytest.raises(img.BillImageError, match="could not be opened"):
        img.sanitize_image(whole[: len(whole) // 2], "image/png")
    with pytest.raises(img.BillImageError, match="not a PNG or JPEG"):
        img.sanitize_image(b"electricity bill, typed out as text\n" * 20, "image/png")  # bill.txt renamed bill.png
    with pytest.raises(img.BillImageError, match="could not be opened"):
        img.sanitize_image(b"\xff\xd8\xff" + b"not really a jpeg" * 50, "image/jpeg")  # JPEG magic, garbage after


def test_a_normal_two_megabyte_photo_still_works() -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    base = np.linspace(40, 220, 2400 * 1800 * 3).reshape(1800, 2400, 3) + rng.normal(0, 18, (1800, 2400, 3))
    photo = None
    for quality in (90, 85, 80, 75, 70, 65, 60):
        out = io.BytesIO()
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(out, format="JPEG", quality=quality)
        if out.tell() <= img.MAX_BYTES:
            photo = out.getvalue()
            break
    assert photo is not None and len(photo) > 1_000_000  # a real-sized photo, just under the 2 MB limit
    clean = img.sanitize_image(img.decode_upload(base64.b64encode(photo).decode()), "image/jpeg")
    with Image.open(io.BytesIO(clean)) as out:
        assert out.size == (1600, 1200) and out.format == "JPEG"


def test_the_endpoint_turns_every_bad_image_into_a_plain_422(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm)
    for data, mime, words in (
        (png_declaring(60_000, 60_000), "image/png", "too many pixels"),
        (noisy_png((200, 150))[:2000], "image/png", "could not be opened"),
        (b"just text" * 10, "image/png", "not a PNG or JPEG"),
    ):
        response = upload(client, data, mime=mime)
        assert response.status_code == 422 and response.json()["reason"] == "image" and words in response.json()["message"], words
    assert llm.calls == []


# --- Part D2: a sample bill never dead-ends --------------------------------------------------------------

RECORDED = "Recorded read of the sample bill"


def pick_sample(client: TestClient, ip: str = "203.0.113.5"):
    return client.post("/api/tariff-from-image", json={"site_id": SITE.site_id, "sample_id": "bill_1_tou_inr"}, headers={"x-forwarded-for": f"198.51.100.1, {ip}"})


def assert_recorded(response, why: str) -> None:
    body = response.json()
    assert response.status_code == 200 and body["source"] == "recorded" and body["label"] == RECORDED and body["why"] == why
    assert body["check"]["ok"] and "Read by" not in body["label"] and "just now" not in body["label"].lower()


def test_live_ai_off_sample_recorded_upload_refused(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM(), live=False)
    assert_recorded(pick_sample(client), "live_mode_off")
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 503 and refused.json()["reason"] == "live_mode_off"


def test_budget_used_up_sample_recorded_upload_refused(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm, demo_budget_usd_cap=0.01)
    assert_recorded(pick_sample(client), "budget")
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 503 and refused.json()["reason"] == "budget" and llm.calls == []


def test_visitor_limit_hit_sample_recorded_upload_refused(settings: Settings, tmp_path: Path) -> None:
    llm = BillLLM()
    client = make_bill_app(settings, tmp_path, llm, bill_reads_per_visitor=1)
    live = pick_sample(client).json()
    assert live["source"] == "live" and live["label"] == "Read by Muse Spark Open in 0.1 s"  # the fake reply's latency
    assert_recorded(pick_sample(client), "rate_limit")
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 429 and refused.json()["reason"] == "rate_limit" and "Retry-After" in refused.headers
    assert len(llm.calls) == 1  # recorded reads use no model call and no limit


def test_global_limit_hit_sample_recorded_upload_refused(settings: Settings, tmp_path: Path) -> None:
    client = make_bill_app(settings, tmp_path, BillLLM(), bill_reads_per_hour=1)
    assert upload(client, jpeg_with_gps(), ip="203.0.113.7").status_code == 200
    assert_recorded(pick_sample(client, ip="203.0.113.8"), "hourly_cap")
    refused = upload(client, jpeg_with_gps(), ip="203.0.113.8")
    assert refused.status_code == 429 and refused.json()["reason"] == "hourly_cap"


def test_a_failed_live_read_falls_back_for_a_sample_only(settings: Settings, tmp_path: Path) -> None:
    from voltproof.llm.client import LLMError

    class Failing(BillLLM):
        def chat(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise LLMError("upstream hiccup")

    client = make_bill_app(settings, tmp_path, Failing())
    assert_recorded(pick_sample(client), "error")
    refused = upload(client, jpeg_with_gps())
    assert refused.status_code == 503 and refused.json()["reason"] == "error"


def test_a_live_read_is_labelled_with_its_time_and_a_recorded_one_never_is(settings: Settings, tmp_path: Path) -> None:
    from tests.test_web_logic import pure

    client = make_bill_app(settings, tmp_path, BillLLM())
    body = upload(client, jpeg_with_gps()).json()
    assert body["source"] == "live" and body["label"] == "Read by Muse Spark Open in 0.1 s"
    bodies = [
        {"source": "recorded", "label": RECORDED, "why": "rate_limit", "recorded_in": "results/vision_probe/results.json"},
        {"source": "recorded", "label": RECORDED, "why": "live_mode_off"},
        {"source": "live", "label": "Read by Muse Spark Open in 7.3 s"},
    ]
    out = pure(lines=f"{json.dumps(bodies)}.map(T.billSourceText)")["lines"]
    assert out[0] == "Recorded read of the sample bill, because you have used this hour's bill reads. It was read earlier by Muse Spark Open (results/vision_probe/results.json)."
    assert out[1].startswith("Recorded read of the sample bill, because the live AI is off here.")
    assert all("Read just now" not in line and not line.startswith("Read by") for line in out[:2])
    assert out[2] == "Read by Muse Spark Open in 7.3 s."
