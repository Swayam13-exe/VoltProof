"""Draw the synthetic sample bills for the tariff-from-image feasibility probe (SPEC §6.12).

PowerShell, from the repo root:
    python scripts\\make_vision_samples.py

Writes data/vision_samples/: three bills in different layouts, each with a ground-truth JSON next to it,
and a phone-photo-like version of each (slight rotation, lighting, blur, noise, JPEG quality 60). Every
utility is fictional and every page carries a visible "SAMPLE - synthetic" mark. Deterministic: the same
files come out every time. No LLM call, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vision_samples"
MARK = "SAMPLE - synthetic"
SIZE = (850, 1150)
INK, MUTED, WHITE = (28, 32, 40), (96, 104, 116), (255, 255, 255)
FONT_DIRS = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts"))
FONT_FILES = {False: ("arial.ttf", "DejaVuSans.ttf", "Arial.ttf"), True: ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf")}

_fonts: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if (size, bold) not in _fonts:
        for directory in FONT_DIRS:
            for name in FONT_FILES[bold]:
                if (directory / name).is_file():
                    _fonts[(size, bold)] = ImageFont.truetype(str(directory / name), size)
                    break
            if (size, bold) in _fonts:
                break
        else:
            raise SystemExit("no usable TrueType font found (looked for Arial and DejaVu Sans)")
    return _fonts[(size, bold)]


def text(d: ImageDraw.ImageDraw, xy: tuple[float, float], s: str, size: int = 18, bold: bool = False, fill=INK, anchor: str = "la") -> None:
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", SIZE, WHITE)
    return page, ImageDraw.Draw(page)


def stamp(page: Image.Image, corner: tuple[int, int]) -> None:
    """The visible mark: a boxed stamp and a faint diagonal watermark."""
    d = ImageDraw.Draw(page)
    x, y = corner
    d.rounded_rectangle((x, y, x + 196, y + 34), radius=5, outline=(190, 30, 30), width=3, fill=WHITE)
    text(d, (x + 98, y + 17), MARK, 17, True, (190, 30, 30), "mm")
    layer = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    big = Image.new("RGBA", (760, 120), (0, 0, 0, 0))
    ImageDraw.Draw(big).text((380, 60), MARK, font=font(78, True), fill=(120, 120, 130, 34), anchor="mm")
    layer.paste(big.rotate(28, expand=True, resample=Image.BICUBIC), (30, 360))
    page.paste(Image.alpha_composite(page.convert("RGBA"), layer).convert("RGB"))


# --- bill 1: time-of-use in INR, table layout -----------------------------------------------------------


def bill_inr() -> tuple[Image.Image, dict]:
    page, d = new_page()
    d.rectangle((0, 0, SIZE[0], 112), fill=(22, 52, 112))
    text(d, (40, 30), "SABARMATI VALLEY ELECTRIC LTD", 30, True, WHITE)
    text(d, (40, 72), "Commercial Electricity Bill  |  Time-of-Day tariff", 17, False, (200, 214, 240))
    text(d, (SIZE[0] - 40, 34), "Bill no. SVE-2025-118842", 16, False, WHITE, "ra")
    text(d, (SIZE[0] - 40, 58), "Bill date: 05-Dec-2025", 16, False, WHITE, "ra")
    text(d, (SIZE[0] - 40, 82), "Due date: 20-Dec-2025", 16, False, WHITE, "ra")
    d.rectangle((40, 136, SIZE[0] - 40, 262), outline=(150, 156, 170), width=2)
    for i, (k, v) in enumerate((("Consumer", "Anand Cold Storage (fictional)"), ("Consumer no.", "40-1182-775"), ("Supply", "LT-III Commercial, 3-phase"), ("Billing period", "01-Nov-2025 to 30-Nov-2025"))):
        text(d, (56, 148 + i * 28), k, 16, False, MUTED)
        text(d, (230, 148 + i * 28), v, 16, True)
    text(d, (40, 292), "Meter readings (kWh)", 18, True)
    for x, (k, v) in zip((40, 250, 460, 650), (("Previous", "128,640"), ("Current", "141,120"), ("Units consumed", "12,480"), ("Meter MF", "1"))):
        text(d, (x, 324), k, 14, False, MUTED)
        text(d, (x, 346), v, 20, True)
    text(d, (40, 402), "Time-of-Day Energy Charges", 22, True, (22, 52, 112))
    cols = (40, 310, 560, SIZE[0] - 40)
    top = 440
    d.rectangle((cols[0], top, cols[-1], top + 44), fill=(226, 232, 246))
    for x, label in zip(cols[:-1], ("Slot", "Time of day", "Rate (₹ per kWh)")):
        text(d, (x + 14, top + 22), label, 17, True, INK, "lm")
    rows = (("Night (off-peak)", "22:00 - 06:00 hrs", "₹ 5.80"), ("Day (normal)", "06:00 - 18:00 hrs", "₹ 7.40"), ("Evening peak", "18:00 - 22:00 hrs", "₹ 9.90"))
    for i, row in enumerate(rows):
        y = top + 44 + i * 50
        for x, cell in zip(cols[:-1], row):
            text(d, (x + 14, y + 25), cell, 19, i == 2 and x == cols[2], INK, "lm")
        d.line((cols[0], y, cols[-1], y), fill=(170, 176, 190), width=1)
    bottom = top + 44 + 3 * 50
    d.rectangle((cols[0], top, cols[-1], bottom), outline=(110, 116, 130), width=2)
    for x in cols[1:-1]:
        d.line((x, top, x, bottom), fill=(170, 176, 190), width=1)
    text(d, (40, 666), "Other charges and credits", 22, True, (22, 52, 112))
    for i, (k, v) in enumerate((("Fixed charge (per month)", "₹ 450.00"), ("Solar export credit (net metering), per kWh exported", "₹ 2.50"), ("Electricity duty", "15 % of energy charges"), ("Fuel and power purchase adjustment", "included in the rates above"))):
        y = 714 + i * 44
        text(d, (56, y), k, 18)
        text(d, (SIZE[0] - 56, y), v, 18, True, INK, "ra")
        d.line((40, y + 32, SIZE[0] - 40, y + 32), fill=(214, 218, 226), width=1)
    d.rectangle((40, 900, SIZE[0] - 40, 990), fill=(240, 244, 252), outline=(22, 52, 112), width=2)
    text(d, (60, 945), "Amount payable by due date", 20, True, INK, "lm")
    text(d, (SIZE[0] - 60, 945), "₹ 1,07,432.60", 30, True, (22, 52, 112), "rm")
    text(d, (40, 1050), "Pay online or at any authorised collection centre. Late payment surcharge applies after the due date.", 14, False, MUTED)
    text(d, (40, 1076), "This is a SAMPLE - synthetic bill for software testing. The utility and consumer are fictional.", 14, False, MUTED)
    stamp(page, (SIZE[0] - 236, 124 - 8 + 130))
    truth = {
        "currency": "INR",
        "tariff_kind": "time_of_use",
        "periods": [{"start_hour": 0, "end_hour": 6, "import_rate": 5.8}, {"start_hour": 6, "end_hour": 18, "import_rate": 7.4}, {"start_hour": 18, "end_hour": 22, "import_rate": 9.9}, {"start_hour": 22, "end_hour": 24, "import_rate": 5.8}],
        "export_rate": 2.5,
        "fixed_charge": 450.0,
        "meta": {"utility": "Sabarmati Valley Electric Ltd (fictional)", "layout": "table", "fixed_charge_unit": "per month", "notes": "the night slot 22:00-06:00 crosses midnight, so it is two periods"},
    }
    return page, truth


# --- bill 2: flat tariff in KES, statement layout (no tables) -------------------------------------------


def dotted(d: ImageDraw.ImageDraw, y: int, left: str, right: str, bold: bool = False, size: int = 18) -> None:
    text(d, (50, y), left, size, bold)
    text(d, (SIZE[0] - 50, y), right, size, bold, INK, "ra")
    width = int(d.textlength(left, font=font(size, bold)))
    right_w = int(d.textlength(right, font=font(size, bold)))
    for x in range(50 + width + 12, SIZE[0] - 50 - right_w - 12, 8):
        d.point((x, y + size), fill=(150, 156, 168))


def bill_kes() -> tuple[Image.Image, dict]:
    page, d = new_page()
    d.rectangle((0, 0, SIZE[0], 14), fill=(28, 120, 78))
    text(d, (50, 44), "Rift Valley Power & Light", 34, True, (28, 120, 78))
    text(d, (50, 90), "Electricity statement", 20, False, MUTED)
    text(d, (SIZE[0] - 50, 50), "Statement 2025-11 / 0067412", 16, False, MUTED, "ra")
    text(d, (SIZE[0] - 50, 76), "Issued 03 December 2025", 16, False, MUTED, "ra")
    text(d, (50, 150), "Account", 14, False, MUTED)
    text(d, (50, 172), "Mwangi Rural Health Clinic (fictional)", 20, True)
    text(d, (50, 202), "Meter 0047-1182-99  ·  Reading period 01 Nov - 30 Nov 2025", 15, False, MUTED)
    d.rounded_rectangle((50, 250, SIZE[0] - 50, 440), radius=16, fill=(236, 247, 241), outline=(28, 120, 78), width=2)
    text(d, (76, 276), "YOUR TARIFF", 14, True, (28, 120, 78))
    text(d, (76, 302), "Commercial Flat Rate (SC-F)", 22, True)
    text(d, (76, 350), "KES 24.50", 48, True, (28, 120, 78))
    text(d, (76 + d.textlength("KES 24.50", font=font(48, True)) + 14, 376), "per kWh", 22, False, INK, "lm")
    text(d, (76, 408), "The same price applies at every hour of the day and night.", 16, False, MUTED)
    units, rate, fixed = 1046, 24.50, 1200.00
    energy = units * rate
    levy = 0.05 * energy
    vat = 0.16 * (energy + fixed + levy)
    total = energy + fixed + levy + vat
    text(d, (50, 478), "How this statement was calculated", 20, True)
    dotted(d, 520, f"Energy charge   {units:,} kWh x KES {rate:.2f}", f"{energy:,.2f}")
    dotted(d, 562, "Fixed service charge (monthly)", f"{fixed:,.2f}")
    dotted(d, 604, "Rural electrification levy (5 % of energy charge)", f"{levy:,.2f}")
    dotted(d, 646, "VAT (16 %)", f"{vat:,.2f}")
    d.line((50, 694, SIZE[0] - 50, 694), fill=INK, width=2)
    dotted(d, 708, "Total due (KES)", f"{total:,.2f}", True, 22)
    d.rounded_rectangle((50, 780, SIZE[0] - 50, 880), radius=12, fill=(247, 247, 249), outline=(190, 194, 204), width=1)
    text(d, (72, 800), "Export to the grid", 16, True)
    text(d, (72, 828), "Not applicable: this account has no feed-in or net-metering agreement.", 16, False, MUTED)
    text(d, (72, 852), "Surplus solar energy is not purchased by the utility.", 16, False, MUTED)
    text(d, (50, 940), "Pay by mobile money or at any branch. Quote your account number.", 15, False, MUTED)
    text(d, (50, 966), "Queries: customer care 0700 000 000 (fictional)  ·  Disconnection notices follow Regulation 14.", 15, False, MUTED)
    text(d, (50, 1076), "This is a SAMPLE - synthetic statement for software testing. The utility and customer are fictional.", 14, False, MUTED)
    stamp(page, (SIZE[0] - 246, 1020))
    truth = {
        "currency": "KES",
        "tariff_kind": "flat",
        "periods": [{"start_hour": 0, "end_hour": 24, "import_rate": 24.5}],
        "export_rate": None,
        "fixed_charge": 1200.0,
        "meta": {"utility": "Rift Valley Power & Light (fictional)", "layout": "statement", "fixed_charge_unit": "per month", "notes": "VAT and levy are separate lines, not part of the rate; no export agreement"},
    }
    return page, truth


# --- bill 3: time-of-use in EUR, German labels, 24-hour bar ---------------------------------------------


def bill_eur() -> tuple[Image.Image, dict]:
    page, d = new_page()
    d.rectangle((0, 0, 26, SIZE[1]), fill=(236, 168, 36))
    d.ellipse((60, 40, 120, 100), fill=(236, 168, 36))
    text(d, (90, 70), "L", 34, True, WHITE, "mm")
    text(d, (140, 44), "Lindenau Energie", 32, True, (60, 48, 20))
    text(d, (140, 86), "Stromrechnung  ·  Zweizeittarif", 18, False, MUTED)
    text(d, (SIZE[0] - 50, 50), "Rechnungsnr. LE-2025-0048811", 15, False, MUTED, "ra")
    text(d, (SIZE[0] - 50, 74), "Datum 02.12.2025", 15, False, MUTED, "ra")
    for i, (k, v) in enumerate((("Kunde", "Familie Berger (fiktiv)"), ("Zählernummer", "1EMH0012345678"), ("Abrechnungszeitraum", "01.11.2025 - 30.11.2025"), ("Verbrauch gesamt", "2.940 kWh"))):
        text(d, (50 + (i % 2) * 400, 150 + (i // 2) * 56), k, 14, False, MUTED)
        text(d, (50 + (i % 2) * 400, 172 + (i // 2) * 56), v, 19, True)
    text(d, (50, 290), "Ihr Tarif: Flex Zweizeit", 24, True, (60, 48, 20))
    left, width, top = 50, 750, 340
    for hour in range(24):
        high = 6 <= hour < 22
        x0 = left + hour * width / 24
        d.rectangle((x0, top, x0 + width / 24, top + 46), fill=(236, 168, 36) if high else (70, 110, 180), outline=WHITE)
    for hour in (0, 6, 12, 18, 24):
        x = left + hour * width / 24
        d.line((x, top + 46, x, top + 58), fill=INK, width=2)
        text(d, (x, top + 62), f"{hour:02d}", 14, False, INK, "ma")
    text(d, (left + width * 3 / 24, top + 23), "NT", 20, True, WHITE, "mm")
    text(d, (left + width * 14 / 24, top + 23), "HT", 20, True, (60, 48, 20), "mm")
    text(d, (left + width * 23 / 24 - 6, top + 23), "NT", 20, True, WHITE, "mm")
    d.rectangle((50, 450, 76, 476), fill=(236, 168, 36))
    text(d, (90, 463), "Hochtarif (HT) 06:00 - 22:00 Uhr", 19, True, INK, "lm")
    text(d, (SIZE[0] - 50, 463), "32,90 ct/kWh", 22, True, INK, "rm")
    d.rectangle((50, 496, 76, 522), fill=(70, 110, 180))
    text(d, (90, 509), "Niedertarif (NT) 22:00 - 06:00 Uhr", 19, True, INK, "lm")
    text(d, (SIZE[0] - 50, 509), "24,90 ct/kWh", 22, True, INK, "rm")
    text(d, (50, 548), "Arbeitspreise inkl. 19 % Umsatzsteuer.", 15, False, MUTED)
    ht_kwh, nt_kwh = 1912, 1028
    ht, nt, base = ht_kwh * 0.3290, nt_kwh * 0.2490, 11.90
    text(d, (50, 610), "Kostenaufstellung", 22, True, (60, 48, 20))
    dotted(d, 654, f"Arbeitspreis HT   {ht_kwh:,} kWh".replace(",", "."), f"{ht:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    dotted(d, 696, f"Arbeitspreis NT   {nt_kwh:,} kWh".replace(",", "."), f"{nt:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."))
    dotted(d, 738, "Grundpreis (pro Monat)", "11,90 €")
    d.line((50, 786, SIZE[0] - 50, 786), fill=INK, width=2)
    total = ht + nt + base
    dotted(d, 800, "Rechnungsbetrag", f"{total:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."), True, 22)
    d.rounded_rectangle((50, 880, SIZE[0] - 50, 960), radius=10, fill=(252, 246, 230), outline=(236, 168, 36), width=2)
    text(d, (72, 902), "Einspeisevergütung (Photovoltaik)", 18, True)
    text(d, (SIZE[0] - 72, 902), "8,10 ct/kWh", 20, True, INK, "ra")
    text(d, (72, 930), "für ins Netz eingespeiste Energie, wird separat gutgeschrieben", 14, False, MUTED)
    text(d, (50, 1060), "Dies ist eine SAMPLE - synthetic Rechnung für Softwaretests. Versorger und Kunde sind frei erfunden.", 14, False, MUTED)
    stamp(page, (SIZE[0] - 246, 1006))
    truth = {
        "currency": "EUR",
        "tariff_kind": "time_of_use",
        "periods": [{"start_hour": 0, "end_hour": 6, "import_rate": 0.249}, {"start_hour": 6, "end_hour": 22, "import_rate": 0.329}, {"start_hour": 22, "end_hour": 24, "import_rate": 0.249}],
        "export_rate": 0.081,
        "fixed_charge": 11.9,
        "meta": {"utility": "Lindenau Energie (fictional)", "layout": "cards and 24-hour bar", "fixed_charge_unit": "per month", "notes": "German labels; rates printed in ct/kWh and decimal commas, truth is in EUR per kWh; the night slot crosses midnight"},
    }
    return page, truth


# --- the phone-photo-like version -----------------------------------------------------------------------


def photo_like(page: Image.Image, angle: float, seed: int) -> Image.Image:
    """A page on a desk, slightly rotated, unevenly lit, a little soft and noisy (JPEG quality 60 on save)."""
    rng = np.random.default_rng(seed)
    margin = 60
    desk = Image.new("RGB", (page.width + 2 * margin, page.height + 2 * margin), (96, 80, 66))
    shadow = Image.new("L", desk.size, 0)
    ImageDraw.Draw(shadow).rectangle((margin + 8, margin + 10, margin + page.width + 8, margin + page.height + 10), fill=120)
    desk.paste((40, 32, 26), mask=shadow.filter(ImageFilter.GaussianBlur(9)))
    desk.paste(page, (margin, margin))
    turned = desk.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(96, 80, 66))
    turned = turned.filter(ImageFilter.GaussianBlur(0.7))
    pixels = np.asarray(turned, dtype=np.float32)
    h, w, _ = pixels.shape
    light = 0.84 + 0.16 * (np.linspace(0, 1, w)[None, :] * 0.6 + np.linspace(1, 0, h)[:, None] * 0.4)
    pixels = pixels * light[:, :, None] + rng.normal(0, 7.0, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


BILLS = (("bill_1_tou_inr", bill_inr, 2.4), ("bill_2_flat_kes", bill_kes, -1.7), ("bill_3_tou_eur", bill_eur, 3.1))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    OUT.mkdir(parents=True, exist_ok=True)
    for seed, (name, draw, angle) in enumerate(BILLS, start=1):
        page, truth = draw()
        page.save(OUT / f"{name}.png", optimize=True)
        (OUT / f"{name}.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        photo_like(page, angle, seed).save(OUT / f"{name}_photo.jpg", quality=60)
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:<28} {path.stat().st_size / 1024:7.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
