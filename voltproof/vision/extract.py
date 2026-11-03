"""Ask Muse Spark Open to read a tariff off a bill image (SPEC §6.12).

The prompt and the reply parsing come from the feasibility probe (scripts/vision_probe.py imports them
from here), with one addition: when the bill is not legible the model must answer {"unreadable": true}
instead of guessing. The model only proposes; voltproof.vision.validate checks every value and the
user confirms the table before anything is planned with it.

A visitor's image is never cached, stored or hashed: the call goes through the project's LLM client
with cache=False (no cache file, no request hash in the ledger).
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Mapping, Sequence

SYSTEM = "You read electricity bills and tariff sheets. Answer with one JSON object only: no markdown, no commentary."
INSTRUCTIONS = """Read this electricity bill and return exactly this JSON object, with these keys and nothing else:
{"currency": "<ISO 4217 code>", "tariff_kind": "flat" | "time_of_use" | "hourly_dynamic", "periods": [{"start_hour": <int>, "end_hour": <int>, "import_rate": <number>}], "export_rate": <number or null>, "fixed_charge": <number or null>}
Rules:
- periods: the energy price paid for grid imports, per kWh, in the main currency unit (for example EUR, not cents), as a plain number with a decimal point. Hours are local time on a 24-hour clock; end_hour is exclusive and 24 means midnight. Every period must have start_hour < end_hour, so split a period that crosses midnight into two entries. Together the periods cover hours 0 to 24 exactly once. A flat tariff is one period from 0 to 24.
- export_rate: the price paid for energy exported to the grid, per kWh, in the same unit; null if the bill gives none.
- fixed_charge: the fixed (standing, service or base) charge as printed, a plain number; null if there is none.
- Ignore taxes, levies, totals, meter readings and consumption. Use only what is printed on the image; do not guess.
- If the bill is not legible enough to read the currency, the hours and the rates with confidence, return exactly {"unreadable": true} and nothing else."""

PURPOSE = "bill"


def image_messages(data: bytes, mime: str) -> list[dict[str, Any]]:
    """The chat messages for one image (an OpenAI-style image_url data-URI part)."""
    uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": INSTRUCTIONS}, {"type": "image_url", "image_url": {"url": uri}}]},
    ]


def parse_reply(content: str | None) -> tuple[Any, str]:
    """(answer, how): how is 'strict' (the whole reply is JSON), 'fenced' (JSON inside a markdown fence or
    with text around it) or 'invalid'."""
    text = (content or "").strip()
    try:
        return json.loads(text), "strict"
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), "fenced"
        except ValueError:
            pass
    return None, "invalid"


def read_bill(llm: Any, data: bytes, mime: str, *, tags: Mapping[str, Any]) -> dict[str, Any]:
    """One model call on one (already sanitized) image. Returns the parsed answer and the call's facts;
    the image itself is not kept."""
    started = time.monotonic()
    response = llm.chat(image_messages(data, mime), purpose=PURPOSE, tags=dict(tags), cache=False)
    answer, how = parse_reply(response.content)
    return {
        "answer": answer if isinstance(answer, dict) else None,
        "json": how,
        "cost_usd": float(getattr(response, "cost_usd", 0.0) or 0.0),
        "latency_s": round(float(getattr(response, "latency_s", 0.0) or (time.monotonic() - started)), 2),
    }


def messages_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The text parts only (for tests: the prompt the model sees, without the image)."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, Mapping) and p.get("type") == "text")
    return "\n".join(parts)
