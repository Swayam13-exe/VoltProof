# VoltProof

An AI operator for sites with rooftop solar and a battery on time-of-day tariffs: a facility manager states a goal in plain language, Meta's Muse Spark Open turns it into constraints, and a verified optimizer plans the battery.

> **Status: early development.** The repository currently contains only the project configuration and the LLM client (disk cache, spend ledger, budget guard, retries) with its tests and a smoke script. The optimizer, verifier, agent, benchmark and web page are not built yet.

## Setup (Windows PowerShell)

Requires Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
notepad .env        # set BASETEN_API_KEY; never commit this file
pytest -q
```

If `Activate.ps1` is blocked, allow scripts for the current window only:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

To check the connection to the model (one paid request of a fraction of a cent; running it again is served from the local cache at $0):

```powershell
python scripts\smoke_llm.py
```

Every LLM call is recorded in `runs/ledger.jsonl` and is refused once spending would pass `BUDGET_USD_CAP` from `.env`.

## Built during the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026)

All code in this repository was written during the event. The model is Muse Spark Open (Meta), served by Baseten through an OpenAI-compatible API.

## License

MIT, see [LICENSE](LICENSE).
