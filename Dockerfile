# VoltProof AI demo: one container serves the page, the replays and the live API (SPEC §6.13, §6.14).
# Runs on any container host that sets $PORT (default 7860, a Hugging Face Docker Space's app_port).
# Only run-time files are copied; the key is a host secret, never in the image. LIVE_MODE=false by
# default: the host turns live runs on.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    LIVE_MODE=false

# Hugging Face Spaces run the container as uid 1000
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=app:app voltproof/ voltproof/
COPY --chown=app:app web/ web/
COPY --chown=app:app data/splits.json data/
COPY --chown=app:app data/processed/ data/processed/
COPY --chown=app:app results/report/summary.json results/report/
COPY --chown=app:app results/e2/summary.json results/e2/
COPY --chown=app:app results/e3/summary.json results/e3/
COPY --chown=app:app results/e4/summary.json results/e4/
COPY --chown=app:app LICENSE ./

# /app is the app user's, so the LLM client can create .cache/ and runs/ (cache and spend ledger) at
# run time; the image itself holds neither, and both reset when the container restarts
RUN chown app:app /app

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '7860'), timeout=4)" || exit 1

CMD ["sh", "-c", "exec python -m uvicorn voltproof.api.app:create_app --factory --host 0.0.0.0 --port ${PORT} --no-access-log"]
