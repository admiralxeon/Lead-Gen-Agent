# Python 3.12 (not 3.14) on purpose: far better wheel availability for
# chromadb / langgraph / their transitive deps. One of the real benefits of
# containerising - the deployed runtime doesn't have to match your dev machine.
FROM python:3.12-slim

# Don't write .pyc files; stream logs instead of buffering them (so `docker logs`
# shows output immediately).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements FIRST and install, so this layer is cached and only
# reinstalls when dependencies change - not on every source edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then the source. Copy the whole context (the .dockerignore filters out venvs,
# caches, secrets and generated files) rather than listing folders one by one -
# a per-folder COPY fails the whole build if that folder happens to be missing.
COPY . .

# Run as a non-root user (basic container hardening)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]