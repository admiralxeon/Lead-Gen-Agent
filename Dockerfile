FROM python:3.14-slim

WORKDIR /app

# Copy requirements FIRST, before the rest of the code. Docker caches each
# instruction as a layer — as long as requirements.txt doesn't change,
# `pip install` won't re-run on every rebuild, only when a dependency
# actually changes. If COPY . . came first, editing any .py file would
# invalidate the cache and force a full reinstall every time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1 — the server must accept connections arriving
# from outside the container (from your host, via the port mapping),
# not just from processes running inside the container itself.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]