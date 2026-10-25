#!/bin/sh
set -eu

api_port="${ECOSORT_INTERNAL_API_PORT:-8000}"
public_port="${PORT:-8080}"

# Streamlit still calls FastAPI over HTTP, but both processes share one container.
export ECOSORT_API_URL="${ECOSORT_API_URL:-http://127.0.0.1:${api_port}}"

uvicorn api.main:app --host 127.0.0.1 --port "${api_port}" &
api_pid=$!

cleanup() {
    kill "${api_pid}" 2>/dev/null || true
    if [ -n "${streamlit_pid:-}" ]; then
        kill "${streamlit_pid}" 2>/dev/null || true
        wait "${streamlit_pid}" 2>/dev/null || true
    fi
    wait "${api_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for FastAPI before exposing the Streamlit interface.
python - "${api_port}" <<'PY'
import sys
import time
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("FastAPI did not become ready in time.")
PY

streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port="${public_port}" \
    --server.headless=true \
    --browser.gatherUsageStats=false &
streamlit_pid=$!

# Exit the container if either required process stops.
while kill -0 "${api_pid}" 2>/dev/null && kill -0 "${streamlit_pid}" 2>/dev/null; do
    sleep 1
done

exit 1
