FROM python:3.11-slim

WORKDIR /app

ENV TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1

COPY requirements-api.txt requirements-streamlit.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-streamlit.txt

COPY api ./api
COPY app ./app
COPY model/ecosort_baseline.keras ./model/ecosort_baseline.keras
COPY scripts/start_snapdeploy.sh ./scripts/start_snapdeploy.sh
RUN chmod +x ./scripts/start_snapdeploy.sh

EXPOSE 8080

CMD ["./scripts/start_snapdeploy.sh"]
