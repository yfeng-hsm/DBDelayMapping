FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POLARS_MAX_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/preload_data.py scripts/preload_data.py
ARG PRELOAD_MONTHS=2026-05,2026-06,2026-07
RUN python scripts/preload_data.py --months "$PRELOAD_MONTHS" --include-stations --include-geojson

COPY app.py .

EXPOSE 8501 10000

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
