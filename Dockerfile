FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POLARS_MAX_THREADS=1

RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown -R user:user /home/user/app

USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user scripts/preload_data.py scripts/preload_data.py
ARG PRELOAD_MONTHS=2026-05,2026-06,2026-07
ENV DATA_CACHE_DIR=/home/user/app/data/cache
RUN python scripts/preload_data.py --months "$PRELOAD_MONTHS" --include-stations --include-geojson

COPY --chown=user static static
COPY --chown=user app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
