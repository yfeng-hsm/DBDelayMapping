---
title: DB Delay Mapping
sdk: docker
app_port: 8501
pinned: false
short_description: Visualize Deutsche Bahn delay propagation from public timetable data.
---

# DB Delay Propagation Visualization

This Dockerized Streamlit app visualizes one day of Deutsche Bahn delay propagation using the Hugging Face dataset `piebro/deutsche-bahn-data`.

Dataset: https://huggingface.co/datasets/piebro/deutsche-bahn-data

## Scope

This is a research and visualization prototype. It is not an official Deutsche Bahn product, not an
operational monitoring system, and not a real-time passenger information service.

The app depends on third-party/open datasets and derived station coordinates. Data can be incomplete,
delayed, revised, missing for some stations or train runs, or inconsistent around day boundaries. Use
the results for exploration, teaching, and model development rather than operational decisions.

When publishing the app, keep the Hugging Face dataset attribution visible and check the terms of the
data sources you use.

## Start

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

The first run downloads the selected monthly Parquet file into:

```text
./data/cache/
```

Monthly files are large. The smallest early files are around 100 MB; newer monthly files are around 600 MB.

## Deploy On Hugging Face Spaces

Use a Hugging Face **Docker Space**. Hugging Face has deprecated Streamlit as a default built-in Space
SDK, so this repository uses Docker metadata at the top of this README:

```yaml
sdk: docker
app_port: 8501
```

Create a Space:

1. Open https://huggingface.co/new-space.
2. Choose **Docker** as the SDK.
3. Push or upload this repository.
4. Keep `app.py`, `requirements.txt`, `Dockerfile`, and `scripts/preload_data.py` in the Space repo.
5. Wait for the build to finish, then open the Space app.

The Docker image preloads these monthly Parquet files during build:

```text
2026-05
2026-06
2026-07
```

The online date picker is limited to `2026-05-01` through `2026-07-30`. The app uses a 48-hour
playback window, so `2026-07-31` would require the August Parquet file.

Small free hosting tiers can have limited memory. Preloading avoids the large runtime download, but
the full moving-train map can still be memory-heavy. Start with the `Data` view, then switch to
`Moving trains` or `Diagnostics` after the page has loaded.

To reduce memory use on small hosts, the app loads each 48-hour window as stop-level events: one row
per `train_line_ride_id`, `service_day`, and `train_line_station_num`. It does not keep every raw
timetable update row in memory.

## What It Shows

- A moving train map:
  - each train is animated between observed stations.
  - red trails show delayed movement.
  - brighter/larger points indicate higher delay.
  - Germany is drawn with a Mercator projection and fixed aspect ratio.
  - playback uses a continuous 48-hour window from the selected day, so the animation moves directly into the next day instead of cutting at midnight.
- Hourly p90 delay by train type.
- A delay propagation heatmap:
  - x-axis: stop sequence within a train run.
  - y-axis: selected train run.
  - color: delay in minutes.
- A selected train run map and curve showing delay along its stop sequence.
- A table of the most delayed train runs for the selected day.

## Why This Works For Propagation

The dataset contains:

- `train_line_ride_id`: one train run.
- `train_line_station_num`: stop order within the run.
- `delay_in_min`: delay observed at that stop event.
- `time`: event time.
- planned and changed arrival/departure timestamps.

Together these fields let you reconstruct how delay changes from stop to stop during a day.

## Current Limitation

The moving train geometry interpolates directly between observed station coordinates. It does not yet follow exact rail-track geometry.

Coordinates are loaded from the public `db-hafas-stations` npm package and joined by normalized EVA/IBNR id. The cached coordinate file is stored under:

```text
./data/cache/db-hafas-stations-full.ndjson
```

The Germany outline is cached under:

```text
./data/cache/germany-states.geo.json
```

Recommended next join for ML/GNN:

```text
station_features(eva, station_name, lat, lon, state, city, station_category)
route_edges(from_eva, to_eva, distance_m, route_type, historical_delay_stats)
```
