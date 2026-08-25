# DB Delay Propagation Visualization

This Dockerized Streamlit app visualizes one day of Deutsche Bahn delay propagation using the Hugging Face dataset `piebro/deutsche-bahn-data`.

Dataset: https://huggingface.co/datasets/piebro/deutsche-bahn-data

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
