# German Trains' Delay Map

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

The first run downloads the selected monthly Parquet file into a writable cache directory. In local
Docker this is:

```text
./data/cache/
```

On hosted environments where the repository directory is read-only, the app falls back to a temporary
cache such as `/tmp/db-delay-mapping/cache`.

Monthly files are large. The smallest early files are around 100 MB; newer monthly files are around 600 MB.

## Deploy On Streamlit Community Cloud

Streamlit Community Cloud is the simplest free option for this app because it runs `app.py` directly
from GitHub and installs packages from `requirements.txt`.

1. Open https://share.streamlit.io.
2. Sign in with GitHub.
3. Deploy from the repository `yfeng-hsm/DBDelayMapping`.
4. Set the main file path to `app.py`.
5. In advanced settings, use Python `3.12`.

If dependency installation fails, expand the build log and look above the final `Installing rich`
lines. The real error is usually earlier in the log.

If the log mentions `cpython-314` and `Failed to download and build pyarrow==21.0.0`, the deployment
is using Python 3.14 with an old PyArrow pin. The current `requirements.txt` uses `pyarrow>=22`, which
has Python 3.14 wheels. Using Python 3.12 in advanced settings is still recommended for consistency
with the Dockerfile.

## Data Window

The app is configured for these months:

```text
2026-05
2026-06
2026-07
```

The date picker defaults to `2026-07-10` and is limited to `2026-05-01` through `2026-07-30`. Each
playback window starts at 06:00 and runs for 36 hours.

Small free hosting tiers can have limited memory. The full moving-train map can still be memory-heavy.
The default view is `Moving trains`; points below 30 minutes of delay are hidden by default and can be
shown with the `<30 min` checkbox.

To reduce memory use on small hosts, the app loads each 36-hour window as stop-level events: one row
per `train_line_ride_id`, `service_day`, and `train_line_station_num`. It does not keep every raw
timetable update row in memory.

## What It Shows

- A moving train map:
  - each train is animated between observed stations.
  - red trails show delayed movement.
  - brighter/larger points indicate higher delay.
  - Germany is drawn with a Mercator projection and fixed aspect ratio.
  - playback uses a continuous 36-hour window from the selected day, so the animation moves directly into the next day instead of cutting at midnight.
- Hourly p90 delay by train type.
- A delay propagation heatmap:
  - x-axis: stop sequence within a train run.
  - y-axis: the five train runs with the highest maximum delay in the window.
  - color: delay in minutes.
- A selected train run map and curve showing delay along its stop sequence.

The main tabs are ordered as `Moving trains`, `Propagation`, `Train run`, and `Diagnostics`.

## License

MIT. See [LICENSE](LICENSE).

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
