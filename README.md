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

## Deploy On Render Free

This repo includes `render.yaml` for a free Render Docker web service. Render provides the `PORT`
environment variable automatically; the Dockerfile uses it in production and falls back to port
8501 locally. The Dockerfile also disables Streamlit's file watcher in production to avoid container
`inotify` limits on Render. Render checks `/_stcore/health` before routing traffic to a deployed
instance.

The Docker image preloads these monthly Parquet files during build:

```text
2026-05
2026-06
2026-07
```

The online date picker is limited to `2026-05-01` through `2026-07-30`. The app uses a 48-hour
playback window, so `2026-07-31` would require the August Parquet file.

Render Free uses an ephemeral filesystem and only a small memory allowance. Preloading avoids the
large runtime download, but the full moving-train map can still be memory-heavy. Start with the
`Data` view, then switch to `Moving trains` or `Diagnostics` after the page has loaded.

To reduce memory use on small hosts, the app loads each 48-hour window as stop-level events: one row
per `train_line_ride_id`, `service_day`, and `train_line_station_num`. It does not keep every raw
timetable update row in memory.

### Option A: Render Dashboard

1. Open https://dashboard.render.com in your normal browser.
2. Click **New** -> **Blueprint**.
3. Connect the GitHub repository `yfeng-hsm/DBDelayMapping`.
4. Select the free plan if Render asks for a plan.
5. Create the service from `render.yaml`.

### Option B: Render Web Service From Public Git URL

If the repository is public and GitHub authorization is inconvenient:

1. Click **New** -> **Web Service**.
2. Choose **Public Git Repository**.
3. Use:

```text
https://github.com/yfeng-hsm/DBDelayMapping
```

4. Set runtime to Docker, plan to Free, and region to Frankfurt.
5. Render will build the Dockerfile.

This public-URL method is useful for testing, but it usually has fewer GitHub integration features
than connecting the repository through your GitHub account.

If Render shows `Not Found` when you use the public Git URL, the repository is probably private or
Render cannot access it. Use one of these fixes:

- Make the GitHub repository public, then retry the public Git URL.
- Keep the repository private and connect GitHub inside Render instead.
- Use Hugging Face Spaces for a simpler Streamlit-only deployment.

If the deployed `*.onrender.com` page itself shows plain `Not Found`, the service probably was not
created, failed before Render assigned the public route, or has a different URL. Open the service in
the Render Dashboard and copy the URL from the service header instead of assuming it is exactly:

```text
https://db-delay-mapping.onrender.com
```

### Option C: Render CLI

Render also provides a CLI. This is useful if the browser flow is unreliable, but it still requires
logging in to Render and connecting GitHub access for private repositories.

### Alternative: Hugging Face Spaces

For a Streamlit app, Hugging Face Spaces is often the simplest free hosting option. Create a new
Space with the Streamlit SDK, then upload `app.py`, `requirements.txt`, `README.md`, and any required
configuration files. It can be easier than Render for demos, but Docker gives you more control over
the runtime.

Minimal Hugging Face Spaces steps:

1. Create a new Space at https://huggingface.co/new-space.
2. Choose **Streamlit** as the SDK.
3. Upload the project files or connect a Git repository.
4. Keep the same `requirements.txt`.
5. Open the Space after the first build finishes.

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
