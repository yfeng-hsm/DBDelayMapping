from __future__ import annotations

import json
import math
import os
import tarfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import requests
import streamlit as st
import streamlit.components.v1 as components


BASE_URL = os.getenv(
    "HF_DATASET_BASE_URL",
    "https://huggingface.co/datasets/piebro/deutsche-bahn-data/resolve/main/monthly_processed_data",
)
CACHE_DIR = Path("/app/data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DERIVED_CACHE_DIR = CACHE_DIR / "derived"
DERIVED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
STATION_TGZ_URL = "https://registry.npmjs.org/db-hafas-stations/-/db-hafas-stations-2.0.0.tgz"
STATION_NDJSON = CACHE_DIR / "db-hafas-stations-full.ndjson"
GERMANY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/1_sehr_hoch.geo.json"
)
GERMANY_GEOJSON = CACHE_DIR / "germany-states.geo.json"
FERN_TYPES = ("ICE", "IC", "EC", "ECE", "TGV", "RJ", "RJX", "NJ", "EN", "FLX")
MAX_SEGMENT_SPEED_KMH = 380
MIN_SPEED_CHECK_DISTANCE_KM = 15
PREPARED_WINDOW_CACHE_VERSION = "prepared-v1"
MOVEMENT_SEGMENT_CACHE_VERSION = "segments-v5"
DELAY_CLASS_COLORS = {
    "0": "#f5f7ff",
    "15": "#f6d8df",
    "30": "#f8b9bf",
    "45": "#f99a9f",
    "75": "#fb5d60",
    "90": "#fd3e40",
    "105": "#fe1f20",
    "120+": "#ff0000",
}
DELAY_CLASS_DIAMETERS = {
    "0": 2.2,
    "15": 2.6,
    "30": 4.2,
    "45": 7.0,
    "75": 12.0,
    "90": 16.0,
    "105": 18.0,
    "120+": 20.0,
}


REQUIRED_COLUMNS = [
    "station_name",
    "eva",
    "train_number",
    "line_number",
    "final_destination_station",
    "delay_in_min",
    "time",
    "arrival_is_canceled",
    "departure_is_canceled",
    "train_type",
    "train_line_ride_id",
    "train_line_station_num",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
    "id",
]


@st.cache_data(show_spinner=False)
def ensure_month_file(year: int, month: int) -> str:
    filename = f"data-{year:04d}-{month:02d}.parquet"
    path = CACHE_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    url = f"{BASE_URL}/{filename}"
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)
    return str(path)


@st.cache_data(show_spinner=False)
def ensure_station_file() -> str:
    if STATION_NDJSON.exists() and STATION_NDJSON.stat().st_size > 0:
        return str(STATION_NDJSON)

    tgz_path = CACHE_DIR / "db-hafas-stations.tgz"
    with requests.get(STATION_TGZ_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tgz_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    with tarfile.open(tgz_path, "r:gz") as archive:
        member = archive.getmember("package/full.ndjson")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("package/full.ndjson not found in db-hafas-stations tarball")
        with STATION_NDJSON.open("wb") as handle:
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                handle.write(chunk)

    return str(STATION_NDJSON)


@st.cache_data(show_spinner=False)
def ensure_germany_geojson() -> str:
    if GERMANY_GEOJSON.exists() and GERMANY_GEOJSON.stat().st_size > 0:
        return str(GERMANY_GEOJSON)

    tmp = GERMANY_GEOJSON.with_suffix(".part")
    with requests.get(GERMANY_GEOJSON_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(GERMANY_GEOJSON)
    return str(GERMANY_GEOJSON)


@st.cache_data(show_spinner=False)
def load_germany_outline(geojson_path: str, stride: int = 12) -> list[list[list[float]]]:
    with open(geojson_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    rings: list[list[list[float]]] = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        geo_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        polygons = [coordinates] if geo_type == "Polygon" else coordinates if geo_type == "MultiPolygon" else []
        for polygon in polygons:
            for ring in polygon[:1]:
                sampled = ring[::stride]
                if ring and sampled[-1] != ring[-1]:
                    sampled.append(ring[-1])
                if len(sampled) >= 3:
                    rings.append([[round(float(lon), 4), round(float(lat), 4)] for lon, lat in sampled])
    return rings


@st.cache_data(show_spinner=False)
def load_day(path: str, selected_day: date, train_types: list[str], max_rows: int) -> pl.DataFrame:
    start = datetime.combine(selected_day, time.min)
    end = start + timedelta(days=1)

    scan = (
        pl.scan_parquet(path)
        .select(REQUIRED_COLUMNS)
        .filter((pl.col("time") >= start) & (pl.col("time") < end))
        .filter(pl.col("train_line_ride_id").is_not_null())
        .filter(pl.col("train_line_station_num").is_not_null())
        .filter(pl.col("delay_in_min").is_not_null())
    )
    if train_types:
        scan = scan.filter(pl.col("train_type").is_in(train_types))

    return scan.limit(max_rows).collect()


def month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1)


def next_month_start(dt: datetime) -> datetime:
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1)
    return datetime(dt.year, dt.month + 1, 1)


@st.cache_data(show_spinner=False)
def load_time_window(
    paths: tuple[str, ...], start: datetime, end: datetime
) -> pl.DataFrame:
    event_time_expr = pl.col("time").alias("event_time")
    scans = []
    for path in paths:
        scan = (
            pl.scan_parquet(path)
            .select(REQUIRED_COLUMNS)
            .with_columns(event_time_expr)
            .filter((pl.col("time") >= start) & (pl.col("time") < end))
            .filter(pl.col("train_line_ride_id").is_not_null())
            .filter(pl.col("train_line_station_num").is_not_null())
            .filter(pl.col("delay_in_min").is_not_null())
        )
        scan = scan.filter(pl.col("train_type").fill_null("") != "Bus")
        scans.append(scan)

    if not scans:
        return pl.DataFrame()
    return pl.concat(scans).collect()


def normalize_eva(value: object) -> str:
    text = "" if value is None else str(value)
    text = "".join(ch for ch in text if ch.isdigit())
    return text.lstrip("0") or text


@st.cache_data(show_spinner=False)
def load_station_coordinates(station_file: str, evas: tuple[str, ...]) -> pl.DataFrame:
    wanted = {normalize_eva(eva) for eva in evas if eva is not None}
    coords: dict[str, dict[str, object]] = {}

    def add_station(station: dict[str, object] | None) -> None:
        if not station:
            return
        station_id = normalize_eva(station.get("id"))
        if station_id not in wanted or station_id in coords:
            return
        location = station.get("location")
        if not isinstance(location, dict):
            return
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            return
        coords[station_id] = {
            "eva_norm": station_id,
            "coord_name": station.get("name"),
            "lat": float(lat),
            "lon": float(lon),
        }

    with open(station_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if len(coords) >= len(wanted):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            add_station(row)
            nested = row.get("station") if isinstance(row, dict) else None
            add_station(nested if isinstance(nested, dict) else None)

    if not coords:
        return pl.DataFrame({"eva_norm": [], "coord_name": [], "lat": [], "lon": []})
    return pl.DataFrame(list(coords.values()))


def attach_coordinates(day_df: pl.DataFrame, station_file: str) -> tuple[pl.DataFrame, int]:
    with_eva = day_df.with_columns(
        pl.col("eva").cast(pl.Utf8).str.replace_all(r"[^0-9]", "").str.replace(r"^0+", "").alias("eva_norm")
    )
    evas = tuple(with_eva["eva_norm"].drop_nulls().unique().to_list())
    coord_df = load_station_coordinates(station_file, evas)
    joined = with_eva.join(coord_df, on="eva_norm", how="left")
    matched = joined.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())["eva_norm"].n_unique()
    return joined, matched


def window_cache_stem(window_start: datetime, window_end: datetime) -> str:
    return f"{window_start:%Y%m%d%H%M}-{window_end:%Y%m%d%H%M}"


def prepared_window_cache_path(window_start: datetime, window_end: datetime) -> Path:
    stem = window_cache_stem(window_start, window_end)
    return DERIVED_CACHE_DIR / f"{PREPARED_WINDOW_CACHE_VERSION}-{stem}.parquet"


def movement_cache_paths(window_start: datetime, window_end: datetime) -> dict[str, Path]:
    stem = window_cache_stem(window_start, window_end)
    prefix = DERIVED_CACHE_DIR / f"{MOVEMENT_SEGMENT_CACHE_VERSION}-{stem}"
    return {
        "segments": prefix.with_suffix(".segments.parquet"),
        "stats": prefix.with_suffix(".stats.json"),
        "issues": prefix.with_suffix(".issues.parquet"),
    }


@st.cache_data(show_spinner=False)
def load_prepared_window(
    parquet_paths: tuple[str, ...], station_file: str, window_start: datetime, window_end: datetime
) -> tuple[pl.DataFrame, int, str]:
    cache_path = prepared_window_cache_path(window_start, window_end)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        df = pl.read_parquet(cache_path)
        matched = df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())["eva_norm"].n_unique()
        return df, matched, "disk cache"

    df = load_time_window(parquet_paths, window_start, window_end)
    if df.is_empty():
        return df, 0, "source parquet"

    df, matched = attach_coordinates(df, station_file)
    tmp = cache_path.with_suffix(".part")
    df.write_parquet(tmp)
    tmp.replace(cache_path)
    return df, matched, "source parquet"


def load_or_build_movement_segments(
    day_df: pl.DataFrame, window_start: datetime, window_end: datetime
) -> tuple[list[dict[str, object]], dict[str, int], pl.DataFrame, str]:
    paths = movement_cache_paths(window_start, window_end)
    if all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        segments_df = pl.read_parquet(paths["segments"])
        with paths["stats"].open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        issues = pl.read_parquet(paths["issues"])
        return segments_df.to_dicts(), {key: int(value) for key, value in stats.items()}, issues, "disk cache"

    segments, stats, issues = build_movement_segments(day_df, window_start, window_end)
    segments_df = pl.DataFrame(
        segments,
        schema={
            "t0": pl.Float64,
            "t1": pl.Float64,
            "lon0": pl.Float64,
            "lat0": pl.Float64,
            "lon1": pl.Float64,
            "lat1": pl.Float64,
            "d0": pl.Int64,
            "d1": pl.Int64,
            "cat": pl.Utf8,
        },
    )
    tmp_segments = paths["segments"].with_suffix(".segments.part")
    tmp_stats = paths["stats"].with_suffix(".stats.part")
    tmp_issues = paths["issues"].with_suffix(".issues.part")
    segments_df.write_parquet(tmp_segments)
    issues.write_parquet(tmp_issues)
    with tmp_stats.open("w", encoding="utf-8") as handle:
        json.dump({key: int(value) for key, value in stats.items()}, handle)
    tmp_segments.replace(paths["segments"])
    tmp_issues.replace(paths["issues"])
    tmp_stats.replace(paths["stats"])
    return segments, stats, issues, "computed"


def add_service_day(df: pl.DataFrame) -> pl.DataFrame:
    if "service_day" in df.columns:
        return df
    return df.with_columns(pl.col("time").dt.date().alias("service_day"))


def add_trip_instance(df: pl.DataFrame) -> pl.DataFrame:
    df = add_service_day(df)
    if "trip_instance" in df.columns:
        return df
    group_keys = ["train_line_ride_id", "service_day"]
    return (
        df.sort([*group_keys, "time", "train_line_station_num"])
        .with_columns(
            pl.col("train_line_station_num").shift(1).over(group_keys).alias("_prev_station_num"),
            pl.col("time").shift(1).over(group_keys).alias("_prev_event_time"),
        )
        .with_columns(
            pl.when(pl.col("_prev_event_time").is_null())
            .then(1)
            .when(pl.col("train_line_station_num") < pl.col("_prev_station_num"))
            .then(1)
            .when((pl.col("time") - pl.col("_prev_event_time")).dt.total_minutes() > 480)
            .then(1)
            .otherwise(0)
            .cum_sum()
            .over(group_keys)
            .alias("trip_instance")
        )
        .drop(["_prev_station_num", "_prev_event_time"])
    )


def build_trip_summary(day_df: pl.DataFrame) -> pl.DataFrame:
    day_df = add_trip_instance(day_df)
    return (
        day_df.group_by(["train_line_ride_id", "service_day", "trip_instance"])
        .agg(
            pl.col("train_type").drop_nulls().first().alias("train_type"),
            pl.col("train_number").drop_nulls().first().alias("train_number"),
            pl.col("line_number").drop_nulls().first().alias("line_number"),
            pl.col("final_destination_station").drop_nulls().last().alias("destination"),
            pl.col("time").min().alias("first_seen"),
            pl.col("time").max().alias("last_seen"),
            pl.col("station_name").n_unique().alias("station_count"),
            pl.col("delay_in_min").max().alias("max_delay"),
            pl.col("delay_in_min").mean().round(1).alias("mean_delay"),
            pl.col("arrival_is_canceled").any().alias("arrival_cancelled"),
            pl.col("departure_is_canceled").any().alias("departure_cancelled"),
        )
        .filter(pl.col("station_count") >= 3)
        .sort(["max_delay", "station_count"], descending=[True, True])
    )


def make_heatmap(df: pl.DataFrame, top_trips: pl.DataFrame) -> go.Figure:
    df = add_trip_instance(df)
    selected_runs = top_trips.select(["train_line_ride_id", "service_day", "trip_instance", "first_order"])
    plot_df = (
        df.join(selected_runs, on=["train_line_ride_id", "service_day", "trip_instance"], how="inner")
        .with_columns(
            (
                pl.col("service_day").dt.strftime("%Y-%m-%d")
                + pl.lit(" | ")
                + pl.col("train_type").fill_null("")
                + pl.lit(" ")
                + pl.col("train_number").fill_null("")
                + pl.lit(" -> ")
                + pl.col("final_destination_station").fill_null("")
                + pl.lit(" | ")
                + pl.col("train_line_ride_id").cast(pl.Utf8).str.slice(0, 8)
                + pl.lit("#")
                + pl.col("trip_instance").cast(pl.Utf8)
            ).alias("train_label")
        )
        .sort(["first_order", "train_line_station_num"])
    )
    pdf = plot_df.to_pandas()
    fig = px.scatter(
        pdf,
        x="train_line_station_num",
        y="train_label",
        color="delay_in_min",
        size="delay_abs",
        hover_name="station_name",
        hover_data={
            "train_line_station_num": True,
            "delay_in_min": True,
            "time": True,
            "arrival_is_canceled": True,
            "departure_is_canceled": True,
            "delay_abs": False,
            "train_label": False,
        },
        color_continuous_scale="RdYlGn_r",
        range_color=[-5, max(20, int(pdf["delay_in_min"].quantile(0.95))) if len(pdf) else 20],
        title="Delay propagation across selected train runs",
        labels={
            "train_line_station_num": "Stop sequence",
            "train_label": "Train run",
            "delay_in_min": "Delay (min)",
        },
    )
    fig.update_layout(height=max(520, 22 * max(1, top_trips.height)), margin=dict(l=20, r=20, t=60, b=30))
    return fig


def make_trip_line(df: pl.DataFrame, ride_id: str, service_day: date, trip_instance: int) -> go.Figure:
    df = add_trip_instance(df)
    trip = (
        df.filter(
            (pl.col("train_line_ride_id") == ride_id)
            & (pl.col("service_day") == service_day)
            & (pl.col("trip_instance") == trip_instance)
        )
        .sort("train_line_station_num")
        .with_columns(
            pl.when(pl.col("station_name").is_null())
            .then(pl.col("eva"))
            .otherwise(pl.col("station_name"))
            .alias("station_label")
        )
    )
    pdf = trip.to_pandas()
    title_parts = [
        str(pdf["train_type"].dropna().iloc[0]) if pdf["train_type"].notna().any() else "",
        str(pdf["train_number"].dropna().iloc[0]) if pdf["train_number"].notna().any() else "",
    ]
    title = " ".join(part for part in title_parts if part).strip() or ride_id
    marker_colors = [delay_symbol_color(value) for value in pdf["delay_in_min"]]
    marker_sizes = [delay_symbol_diameter(value) for value in pdf["delay_in_min"]]
    fig = px.line(
        pdf,
        x="train_line_station_num",
        y="delay_in_min",
        markers=True,
        hover_name="station_label",
        hover_data=["service_day", "time", "final_destination_station", "arrival_is_canceled", "departure_is_canceled"],
        title=f"{service_day:%Y-%m-%d} #{trip_instance} | {title}: delay along route",
        labels={"train_line_station_num": "Stop sequence", "delay_in_min": "Delay (min)"},
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig.update_traces(marker=dict(size=marker_sizes, color=marker_colors, line=dict(width=0)))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=30))
    return fig


def make_hourly_chart(df: pl.DataFrame) -> go.Figure:
    hourly = (
        df.with_columns(pl.col("time").dt.truncate("1h").alias("hour"))
        .group_by(["hour", "train_type"])
        .agg(
            pl.col("delay_in_min").mean().alias("mean_delay"),
            pl.col("delay_in_min").quantile(0.9).alias("p90_delay"),
            pl.len().alias("events"),
        )
        .sort("hour")
    )
    pdf = hourly.to_pandas()
    fig = px.line(
        pdf,
        x="hour",
        y="p90_delay",
        color="train_type",
        markers=True,
        title="Hourly p90 delay by train type",
        labels={"hour": "Hour", "p90_delay": "P90 delay (min)", "train_type": "Train type"},
    )
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=30))
    return fig


def style_delay_geo(fig: go.Figure, title: str, height: int = 780) -> go.Figure:
    fig.update_geos(
        visible=True,
        resolution=50,
        lataxis_range=[47.0, 55.6],
        lonaxis_range=[5.2, 15.6],
        showland=True,
        landcolor="#071016",
        showocean=True,
        oceancolor="#020609",
        showlakes=False,
        showcountries=True,
        countrycolor="#536572",
        showsubunits=True,
        subunitcolor="#26333c",
        coastlinecolor="#536572",
        bgcolor="#020609",
        projection_type="mercator",
    )
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="#020609",
        plot_bgcolor="#020609",
        font=dict(color="#eef3f8"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def make_delay_map(df: pl.DataFrame, bin_minutes: int, min_delay: int) -> go.Figure:
    bin_expr = f"{bin_minutes}m"
    base = (
        df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        .filter(pl.col("delay_in_min") >= min_delay)
        .with_columns(pl.col("time").dt.truncate(bin_expr).alias("time_bin"))
        .group_by(["time_bin", "eva_norm", "station_name", "coord_name", "lat", "lon"])
        .agg(
            pl.col("delay_in_min").max().alias("max_delay"),
            pl.col("delay_in_min").mean().round(1).alias("mean_delay"),
            pl.col("train_line_ride_id").n_unique().alias("train_runs"),
            pl.len().alias("events"),
        )
        .with_columns(
            pl.col("max_delay").clip(1, 90).alias("bubble_size"),
            pl.col("time_bin").dt.strftime("%H:%M").alias("time_label"),
        )
        .sort(["time_bin", "max_delay"], descending=[False, True])
    )
    if base.is_empty():
        fig = go.Figure()
        fig.update_layout(title="No mappable delayed events for this filter")
        return fig

    bins = base.select("time_bin").unique().sort("time_bin")["time_bin"].to_list()
    color_max = max(20, int(base["max_delay"].quantile(0.98)))
    trail_steps = 4
    frames = []

    for current_idx, current_bin in enumerate(bins):
        window_bins = bins[max(0, current_idx - trail_steps + 1) : current_idx + 1]
        frame_traces = []
        for age in range(trail_steps):
            opacity = max(0.16, 0.82 - age * 0.18)
            size_scale = max(0.45, 1.0 - age * 0.15)
            if age < len(window_bins):
                bin_value = list(reversed(window_bins))[age]
                slice_df = base.filter(pl.col("time_bin") == bin_value)
                pdf = slice_df.to_pandas()
                name = f"{bin_value:%H:%M}" if age == 0 else f"trail -{age}"
                lat = pdf["lat"]
                lon = pdf["lon"]
                marker_size = pdf["bubble_size"].clip(1, 90) * size_scale + 4
                marker_color = pdf["max_delay"]
                text = pdf["station_name"]
                customdata = pdf[["time_label", "max_delay", "mean_delay", "train_runs", "events"]].to_numpy()
            else:
                name = f"trail -{age}"
                lat = []
                lon = []
                marker_size = []
                marker_color = []
                text = []
                customdata = []
            frame_traces.append(
                go.Scattergeo(
                    lat=lat,
                    lon=lon,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=marker_color,
                        colorscale="Reds",
                        cmin=0,
                        cmax=color_max,
                        opacity=opacity,
                        line=dict(width=0),
                        colorbar=dict(title="minutes late") if age == 0 else None,
                    ),
                    text=text,
                    customdata=customdata,
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Max delay %{customdata[1]} min<br>Mean delay %{customdata[2]} min<br>Train runs %{customdata[3]}<br>Events %{customdata[4]}<extra></extra>",
                    name=name,
                    showlegend=False,
                )
            )
        frames.append(go.Frame(data=frame_traces, name=f"{current_bin:%H:%M}"))

    fig = go.Figure(data=frames[0].data, frames=frames)
    slider_steps = [
        {
            "args": [
                [frame.name],
                {"frame": {"duration": 350, "redraw": True}, "mode": "immediate", "transition": {"duration": 150}},
            ],
            "label": frame.name,
            "method": "animate",
        }
        for frame in frames
    ]
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.05,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 350, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 150},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Time "},
                "pad": {"t": 35},
                "steps": slider_steps,
            }
        ],
    )
    return style_delay_geo(fig, "Delay diffusion map with fading trail")


def make_trip_map(df: pl.DataFrame, ride_id: str, service_day: date, trip_instance: int) -> go.Figure:
    df = add_trip_instance(df)
    trip = (
        df.filter(
            (pl.col("train_line_ride_id") == ride_id)
            & (pl.col("service_day") == service_day)
            & (pl.col("trip_instance") == trip_instance)
        )
        .filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        .sort("train_line_station_num")
    )
    fig = go.Figure()
    if trip.is_empty():
        fig.update_layout(title="No coordinates found for this train run")
        return fig

    pdf = trip.to_pandas()
    label = " ".join(
        part
        for part in [
            str(pdf["train_type"].dropna().iloc[0]) if pdf["train_type"].notna().any() else "",
            str(pdf["train_number"].dropna().iloc[0]) if pdf["train_number"].notna().any() else "",
        ]
        if part
    ).strip()
    pdf["delay_class"] = [delay_class_value(max(0.0, float(value))) for value in pdf["delay_in_min"]]
    pdf["symbol_color"] = [delay_symbol_color(value) for value in pdf["delay_in_min"]]
    pdf["symbol_size"] = [delay_symbol_diameter(value) for value in pdf["delay_in_min"]]
    fig.add_trace(
        go.Scattergeo(
            lat=pdf["lat"],
            lon=pdf["lon"],
            mode="lines+markers",
            line=dict(width=2, color="rgba(170,195,210,0.58)"),
            marker=dict(
                size=pdf["symbol_size"],
                color=pdf["symbol_color"],
                opacity=0.82,
                line=dict(width=0),
            ),
            text=pdf["station_name"],
            customdata=pdf[["train_line_station_num", "delay_in_min", "delay_class", "time"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>Stop %{customdata[0]}<br>Delay %{customdata[1]} min<br>Class %{customdata[2]} min<br>%{customdata[3]}<extra></extra>",
        )
    )
    return style_delay_geo(fig, f"{service_day:%Y-%m-%d} #{trip_instance} | {label or ride_id}: route delay trace", height=620)


def build_movement_segments(
    df: pl.DataFrame, window_start: datetime, window_end: datetime
) -> tuple[list[dict[str, object]], dict[str, int], pl.DataFrame]:
    base = (
        df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        .filter(pl.col("train_line_ride_id").is_not_null())
        .with_columns(pl.col("time").alias("event_time"))
        .filter(pl.col("event_time").is_not_null())
        .filter((pl.col("event_time") >= window_start) & (pl.col("event_time") < window_end))
        .sort(["train_line_ride_id", "event_time", "train_line_station_num"])
        .with_columns(
            pl.col("train_line_station_num").shift(1).over("train_line_ride_id").alias("_prev_station_num"),
            pl.col("event_time").shift(1).over("train_line_ride_id").alias("_prev_event_time"),
        )
        .with_columns(
            pl.when(pl.col("_prev_event_time").is_null())
            .then(1)
            .when(pl.col("train_line_station_num") < pl.col("_prev_station_num"))
            .then(1)
            .when((pl.col("event_time") - pl.col("_prev_event_time")).dt.total_minutes() > 480)
            .then(1)
            .otherwise(0)
            .cum_sum()
            .over("train_line_ride_id")
            .alias("trip_instance")
        )
        .sort(["train_line_ride_id", "trip_instance", "train_line_station_num", "event_time"])
    )
    stats = {
        "candidate_rows": base.height,
        "candidate_train_runs": base.select(["train_line_ride_id", "trip_instance"]).unique().height
        if not base.is_empty()
        else 0,
        "rendered_train_runs": 0,
        "rendered_points": 0,
        "non_monotonic_points": 0,
        "short_segments": 0,
        "long_segments": 0,
        "implausible_speed_segments": 0,
        "single_point_runs": 0,
        "no_valid_segment_runs": 0,
    }
    if base.is_empty():
        return [], stats, pl.DataFrame()

    stops = (
        base.group_by(["train_line_ride_id", "trip_instance", "train_line_station_num"])
        .agg(
            pl.col("event_time").min().alias("first_event_time"),
            pl.col("event_time").max().alias("last_event_time"),
            pl.col("delay_in_min").sort_by("event_time").first().alias("first_delay"),
            pl.col("delay_in_min").sort_by("event_time").last().alias("last_delay"),
            pl.col("arrival_change_time").sort_by("event_time").drop_nulls().first().alias("arrival_time"),
            pl.col("departure_change_time").sort_by("event_time").drop_nulls().last().alias("departure_time"),
            pl.col("arrival_planned_time").sort_by("event_time").drop_nulls().first().alias("arrival_planned_time"),
            pl.col("departure_planned_time").sort_by("event_time").drop_nulls().last().alias("departure_planned_time"),
            pl.col("lat").sort_by("event_time").first().alias("lat"),
            pl.col("lon").sort_by("event_time").first().alias("lon"),
            pl.col("train_type").sort_by("event_time").drop_nulls().first().alias("train_type"),
        )
        .with_columns(
            pl.when(pl.col("arrival_time").is_not_null() & pl.col("arrival_planned_time").is_not_null())
            .then((pl.col("arrival_time") - pl.col("arrival_planned_time")).dt.total_minutes())
            .otherwise(None)
            .alias("arrival_delay"),
            pl.when(pl.col("departure_time").is_not_null() & pl.col("departure_planned_time").is_not_null())
            .then((pl.col("departure_time") - pl.col("departure_planned_time")).dt.total_minutes())
            .otherwise(None)
            .alias("departure_delay"),
        )
        .with_columns(
            pl.coalesce(["departure_time", "last_event_time", "arrival_time", "first_event_time"]).alias(
                "segment_start_time"
            ),
            pl.coalesce(["departure_delay", "last_delay", "arrival_delay", "first_delay"]).alias(
                "segment_start_delay"
            ),
            pl.coalesce(["arrival_time", "first_event_time", "departure_time", "last_event_time"]).alias(
                "segment_end_time"
            ),
            pl.coalesce(["arrival_delay", "first_delay", "departure_delay", "last_delay"]).alias(
                "segment_end_delay"
            ),
            pl.when(pl.col("train_type").is_in(FERN_TYPES))
            .then(pl.lit("fern"))
            .otherwise(pl.lit("regional"))
            .alias("train_category"),
        )
        .sort(["train_line_ride_id", "trip_instance", "train_line_station_num", "first_event_time"])
    )
    stats["rendered_points"] = stops.height
    run_sizes = stops.group_by(["train_line_ride_id", "trip_instance"]).agg(pl.len().alias("points"))
    stats["single_point_runs"] = run_sizes.filter(pl.col("points") < 2).height

    pairs = (
        stops.with_columns(
            pl.col("segment_end_time")
            .shift(-1)
            .over(["train_line_ride_id", "trip_instance"])
            .alias("next_arrival_time"),
            pl.col("segment_end_delay")
            .shift(-1)
            .over(["train_line_ride_id", "trip_instance"])
            .alias("next_arrival_delay"),
            pl.col("lat").shift(-1).over(["train_line_ride_id", "trip_instance"]).alias("next_lat"),
            pl.col("lon").shift(-1).over(["train_line_ride_id", "trip_instance"]).alias("next_lon"),
        )
        .filter(pl.col("next_arrival_time").is_not_null())
        .with_columns((pl.col("next_arrival_time") - pl.col("segment_start_time")).dt.total_minutes().alias("gap_min"))
        .with_columns(
            (
                (
                    (
                        (pl.col("next_lon") - pl.col("lon"))
                        * 111.32
                        * (((pl.col("lat") + pl.col("next_lat")) / 2) * math.pi / 180).cos()
                    )
                    ** 2
                    + ((pl.col("next_lat") - pl.col("lat")) * 110.57) ** 2
                )
                ** 0.5
            ).alias("distance_km")
        )
        .with_columns((pl.col("distance_km") / (pl.col("gap_min") / 60)).alias("speed_kmh"))
        .with_columns(
            (
                (pl.col("distance_km") >= MIN_SPEED_CHECK_DISTANCE_KM)
                & (pl.col("speed_kmh") > MAX_SEGMENT_SPEED_KMH)
            ).alias("is_implausible_speed")
        )
    )
    stats["short_segments"] = pairs.filter(pl.col("gap_min") < 2).height
    stats["long_segments"] = pairs.filter(pl.col("gap_min") > 360).height
    stats["implausible_speed_segments"] = pairs.filter(pl.col("is_implausible_speed")).height
    valid = pairs.filter((pl.col("gap_min") >= 2) & (pl.col("gap_min") <= 360) & ~pl.col("is_implausible_speed"))
    stats["rendered_train_runs"] = valid.select(["train_line_ride_id", "trip_instance"]).unique().height
    stats["no_valid_segment_runs"] = max(0, stats["candidate_train_runs"] - stats["rendered_train_runs"] - stats["single_point_runs"])

    travel_segments_df = (
        valid.with_columns(
            ((pl.col("segment_start_time") - pl.lit(window_start)).dt.total_seconds() / 60).round(1).alias("t0"),
            ((pl.col("next_arrival_time") - pl.lit(window_start)).dt.total_seconds() / 60).round(1).alias("t1"),
            pl.col("lon").round(4).alias("lon0"),
            pl.col("lat").round(4).alias("lat0"),
            pl.col("next_lon").round(4).alias("lon1"),
            pl.col("next_lat").round(4).alias("lat1"),
            pl.col("segment_start_delay").fill_null(0).round(0).cast(pl.Int32).alias("d0"),
            pl.col("next_arrival_delay").fill_null(0).round(0).cast(pl.Int32).alias("d1"),
            pl.col("train_category").alias("cat"),
        )
        .select(["t0", "t1", "lon0", "lat0", "lon1", "lat1", "d0", "d1", "cat"])
    )
    dwell_segments_df = (
        stops.with_columns((pl.col("segment_start_time") - pl.col("segment_end_time")).dt.total_minutes().alias("dwell_min"))
        .filter((pl.col("dwell_min") >= 1) & (pl.col("dwell_min") <= 120))
        .with_columns(
            ((pl.col("segment_end_time") - pl.lit(window_start)).dt.total_seconds() / 60).round(1).alias("t0"),
            ((pl.col("segment_start_time") - pl.lit(window_start)).dt.total_seconds() / 60).round(1).alias("t1"),
            pl.col("lon").round(4).alias("lon0"),
            pl.col("lat").round(4).alias("lat0"),
            pl.col("lon").round(4).alias("lon1"),
            pl.col("lat").round(4).alias("lat1"),
            pl.col("segment_end_delay").fill_null(0).round(0).cast(pl.Int32).alias("d0"),
            pl.col("segment_start_delay").fill_null(0).round(0).cast(pl.Int32).alias("d1"),
            pl.col("train_category").alias("cat"),
        )
        .select(["t0", "t1", "lon0", "lat0", "lon1", "lat1", "d0", "d1", "cat"])
    )
    segments_df = pl.concat([travel_segments_df, dwell_segments_df]).sort("t0")
    issue_df = (
        pairs.filter((pl.col("gap_min") < 2) | (pl.col("gap_min") > 360) | pl.col("is_implausible_speed"))
        .select(
            [
                "train_line_ride_id",
                "trip_instance",
                "segment_start_time",
                "next_arrival_time",
                "gap_min",
                "distance_km",
                "speed_kmh",
                "is_implausible_speed",
            ]
        )
        .sort("speed_kmh", descending=True)
        .head(500)
    )
    return segments_df.to_dicts(), stats, issue_df


def delay_class_value(delay: float) -> str:
    if delay <= 0.5:
        return "0"
    if delay < 30:
        return "15"
    if delay < 45:
        return "30"
    if delay < 75:
        return "45"
    if delay < 90:
        return "75"
    if delay < 105:
        return "90"
    if delay < 120:
        return "105"
    return "120+"


def delay_symbol_color(delay: object) -> str:
    value = 0.0 if delay is None else float(delay)
    return DELAY_CLASS_COLORS[delay_class_value(max(0.0, value))]


def delay_symbol_diameter(delay: object) -> float:
    value = 0.0 if delay is None else float(delay)
    return DELAY_CLASS_DIAMETERS[delay_class_value(max(0.0, value))]


def build_active_segment_audit(
    segments: list[dict[str, object]], window_start: datetime, window_end: datetime
) -> pl.DataFrame:
    rows = []
    max_minute = int((window_end - window_start).total_seconds() // 60)
    for minute in range(0, max_minute + 1, 15):
        counts = {"0": 0, "15": 0, "30": 0, "45": 0, "75": 0, "90": 0, "105": 0, "120+": 0}
        active_segments = 0
        for segment in segments:
            if segment["t0"] <= minute < segment["t1"]:
                span = max(0.001, segment["t1"] - segment["t0"])
                u = (minute - segment["t0"]) / span
                delay = segment["d0"] + (segment["d1"] - segment["d0"]) * u
                counts[delay_class_value(max(0, delay))] += 1
                active_segments += 1
        rows.append(
            {
                "time": window_start + timedelta(minutes=minute),
                "active_segments": active_segments,
                "delay_0": counts["0"],
                "delay_15": counts["15"],
                "delay_30": counts["30"],
                "delay_45": counts["45"],
                "delay_75": counts["75"],
                "delay_90": counts["90"],
                "delay_105": counts["105"],
                "delay_120_plus": counts["120+"],
            }
        )
    return pl.DataFrame(rows)


def build_hourly_coverage(df: pl.DataFrame, window_start: datetime, window_end: datetime) -> pl.DataFrame:
    hours = []
    current = window_start
    while current < window_end:
        hours.append({"hour": current})
        current += timedelta(hours=1)
    timeline = pl.DataFrame(hours).with_columns(pl.col("hour").cast(pl.Datetime("us")))
    coverage = (
        df.with_columns(pl.col("event_time").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg(
            pl.len().alias("rows"),
            pl.col("train_line_ride_id").n_unique().alias("train_runs"),
            pl.col("train_type").n_unique().alias("train_types"),
            pl.col("delay_in_min").max().alias("max_delay"),
        )
        .with_columns(pl.col("hour").cast(pl.Datetime("us")))
        .sort("hour")
    )
    return (
        timeline.join(coverage, on="hour", how="left")
        .with_columns(
            pl.col("rows").fill_null(0),
            pl.col("train_runs").fill_null(0),
            pl.col("train_types").fill_null(0),
            pl.col("max_delay").fill_null(0),
        )
        .sort("hour")
    )


def make_train_flow_animation(
    segments: list[dict[str, object]],
    window_start: datetime,
    window_end: datetime,
    outline: list[list[list[float]]],
) -> str:
    if not segments:
        return "<div style='color:#eee;padding:24px'>No mappable train movement data for this filter.</div>"

    start_time = window_start
    end_time = window_end
    payload = {
        "segments": segments,
        "outline": outline,
        "minT": 0,
        "maxT": round((end_time - start_time).total_seconds() / 60, 2),
        "startLabel": start_time.strftime("%Y-%m-%d %H:%M"),
        "startEpochMs": int(start_time.timestamp() * 1000),
        "startParts": [start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute],
        "segmentCount": len(segments),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""
<div id="train-flow-root" style="height:780px;background:#020609;color:#eef3f8;position:relative;overflow:hidden;border:1px solid #182630">
  <canvas id="train-flow-canvas" style="width:100%;height:100%;display:block"></canvas>
  <div style="position:absolute;left:24px;top:18px;font:600 15px system-ui, sans-serif;letter-spacing:.02em">Moving train delay flow</div>
  <div style="position:absolute;left:24px;top:46px;display:flex;align-items:center;gap:10px;font:13px system-ui, sans-serif;color:#cbd7df;background:rgba(2,6,9,.64);padding:6px 8px;border:1px solid #24343e">
    <label style="display:flex;align-items:center;gap:6px"><input id="train-flow-fern" type="checkbox" checked style="accent-color:#ff5f55"><span>Fernverkehr</span></label>
    <label style="display:flex;align-items:center;gap:6px"><input id="train-flow-regional" type="checkbox" checked style="accent-color:#ff5f55"><span>Regionalverkehr</span></label>
    <label style="display:flex;align-items:center;gap:6px"><input id="train-flow-under-30" type="checkbox" checked style="accent-color:#ff5f55"><span>&lt;30 min</span></label>
  </div>
  <div id="train-flow-clock" style="position:absolute;right:28px;top:20px;font:700 28px ui-monospace, SFMono-Regular, Menlo, monospace;color:#fff"></div>
  <div id="train-flow-count" style="position:absolute;right:30px;top:58px;font:13px ui-monospace, SFMono-Regular, Menlo, monospace;color:#aebdc6"></div>
  <div style="position:absolute;right:24px;bottom:20px;font:12px ui-monospace, SFMono-Regular, Menlo, monospace;color:#cbd7df;background:rgba(2,6,9,.76);padding:12px 14px;border:1px solid #2e404b;min-width:132px">
    <div style="margin-bottom:8px;color:#eef3f8">Delay class</div>
    <div style="display:flex;align-items:center;gap:9px;height:22px"><span style="width:3px;height:3px;border-radius:50%;background:#f5f7ff;display:inline-block"></span><span>0 min</span></div>
    <div style="display:flex;align-items:center;gap:9px;height:24px"><span style="width:3px;height:3px;border-radius:50%;background:#f6d8df;display:inline-block"></span><span>15 min</span></div>
    <div style="display:flex;align-items:center;gap:9px;height:24px"><span style="width:4px;height:4px;border-radius:50%;background:#f8b9bf;display:inline-block"></span><span>30 min</span></div>
    <div style="display:flex;align-items:center;gap:8px;height:26px"><span style="width:7px;height:7px;border-radius:50%;background:#f99a9f;display:inline-block"></span><span>45 min</span></div>
    <div style="display:flex;align-items:center;gap:7px;height:30px"><span style="width:12px;height:12px;border-radius:50%;background:#fb5d60;display:inline-block;box-shadow:0 0 8px rgba(251,93,96,.55)"></span><span>75 min</span></div>
    <div style="display:flex;align-items:center;gap:6px;height:34px"><span style="width:16px;height:16px;border-radius:50%;background:#fd3e40;display:inline-block;box-shadow:0 0 12px rgba(253,62,64,.72)"></span><span>90 min</span></div>
    <div style="display:flex;align-items:center;gap:5px;height:36px"><span style="width:18px;height:18px;border-radius:50%;background:#fe1f20;display:inline-block;box-shadow:0 0 13px rgba(254,31,32,.74)"></span><span>105 min</span></div>
    <div style="display:flex;align-items:center;gap:4px;height:38px"><span style="width:20px;height:20px;border-radius:50%;background:#ff0000;display:inline-block;box-shadow:0 0 14px rgba(255,0,0,.78)"></span><span>120+ min</span></div>
  </div>
</div>
<script>
(() => {{
  const payload = {payload_json};
  const root = document.getElementById("train-flow-root");
  const canvas = document.getElementById("train-flow-canvas");
  const ctx = canvas.getContext("2d");
  const clock = document.getElementById("train-flow-clock");
  const count = document.getElementById("train-flow-count");
  const fernToggle = document.getElementById("train-flow-fern");
  const regionalToggle = document.getElementById("train-flow-regional");
  const under30Toggle = document.getElementById("train-flow-under-30");
  const cities = [
    ["Hamburg", 10.00, 53.55], ["Berlin", 13.40, 52.52], ["Hannover", 9.73, 52.37],
    ["Köln", 6.96, 50.94], ["Frankfurt", 8.68, 50.11], ["Leipzig", 12.37, 51.34],
    ["Dresden", 13.74, 51.05], ["Nürnberg", 11.08, 49.45], ["Stuttgart", 9.18, 48.78],
    ["München", 11.58, 48.14], ["Bremen", 8.80, 53.08]
  ];
  const lonMin = 5.2, lonMax = 15.6, latMin = 47.0, latMax = 55.6;
  let w = 0, h = 0, dpr = 1;
  let fit = null;
  let simT = payload.minT;
  let last = performance.now();
  let lastDraw = 0;
  let showFern = fernToggle.checked;
  let showRegional = regionalToggle.checked;
  let showUnder30 = under30Toggle.checked;
  const speed = 36;
  const bucketSize = 5;
  const trailMinutes = 12;
  const frameInterval = 1000 / 24;
  const segments = payload.segments || [];
  const buckets = Array.from({{length: Math.ceil((payload.maxT - payload.minT) / bucketSize) + 2}}, () => []);
  const starts = segments.map((segment, index) => [segment.t0, index]).sort((a, b) => a[0] - b[0]);
  segments.forEach((segment, index) => {{
    const startBucket = Math.max(0, Math.floor(segment.t0 / bucketSize));
    const endBucket = Math.min(buckets.length - 1, Math.floor(segment.t1 / bucketSize));
    for (let bucket = startBucket; bucket <= endBucket; bucket++) buckets[bucket].push(index);
  }});
  let bgCanvas = null;
  let bgCtx = null;

  function resize() {{
    dpr = 1;
    const rect = root.getBoundingClientRect();
    w = Math.max(320, rect.width);
    h = Math.max(520, rect.height);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fit = computeFit();
    bgCanvas = null;
    bgCtx = null;
  }}

  function merc(lon, lat) {{
    const y = Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI / 180) / 2)) * 180 / Math.PI;
    return [lon, y];
  }}

  function computeFit() {{
    const [x0, y0] = merc(lonMin, latMin);
    const [x1, y1] = merc(lonMax, latMax);
    const padX = 42, padY = 34;
    const scale = Math.min((w - padX * 2) / (x1 - x0), (h - padY * 2) / (y1 - y0));
    const mapW = (x1 - x0) * scale;
    const mapH = (y1 - y0) * scale;
    return {{x0, y0, scale, ox: (w - mapW) / 2, oy: (h + mapH) / 2}};
  }}

  function project(lon, lat) {{
    const [mx, my] = merc(lon, lat);
    const x = fit.ox + (mx - fit.x0) * fit.scale;
    const y = fit.oy - (my - fit.y0) * fit.scale;
    return [x, y];
  }}

  function segmentPosition(segment, t) {{
    const span = Math.max(0.001, segment.t1 - segment.t0);
    const u = Math.max(0, Math.min(1, (t - segment.t0) / span));
    const lon = segment.lon0 + (segment.lon1 - segment.lon0) * u;
    const lat = segment.lat0 + (segment.lat1 - segment.lat0) * u;
    const delay = segment.d0 + (segment.d1 - segment.d0) * u;
    return {{lon, lat, delay}};
  }}

  function activeSegmentIndices(t) {{
    const bucket = Math.max(0, Math.min(buckets.length - 1, Math.floor(t / bucketSize)));
    return buckets[bucket].filter(index => {{
      const segment = segments[index];
      return t >= segment.t0 && t < segment.t1;
    }});
  }}

  function nextSegmentStartAfter(t) {{
    let lo = 0, hi = starts.length;
    while (lo < hi) {{
      const mid = (lo + hi) >> 1;
      if (starts[mid][0] <= t) lo = mid + 1;
      else hi = mid;
    }}
    return lo < starts.length ? starts[lo][0] : null;
  }}

  function delayColor(delay, alpha) {{
    const [r, g, b] = classColor(delayClass(delay));
    return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
  }}

  function classColor(cls) {{
    if (cls === 0) return [245, 247, 255];
    if (cls === 15) return [246, 216, 223];
    if (cls === 30) return [248, 185, 191];
    if (cls === 45) return [249, 154, 159];
    if (cls === 75) return [251, 93, 96];
    if (cls === 90) return [253, 62, 64];
    if (cls === 105) return [254, 31, 32];
    return [255, 0, 0];
  }}

  function delayClass(delay) {{
    if (delay <= 0.5) return 0;
    if (delay < 30) return 15;
    if (delay < 45) return 30;
    if (delay < 75) return 45;
    if (delay < 90) return 75;
    if (delay < 105) return 90;
    if (delay < 120) return 105;
    return 120;
  }}

  function symbolRadius(delay) {{
    const cls = delayClass(delay);
    if (cls === 0) return 1.1;
    if (cls === 15) return 1.3;
    if (cls === 30) return 2.1;
    if (cls === 45) return 3.5;
    if (cls === 75) return 6.0;
    if (cls === 90) return 8.0;
    if (cls === 105) return 9.0;
    return 10.0;
  }}

  function trailWidth(delay) {{
    const cls = delayClass(delay);
    if (cls === 0) return 0.5;
    if (cls === 15) return 1.2;
    if (cls === 30) return 1.2;
    if (cls === 45) return 2.0;
    if (cls === 75) return 2.8;
    if (cls === 90) return 3.7;
    if (cls === 105) return 3.7;
    return 3.7;
  }}

  function formatSimTime(minutes) {{
    const [year, month, day, hour, minute] = payload.startParts;
    const value = new Date(Date.UTC(year, month - 1, day, hour, minute + Math.floor(minutes)));
    const yyyy = value.getUTCFullYear();
    const mm = String(value.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(value.getUTCDate()).padStart(2, "0");
    const hh = String(value.getUTCHours()).padStart(2, "0");
    const mi = String(value.getUTCMinutes()).padStart(2, "0");
    return `${{yyyy}}-${{mm}}-${{dd}} ${{hh}}:${{mi}}`;
  }}

  function drawBackground() {{
    if (!bgCanvas) {{
      bgCanvas = document.createElement("canvas");
      bgCanvas.width = canvas.width;
      bgCanvas.height = canvas.height;
      bgCtx = bgCanvas.getContext("2d");
      bgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      bgCtx.fillStyle = "#020609";
      bgCtx.fillRect(0, 0, w, h);

      bgCtx.fillStyle = "rgba(7,16,22,.68)";
      bgCtx.strokeStyle = "rgba(86,105,118,.62)";
      bgCtx.lineWidth = 1.15;
      for (const ring of payload.outline) {{
        if (!ring.length) continue;
        bgCtx.beginPath();
        ring.forEach(([lon, lat], i) => {{
          const [x, y] = project(lon, lat);
          if (i === 0) bgCtx.moveTo(x, y);
          else bgCtx.lineTo(x, y);
        }});
        bgCtx.closePath();
        bgCtx.fill();
        bgCtx.stroke();
      }}

      bgCtx.fillStyle = "rgba(82,112,130,.20)";
      for (let i = 0; i < 520; i++) {{
        const lon = lonMin + (((i * 37) % 1000) / 1000) * (lonMax - lonMin);
        const lat = latMin + (((i * 91) % 1000) / 1000) * (latMax - latMin);
        const [x, y] = project(lon, lat);
        bgCtx.fillRect(x, y, 1.1, 1.1);
      }}
      bgCtx.font = "13px system-ui, sans-serif";
      bgCtx.fillStyle = "rgba(228,236,242,.78)";
      for (const [name, lon, lat] of cities) {{
        const [x, y] = project(lon, lat);
        bgCtx.fillText(name, x + 6, y - 4);
        bgCtx.beginPath();
        bgCtx.arc(x, y, 2.2, 0, Math.PI * 2);
        bgCtx.fill();
      }}
    }}
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(bgCanvas, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  function draw(now) {{
    if (now - lastDraw < frameInterval) {{
      requestAnimationFrame(draw);
      return;
    }}
    lastDraw = now;
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    simT += dt * speed;
    if (simT > payload.maxT) {{
      simT = payload.minT;
    }}
    let activeIndices = activeSegmentIndices(simT);
    if (activeIndices.length === 0) {{
      const nextStart = nextSegmentStartAfter(simT);
      simT = nextStart === null ? payload.minT : nextStart;
      activeIndices = activeSegmentIndices(simT);
    }}

    drawBackground();

    function drawTrailSegment(a, b) {{
      const delay = Math.max(0, b.delay);
      ctx.strokeStyle = delayColor(delay, delay <= 0.5 ? 0.22 : 0.78);
      ctx.lineWidth = trailWidth(delay);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }}

    function drawHead(head) {{
      const delay = Math.max(0, head.delay);
      const r = symbolRadius(delay);
      ctx.fillStyle = delayColor(delay, delay <= 0.5 ? 0.58 : 0.92);
      ctx.beginPath();
      ctx.arc(head.x, head.y, r, 0, Math.PI * 2);
      ctx.fill();
      if (delay > 0.5) {{
        ctx.shadowColor = "rgba(255,40,40,.78)";
        if (delayClass(delay) >= 75) {{
          ctx.shadowBlur = delayClass(delay) >= 90 ? 24 : 17;
          ctx.beginPath();
          ctx.arc(head.x, head.y, r * 0.58, 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        }}
      }}
    }}

    const classes = [0, 15, 30, 45, 75, 90, 105, 120];
    const active = activeIndices.map(index => {{
      const segment = segments[index];
      const headGeo = segmentPosition(segment, simT);
      const tailGeo = segmentPosition(segment, Math.max(segment.t0, simT - trailMinutes));
      const [hx, hy] = project(headGeo.lon, headGeo.lat);
      const [tx, ty] = project(tailGeo.lon, tailGeo.lat);
      return {{
        cat: segment.cat || "regional",
        cls: delayClass(Math.max(0, headGeo.delay)),
        tail: {{x: tx, y: ty, delay: tailGeo.delay}},
        head: {{x: hx, y: hy, delay: headGeo.delay}},
      }};
    }});
    const visible = active.filter(item => {{
      const trainTypeVisible = (item.cat === "fern" && showFern) || (item.cat === "regional" && showRegional);
      return trainTypeVisible && (showUnder30 || item.cls >= 30);
    }});

    for (const cls of [45, 75, 90, 105, 120]) {{
      for (const item of visible) {{
        if (item.cls === cls) drawTrailSegment(item.tail, item.head);
      }}
    }}
    for (const cls of classes) {{
      for (const item of visible) {{
        if (item.cls === cls) drawHead(item.head);
      }}
    }}

    clock.textContent = formatSimTime(simT);
    count.textContent = `${{active.length}} active · ${{visible.length}} drawn · ${{payload.segmentCount}} segments`;
    requestAnimationFrame(draw);
  }}

  resize();
  window.addEventListener("resize", resize);
  fernToggle.addEventListener("change", () => {{
    showFern = fernToggle.checked;
  }});
  regionalToggle.addEventListener("change", () => {{
    showRegional = regionalToggle.checked;
  }});
  under30Toggle.addEventListener("change", () => {{
    showUnder30 = under30Toggle.checked;
  }});
  requestAnimationFrame(draw);
}})();
</script>
"""


def main() -> None:
    st.set_page_config(page_title="DB Delay Propagation", layout="wide")
    st.title("Deutsche Bahn delay propagation")

    with st.sidebar:
        selected_day = st.date_input("Day", value=date(2026, 7, 1), min_value=date(2024, 7, 1))

    view = st.radio(
        "View",
        ["Moving trains", "Diagnostics", "Propagation charts", "Train run", "Data"],
        horizontal=True,
    )
    needs_movement = view in {"Moving trains", "Diagnostics"}

    window_start = datetime.combine(selected_day, time.min)
    window_end = window_start + timedelta(days=2)
    needed_months = [(window_start.year, window_start.month)]
    if next_month_start(window_start) < window_end:
        needed_months.append((window_end.year, window_end.month))

    with st.spinner("Loading monthly Parquet from Hugging Face cache..."):
        parquet_paths = tuple(ensure_month_file(year, month) for year, month in needed_months)
    with st.spinner("Preparing station coordinate file..."):
        station_file = ensure_station_file()
    with st.spinner("Loading prepared 48h playback window..."):
        day_df, matched_station_count, prepared_source = load_prepared_window(
            parquet_paths, station_file, window_start, window_end
        )

    if day_df.is_empty():
        st.warning("No rows found for this day and filter.")
        return

    day_df = add_trip_instance(day_df).with_columns(pl.col("delay_in_min").abs().clip(1, 90).alias("delay_abs"))

    movement_segments: list[dict[str, object]] = []
    movement_stats: dict[str, int] = {}
    movement_issues = pl.DataFrame()
    movement_source = "not loaded"
    if needs_movement:
        with st.spinner("Building movement segments..."):
            movement_segments, movement_stats, movement_issues, movement_source = load_or_build_movement_segments(
                day_df, window_start, window_end
            )

    outline: list[list[list[float]]] = []
    if view == "Moving trains":
        with st.spinner("Loading Germany map outline..."):
            outline = load_germany_outline(ensure_germany_geojson())

    train_run_count = day_df["train_line_ride_id"].n_unique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in 48h window", f"{day_df.height:,}")
    c2.metric("Train runs", f"{train_run_count:,}")
    c3.metric("Max delay", f"{int(day_df['delay_in_min'].max())} min")
    c4.metric("Mapped stations", f"{matched_station_count:,}")
    st.caption(
        f"Playback window: {window_start:%Y-%m-%d %H:%M} -> {window_end:%Y-%m-%d %H:%M}; "
        f"speed: 36 simulated minutes/second; prepared data: {prepared_source}; movement: {movement_source}"
    )

    if view == "Moving trains":
        components.html(
            make_train_flow_animation(movement_segments, window_start, window_end, outline),
            height=800,
            scrolling=False,
        )
    elif view == "Diagnostics":
        with st.spinner("Building active segment audit..."):
            active_segment_audit = build_active_segment_audit(movement_segments, window_start, window_end)
            hourly_coverage = build_hourly_coverage(day_df, window_start, window_end)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rendered train runs", f"{movement_stats['rendered_train_runs']:,}")
        d2.metric("Rendered points", f"{movement_stats['rendered_points']:,}")
        d3.metric("Candidate train runs", f"{movement_stats['candidate_train_runs']:,}")
        d4.metric("Candidate rows", f"{movement_stats['candidate_rows']:,}")

        issue_rows = [
            {"issue": key, "count": movement_stats[key]}
            for key in [
                "non_monotonic_points",
                "short_segments",
                "long_segments",
                "implausible_speed_segments",
                "single_point_runs",
                "no_valid_segment_runs",
            ]
        ]
        st.dataframe(pl.DataFrame(issue_rows), use_container_width=True, hide_index=True)
        st.subheader("Hourly loaded data coverage")
        st.dataframe(hourly_coverage, use_container_width=True, hide_index=True)
        st.subheader("15-minute active segment audit")
        st.dataframe(active_segment_audit, use_container_width=True, hide_index=True)
        if movement_issues.is_empty():
            st.success("No segment-level timing issues found in the rendered movement data.")
        else:
            st.dataframe(movement_issues.head(500), use_container_width=True, hide_index=True)
    elif view == "Propagation charts":
        top_n = st.slider("Top delayed train runs", 5, 80, 30, 5)
        with st.spinner("Building chart summary..."):
            summary = build_trip_summary(day_df)
            top_trips = summary.head(top_n).with_row_index("first_order")
        st.plotly_chart(make_hourly_chart(day_df), use_container_width=True)
        st.plotly_chart(make_heatmap(day_df, top_trips), use_container_width=True)
    elif view == "Train run":
        with st.spinner("Building train run summary..."):
            summary = build_trip_summary(day_df)
        if summary.is_empty():
            st.warning("No train runs with at least three observed stops.")
            return
        run_days = summary.select("service_day").unique().sort("service_day")["service_day"].to_list()
        selected_run_day = st.selectbox(
            "Run day",
            run_days,
            format_func=lambda value: value.strftime("%Y-%m-%d"),
            key=f"train-run-day-{selected_day:%Y-%m-%d}",
        )
        day_summary = summary.filter(pl.col("service_day") == selected_run_day)
        option_rows = []
        option_labels = {}
        for r in day_summary.to_dicts():
            run_key = f"{r['service_day']:%Y-%m-%d}::{r['trip_instance']}::{r['train_line_ride_id']}"
            option_rows.append(
                {
                    "key": run_key,
                    "train_line_ride_id": r["train_line_ride_id"],
                    "service_day": r["service_day"],
                    "trip_instance": int(r["trip_instance"]),
                }
            )
            option_labels[run_key] = (
                f"{r['service_day']:%Y-%m-%d} #{r['trip_instance']} | "
                f"{r['train_type'] or ''} {r['train_number'] or ''} -> {r['destination'] or ''} | "
                f"max {r['max_delay']} min | {r['train_line_ride_id']}"
            )
        options = {row["key"]: row for row in option_rows}
        selected_key = st.selectbox(
            "Inspect one train run",
            [row["key"] for row in option_rows],
            format_func=lambda value: option_labels[value],
            key=f"train-run-choice-v2-{selected_run_day:%Y-%m-%d}",
        )
        selected_run = options[selected_key]
        selected_ride_id = selected_run["train_line_ride_id"]
        selected_service_day = selected_run["service_day"]
        selected_trip_instance = selected_run["trip_instance"]
        st.caption(f"Selected: {option_labels[selected_key]}")
        chart_key = f"{selected_service_day:%Y-%m-%d}-{selected_trip_instance}-{selected_ride_id}"
        st.plotly_chart(
            make_trip_map(day_df, selected_ride_id, selected_service_day, selected_trip_instance),
            use_container_width=True,
            key=f"train-run-map-{chart_key}",
        )
        st.plotly_chart(
            make_trip_line(day_df, selected_ride_id, selected_service_day, selected_trip_instance),
            use_container_width=True,
            key=f"train-run-line-{chart_key}",
        )
    elif view == "Data":
        rows_shown = st.slider("Rows shown", 20, 500, 100, 20)
        with st.spinner("Building data summary..."):
            summary = build_trip_summary(day_df)
            top_trips = summary.head(rows_shown).with_row_index("first_order")
        st.dataframe(
            top_trips.select(
                [
                    "service_day",
                    "trip_instance",
                    "train_type",
                    "train_number",
                    "line_number",
                    "destination",
                    "first_seen",
                    "last_seen",
                    "station_count",
                    "max_delay",
                    "mean_delay",
                    "arrival_cancelled",
                    "departure_cancelled",
                    "train_line_ride_id",
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
