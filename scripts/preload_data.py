from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

import requests


BASE_URL = os.getenv(
    "HF_DATASET_BASE_URL",
    "https://huggingface.co/datasets/piebro/deutsche-bahn-data/resolve/main/monthly_processed_data",
)
CACHE_DIR = Path(os.getenv("DATA_CACHE_DIR", "/app/data/cache"))
STATION_TGZ_URL = "https://registry.npmjs.org/db-hafas-stations/-/db-hafas-stations-2.0.0.tgz"
GERMANY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/isellsoap/deutschlandGeoJSON/main/2_bundeslaender/1_sehr_hoch.geo.json"
)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"cached {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url} -> {path}")
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(path)
    print(f"stored {path} ({path.stat().st_size:,} bytes)")


def preload_month(month: str) -> None:
    year_text, month_text = month.split("-", maxsplit=1)
    filename = f"data-{int(year_text):04d}-{int(month_text):02d}.parquet"
    download(f"{BASE_URL}/{filename}", CACHE_DIR / filename)


def preload_stations() -> None:
    ndjson_path = CACHE_DIR / "db-hafas-stations-full.ndjson"
    if ndjson_path.exists() and ndjson_path.stat().st_size > 0:
        print(f"cached {ndjson_path}")
        return

    tgz_path = CACHE_DIR / "db-hafas-stations.tgz"
    download(STATION_TGZ_URL, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as archive:
        member = archive.getmember("package/full.ndjson")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError("package/full.ndjson not found in station package")
        with ndjson_path.open("wb") as handle:
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                handle.write(chunk)
    print(f"stored {ndjson_path} ({ndjson_path.stat().st_size:,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="2026-05,2026-06,2026-07")
    parser.add_argument("--include-stations", action="store_true")
    parser.add_argument("--include-geojson", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for month in [part.strip() for part in args.months.split(",") if part.strip()]:
        preload_month(month)

    if args.include_stations:
        preload_stations()

    if args.include_geojson:
        download(GERMANY_GEOJSON_URL, CACHE_DIR / "germany-states.geo.json")


if __name__ == "__main__":
    main()
