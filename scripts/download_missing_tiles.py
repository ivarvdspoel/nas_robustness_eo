#!/usr/bin/env python3
from pathlib import Path
import argparse
import time
import urllib.request
import urllib.error

import pandas as pd


BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"


def read_missing_tiles(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing tile list not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        if "worldcover_tile" not in df.columns:
            raise ValueError(f"CSV must contain column 'worldcover_tile': {path}")

        tiles = df["worldcover_tile"].dropna().astype(str).tolist()
    else:
        tiles = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    tiles = sorted(set(tiles))
    return tiles


def download_tile(tile, out_dir, overwrite=False, retries=3, sleep_sec=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / tile
    url = f"{BASE_URL}/{tile}"

    if dst.exists() and dst.stat().st_size > 0 and not overwrite:
        return "exists", dst

    tmp = dst.with_suffix(dst.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()

            urllib.request.urlretrieve(url, tmp)

            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError("empty download")

            tmp.replace(dst)
            return "downloaded", dst

        except Exception as e:
            if tmp.exists():
                tmp.unlink()

            if attempt == retries:
                return f"failed: {type(e).__name__}: {e}", dst

            time.sleep(sleep_sec)

    return "failed", dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--missing-tiles",
        required=True,
        help="Path to missing_worldcover_tiles.txt or missing_worldcover_tiles.csv",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where WorldCover .tif tiles should be stored.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=2.0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    tiles = read_missing_tiles(args.missing_tiles)

    print("=" * 100)
    print("Download missing ESA WorldCover tiles")
    print("Missing tile list:", args.missing_tiles)
    print("Output tile dir:", out_dir)
    print("N tiles:", len(tiles))
    print("=" * 100)

    downloaded = []
    existing = []
    failed = []

    for i, tile in enumerate(tiles, 1):
        status, dst = download_tile(
            tile=tile,
            out_dir=out_dir,
            overwrite=args.overwrite,
            retries=args.retries,
            sleep_sec=args.sleep_sec,
        )

        print(f"[{i}/{len(tiles)}] {status} {tile}")

        if status == "downloaded":
            downloaded.append(tile)
        elif status == "exists":
            existing.append(tile)
        else:
            failed.append({"worldcover_tile": tile, "status": status})

    (out_dir / "downloaded_missing_tiles.txt").write_text(
        "\n".join(downloaded) + ("\n" if downloaded else ""),
        encoding="utf-8",
    )

    (out_dir / "existing_missing_tiles.txt").write_text(
        "\n".join(existing) + ("\n" if existing else ""),
        encoding="utf-8",
    )

    failed_path = out_dir / "failed_missing_tile_downloads.csv"
    pd.DataFrame(failed).to_csv(failed_path, index=False)

    print()
    print("Done.")
    print("Downloaded:", len(downloaded))
    print("Already existed:", len(existing))
    print("Failed:", len(failed))
    print("Failed log:", failed_path)


if __name__ == "__main__":
    main()