#!/usr/bin/env python3
"""Build 128x128 Sentinel-2 image patches and 4-class WorldCover masks from a CSV of patch corners.

Example:
  python build_s2_worldcover_128.py \
    --csv /shared/home/ivanderspoel/worldcover_handoff/worldcover_handoff/outputs/worldcover_queries/balanced_4class_dataset.csv \
    --s2-dir /shared/projects/phisat2/data/interim/s2b_croped \
    --worldcover-dir /shared/home/ivanderspoel/worldcover_handoff/worldcover_handoff/cache/worldcover/all_trusted/tiles \
    --out-dir /shared/home/ivanderspoel/worldcover_handoff/worldcover_handoff/outputs/s2_worldcover_128 \
    -N 100

Outputs:
  <out-dir>/s2_images.npy     float32, shape (n, bands, 128, 128)
  <out-dir>/s2_masks.npy      uint8,   shape (n, 128, 128), values 0..3
  <out-dir>/manifest.csv      one row per selected CSV row, including status/error/output_index
  <out-dir>/quicklooks/*.png  only when -N > 0, unless --quicklooks is set explicitly
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# WorldCover -> task class mapping requested in the handoff.
TASK_CLASSES = {
    0: {0, 60, 70},                  # background: no data, bare/sparse, snow/ice
    1: {10, 20, 30, 40, 90, 95, 100}, # vegetation
    2: {50},                         # built-up
    3: {80},                         # water
}
TASK_CLASS_NAMES = {
    0: "background",
    1: "vegetation",
    2: "built_up",
    3: "water",
}

WC_CLASS_NAMES = {
    0: "No data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

TASK_COLORS = ["#666666", "#2ca02c", "#d62728", "#1f77b4"]
TASK_CMAP = ListedColormap(TASK_COLORS)
TASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], TASK_CMAP.N)


def tile_name_from_lat_lon(lat: float, lon: float) -> str:
    """Return ESA WorldCover 3x3-degree tile name covering a lat/lon point."""
    lat0 = math.floor(float(lat) / 3) * 3
    lon0 = math.floor(float(lon) / 3) * 3
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"


def patch_affine_from_corners(row: pd.Series, width: int, height: int) -> Affine:
    """Build an EPSG:4326 affine transform from the CSV quadrilateral corners.

    The transform allows rotation/shear and maps output pixel coordinates into lon/lat.
    It uses UL, UR, and LL corners, matching the prior handoff script.
    """
    ul_lon, ul_lat = float(row["ul_lon"]), float(row["ul_lat"])
    ur_lon, ur_lat = float(row["ur_lon"]), float(row["ur_lat"])
    ll_lon, ll_lat = float(row["ll_lon"]), float(row["ll_lat"])

    a = (ur_lon - ul_lon) / width
    d = (ur_lat - ul_lat) / width
    b = (ll_lon - ul_lon) / height
    e = (ll_lat - ul_lat) / height
    c = ul_lon
    f = ul_lat
    return Affine(a, b, c, d, e, f)


def parse_bands(bands_arg: str | None, src_count: int) -> list[int]:
    """Parse 1-based raster band numbers for rasterio."""
    if not bands_arg:
        return list(range(1, src_count + 1))
    bands = [int(x.strip()) for x in bands_arg.split(",") if x.strip()]
    bad = [b for b in bands if b < 1 or b > src_count]
    if bad:
        raise ValueError(f"requested bands {bad} outside available range 1..{src_count}")
    return bands


def s2_path_for_product(s2_dir: Path, pattern: str, product_id: int) -> Path:
    return s2_dir / pattern.format(product_id=product_id)


def read_s2_patch(
    src: rasterio.io.DatasetReader,
    row: pd.Series,
    bands: Iterable[int],
    shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reproject a Sentinel-2 GeoTIFF into the patch grid as (bands, H, W) float32."""
    height, width = shape
    dst_transform = patch_affine_from_corners(row, width=width, height=height)
    dst = np.zeros((len(list(bands)), height, width), dtype="float32")

    src_nodata = src.nodata
    for out_i, band_i in enumerate(bands):
        reproject(
            source=rasterio.band(src, band_i),
            destination=dst[out_i],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=resampling,
        )
    return dst


def read_worldcover_patch(
    src: rasterio.io.DatasetReader,
    row: pd.Series,
    shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour WorldCover patch in the same patch grid."""
    height, width = shape
    dst_transform = patch_affine_from_corners(row, width=width, height=height)
    dst = np.zeros((height, width), dtype="uint8")
    reproject(
        source=rasterio.band(src, 1),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return dst


def remap_worldcover_to_task(wc: np.ndarray) -> np.ndarray:
    """Map ESA WorldCover codes to the requested 4 land-cover task classes."""
    out = np.zeros(wc.shape, dtype="uint8")
    for task_id, wc_ids in TASK_CLASSES.items():
        out[np.isin(wc, list(wc_ids))] = task_id
    return out


def robust_stretch(x: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    x = x.astype("float32", copy=False)
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return np.zeros_like(x, dtype="float32")
    lo, hi = np.percentile(x[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype="float32")
    return np.clip((x - lo) / (hi - lo), 0, 1)


def s2_rgb_for_quicklook(s2_patch: np.ndarray, rgb_bands_1based: list[int]) -> np.ndarray:
    """Create an RGB quicklook from a (C,H,W) patch using 1-based indices in the selected output stack."""
    max_idx = s2_patch.shape[0]
    idx = []
    for b in rgb_bands_1based:
        # Convert quicklook band numbers to zero-based indices in the written stack.
        idx.append(min(max(b, 1), max_idx) - 1)
    return np.stack([robust_stretch(s2_patch[i]) for i in idx], axis=-1)


def overlay_mask_on_rgb(rgb: np.ndarray, mask4: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_rgb = TASK_CMAP(TASK_NORM(mask4))[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * mask_rgb, 0, 1)


def task_fractions(mask4: np.ndarray) -> dict[str, float | int | str]:
    n = float(mask4.size)
    out: dict[str, float | int | str] = {}
    for cls, name in TASK_CLASS_NAMES.items():
        out[f"task_frac_{name}"] = float((mask4 == cls).sum() / n)
    vals, counts = np.unique(mask4, return_counts=True)
    dom = int(vals[np.argmax(counts)])
    out["task_dominant_class"] = TASK_CLASS_NAMES[dom]
    out["task_dominant_id"] = dom
    out["task_dominant_frac"] = float(counts.max() / n)
    out["task_total_frac"] = float(sum(out[f"task_frac_{name}"] for name in TASK_CLASS_NAMES.values()))
    return out


def make_quicklook(
    s2_patch: np.ndarray,
    wc_raw: np.ndarray,
    mask4: np.ndarray,
    row: pd.Series,
    out_path: Path,
    rgb_bands: list[int],
) -> None:
    rgb = s2_rgb_for_quicklook(s2_patch, rgb_bands)
    over = overlay_mask_on_rgb(rgb, mask4)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(wc_raw, cmap="tab20", interpolation="nearest")
    axes[1].set_title("WorldCover raw")
    axes[2].imshow(mask4, cmap=TASK_CMAP, norm=TASK_NORM, interpolation="nearest")
    axes[2].set_title("WorldCover 4-class")
    axes[3].imshow(over)
    axes[3].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    lines = [
        f"patch_index={int(row['patch_index'])}",
        f"product_id={int(row['product_id'])}",
        f"lat={float(row['center_lat']):.5f}, lon={float(row['center_lon']):.5f}",
    ]
    fracs = task_fractions(mask4)
    lines += [f"{name}: {100 * float(fracs[f'task_frac_{name}']):.1f}%" for name in TASK_CLASS_NAMES.values()]
    fig.suptitle(" | ".join(lines[:3]) + "\n" + "   ".join(lines[3:]), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def infer_band_count(df: pd.DataFrame, s2_dir: Path, pattern: str) -> int:
    last_error = None
    for _, row in df.iterrows():
        path = s2_path_for_product(s2_dir, pattern, int(row["product_id"]))
        try:
            with rasterio.open(path) as src:
                return src.count
        except Exception as e:  # keep trying; some products might be missing
            last_error = e
    raise RuntimeError(f"Could not open any Sentinel-2 file to infer band count. Last error: {last_error}")


def resolve_worldcover_tile(row: pd.Series, wc_dir: Path) -> tuple[str, Path]:
    if "worldcover_tile" in row and pd.notna(row["worldcover_tile"]) and str(row["worldcover_tile"]).strip():
        tname = Path(str(row["worldcover_tile"])).name
    else:
        tname = tile_name_from_lat_lon(float(row["center_lat"]), float(row["center_lon"]))
    return tname, wc_dir / tname


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv", required=True, help="Input CSV with patch bounding-box/corner columns.")
    parser.add_argument("--s2-dir", required=True, help="Directory containing Sentinel-2 cropped GeoTIFFs.")
    parser.add_argument("--s2-pattern", default="{product_id}_s2b_cropped.tif", help="Filename pattern relative to --s2-dir.")
    parser.add_argument("--worldcover-dir", required=True, help="Directory containing ESA WorldCover tile GeoTIFFs.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("-N", "--n", type=int, default=0, help="Number of examples. 0 means all rows and disables quicklooks by default.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --sample is used.")
    parser.add_argument("--sample", action="store_true", help="Randomly sample N rows instead of taking the first N.")
    parser.add_argument("--size", type=int, default=128, help="Output patch size in pixels; creates size x size patches.")
    parser.add_argument("--bands", default=None, help="Comma-separated 1-based Sentinel-2 raster bands to write. Default: all bands.")
    parser.add_argument("--rgb-bands", default="3,2,1", help="1-based indices in the written S2 stack for quicklook RGB.")
    parser.add_argument("--quicklooks", type=int, default=None, help="Number of quicklooks. Default: N if N>0 else 0.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    s2_dir = Path(args.s2_dir)
    wc_dir = Path(args.worldcover_dir)
    out_dir = Path(args.out_dir)
    ql_dir = out_dir / "quicklooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = [
        "patch_index", "product_id", "center_lat", "center_lon",
        "ul_lat", "ul_lon", "ur_lat", "ur_lon", "ll_lat", "ll_lon",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required columns: {missing_cols}")

    df = df.dropna(subset=["ul_lat", "ul_lon", "ur_lat", "ur_lon", "ll_lat", "ll_lon"]).copy()
    if args.n and args.n > 0:
        if args.sample:
            df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
        else:
            df = df.iloc[:args.n]
    df = df.reset_index(drop=True)

    quicklooks = args.quicklooks if args.quicklooks is not None else (len(df) if args.n and args.n > 0 else 0)
    if quicklooks > 0:
        ql_dir.mkdir(parents=True, exist_ok=True)

    src_band_count = infer_band_count(df, s2_dir, args.s2_pattern)
    bands = parse_bands(args.bands, src_band_count)
    rgb_bands = [int(x.strip()) for x in args.rgb_bands.split(",") if x.strip()]
    shape = (args.size, args.size)

    images_path = out_dir / "s2_images.npy"
    masks_path = out_dir / "s2_masks.npy"
    images = np.lib.format.open_memmap(images_path, mode="w+", dtype="float32", shape=(len(df), len(bands), args.size, args.size))
    masks = np.lib.format.open_memmap(masks_path, mode="w+", dtype="uint8", shape=(len(df), args.size, args.size))

    manifest_rows: list[dict] = []
    s2_cache: dict[Path, rasterio.io.DatasetReader] = {}
    wc_cache: dict[Path, rasterio.io.DatasetReader] = {}

    print("=" * 100)
    print("S2 + WorldCover 128x128 builder")
    print("CSV:", csv_path)
    print("Rows selected:", len(df))
    print("S2 dir:", s2_dir)
    print("S2 pattern:", args.s2_pattern)
    print("S2 bands written:", bands)
    print("WorldCover dir:", wc_dir)
    print("Output:", out_dir)
    print("Quicklooks:", quicklooks)
    print("=" * 100)

    try:
        for i, row in df.iterrows():
            row_out = row.to_dict()
            patch_index = int(row["patch_index"])
            product_id = int(row["product_id"])
            s2_path = s2_path_for_product(s2_dir, args.s2_pattern, product_id)
            tname, wc_path = resolve_worldcover_tile(row, wc_dir)

            row_out.update({
                "output_index": i,
                "s2_path": str(s2_path),
                "worldcover_tile": tname,
                "worldcover_path": str(wc_path),
                "status": "ok",
                "error": "",
                "quicklook_path": "",
            })

            try:
                if not s2_path.exists():
                    raise FileNotFoundError(f"missing Sentinel-2 file: {s2_path}")
                if not wc_path.exists():
                    raise FileNotFoundError(f"missing WorldCover tile: {wc_path}")

                if s2_path not in s2_cache:
                    s2_cache[s2_path] = rasterio.open(s2_path)
                if wc_path not in wc_cache:
                    wc_cache[wc_path] = rasterio.open(wc_path)

                s2_patch = read_s2_patch(s2_cache[s2_path], row, bands=bands, shape=shape)
                wc_raw = read_worldcover_patch(wc_cache[wc_path], row, shape=shape)
                mask4 = remap_worldcover_to_task(wc_raw)

                images[i] = s2_patch
                masks[i] = mask4
                row_out.update(task_fractions(mask4))

                # Preserve raw WorldCover fractions for sanity checks.
                for wc_id, wc_name in WC_CLASS_NAMES.items():
                    row_out[f"raw_wc_frac_{wc_id}"] = float((wc_raw == wc_id).sum() / wc_raw.size)

                if i < quicklooks:
                    ql_path = ql_dir / f"s2_worldcover_patch{patch_index}_product{product_id}.png"
                    make_quicklook(s2_patch, wc_raw, mask4, row, ql_path, rgb_bands=rgb_bands)
                    row_out["quicklook_path"] = str(ql_path)

            except Exception as e:
                images[i] = 0
                masks[i] = 0
                row_out["status"] = "failed"
                row_out["error"] = f"{type(e).__name__}: {e}"
                print(f"[failed] row={i} patch={patch_index} product={product_id}: {row_out['error']}")

            manifest_rows.append(row_out)
            if (i + 1) % 100 == 0 or (i + 1) == len(df):
                print(f"[{i + 1}/{len(df)}]")
    finally:
        # Flush memmaps and close open rasters.
        images.flush()
        masks.flush()
        for src in s2_cache.values():
            src.close()
        for src in wc_cache.values():
            src.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    ok = manifest[manifest["status"] == "ok"].copy()
    summary = {
        "csv": str(csv_path),
        "s2_dir": str(s2_dir),
        "s2_pattern": args.s2_pattern,
        "worldcover_dir": str(wc_dir),
        "out_dir": str(out_dir),
        "n_selected": int(len(df)),
        "n_ok": int((manifest["status"] == "ok").sum()),
        "n_failed": int((manifest["status"] == "failed").sum()),
        "patch_size": args.size,
        "s2_bands_written_1based": bands,
        "s2_images": str(images_path),
        "s2_masks": str(masks_path),
        "manifest": str(manifest_path),
        "task_classes": {str(k): sorted(v) for k, v in TASK_CLASSES.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nStatus counts:")
    print(manifest["status"].value_counts(dropna=False).to_string())
    if len(ok):
        print("\nMean task fractions over successful patches:")
        for name in TASK_CLASS_NAMES.values():
            print(f"  {name:>10s}: {ok[f'task_frac_{name}'].mean():.6f}")
    print("\nSaved:")
    print("  images:", images_path)
    print("  masks: ", masks_path)
    print("  manifest:", manifest_path)
    print("  summary: ", out_dir / "summary.json")
    if quicklooks > 0:
        print("  quicklooks:", ql_dir)


if __name__ == "__main__":
    main()
