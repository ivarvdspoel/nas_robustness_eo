from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from affine import Affine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


WC_CLASSES = {
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

WC_COLORS = {
    0: "#000000",
    10: "#006400",
    20: "#ffbb22",
    30: "#ffff4c",
    40: "#f096ff",
    50: "#fa0000",
    60: "#b4b4b4",
    70: "#f0f0f0",
    80: "#0064c8",
    90: "#0096a0",
    95: "#00cf75",
    100: "#fae6a0",
}

CLASS_ORDER = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

REAL_RGB = (3, 2, 1)
SIM_RGB = (3, 2, 1)
S2_RGB = (2, 1, 0)


def tile_name(lat, lon):
    lat0 = math.floor(float(lat) / 3) * 3
    lon0 = math.floor(float(lon) / 3) * 3

    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"

    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}_Map.tif"


def robust_stretch(x, p_low=2, p_high=98):
    x = x.astype("float32")
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return np.zeros_like(x, dtype="float32")
    lo, hi = np.percentile(x[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype="float32")
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def rgb_from_stack(stack, bands):
    return np.stack([robust_stretch(stack[b]) for b in bands], axis=-1)


def patch_affine_from_corners(row, width=256, height=256):
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


def read_worldcover_patch(src, row, shape=(256, 256)):
    dst = np.zeros(shape, dtype="uint8")
    dst_transform = patch_affine_from_corners(row, width=shape[1], height=shape[0])

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


def class_fractions(mask):
    n = mask.size
    out = {}
    for cls in CLASS_ORDER:
        out[f"wc_frac_{cls}"] = float((mask == cls).sum() / n)

    vals, counts = np.unique(mask, return_counts=True)
    if len(vals) == 0:
        out["wc_dominant_class"] = 0
        out["wc_dominant_name"] = "No data"
        out["wc_dominant_frac"] = 0.0
    else:
        k = int(vals[np.argmax(counts)])
        out["wc_dominant_class"] = k
        out["wc_dominant_name"] = WC_CLASSES.get(k, f"Unknown {k}")
        out["wc_dominant_frac"] = float(counts.max() / n)

    out["wc_nodata_frac"] = out.get("wc_frac_0", 0.0)
    return out


def worldcover_rgb(mask):
    color_list = [WC_COLORS[k] for k in CLASS_ORDER]
    cmap = ListedColormap(color_list)
    boundaries = [k - 0.5 for k in CLASS_ORDER] + [CLASS_ORDER[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap(norm(mask))


def overlay_rgb(real_rgb, wc_mask, alpha=0.45):
    wc_rgba = worldcover_rgb(wc_mask)
    wc_rgb = wc_rgba[..., :3]
    valid = wc_mask != 0
    out = real_rgb.copy()
    out[valid] = (1 - alpha) * real_rgb[valid] + alpha * wc_rgb[valid]
    return np.clip(out, 0, 1)


def make_quicklook(real, sim, s2, wc, row, out_path):
    real_rgb = rgb_from_stack(real, REAL_RGB)
    sim_rgb = rgb_from_stack(sim, SIM_RGB)
    s2_rgb = rgb_from_stack(s2, S2_RGB)
    wc_rgba = worldcover_rgb(wc)
    over = overlay_rgb(real_rgb, wc)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    ax = axes.ravel()

    ax[0].imshow(real_rgb)
    ax[0].set_title("Real ΦSat-2 RGB")

    ax[1].imshow(sim_rgb)
    ax[1].set_title("Simulated ΦSat-2 RGB")

    ax[2].imshow(s2_rgb)
    ax[2].set_title("Aligned Sentinel-2 RGB")

    ax[3].imshow(wc_rgba)
    ax[3].set_title("WorldCover labels")

    ax[4].imshow(over)
    ax[4].set_title("WorldCover overlay on real ΦSat-2")

    ax[5].axis("off")
    classes_present = []
    for cls in CLASS_ORDER:
        frac = float((wc == cls).sum() / wc.size)
        if frac >= 0.02:
            classes_present.append((cls, frac))

    lines = [
        f"patch_index: {int(row['patch_index'])}",
        f"product_id: {int(row['product_id'])}",
        f"tier: {row.get('transfer_tier_v1', 'NA')}",
        f"score: {float(row.get('transfer_quality_score_v0', np.nan)):.3f}",
        f"dominant: {WC_CLASSES.get(int(row.get('wc_dominant_class', 0)), 'NA')}",
        "",
        "Classes >= 2%:",
    ]

    for cls, frac in sorted(classes_present, key=lambda x: -x[1])[:8]:
        lines.append(f"{cls:>3} {WC_CLASSES.get(cls, 'Unknown')}: {100*frac:.1f}%")

    ax[5].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    for a in ax[:5]:
        a.axis("off")

    title = (
        f"WorldCover pilot | patch={int(row['patch_index'])} | "
        f"product={int(row['product_id'])} | "
        f"lat={float(row['center_lat']):.4f}, lon={float(row['center_lon']):.4f}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worldcover-dir", required=True, help="Directory containing WorldCover GeoTIFF tiles.")
    parser.add_argument("--h5-path", default="/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5")
    parser.add_argument("--trusted", default="cache/worldcover/pilot_200/trusted_patches_worldcover_pilot_200.csv")
    parser.add_argument("--patch-meta", default="outputs/triplets_v1_patch_audit/patch_manifest_v1_clean_product_ids.csv")
    parser.add_argument("--out-dir", default="outputs/worldcover_pilot_v1")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quicklooks", type=int, default=80)
    args = parser.parse_args()

    wc_dir = Path(args.worldcover_dir)
    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "masks_npy"
    ql_dir = out_dir / "quicklooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    ql_dir.mkdir(exist_ok=True)

    trusted = pd.read_csv(args.trusted)

    # Minimal-handoff mode:
    # trusted_patches_v1.csv identifies which patch_index values are reliable,
    # but it may only contain patch centers. The exact patch corners are then
    # read directly from the HDF5 metadata.
    required_geo_cols = [
        "ul_lat", "ul_lon", "ur_lat", "ur_lon", "lr_lat", "lr_lon", "ll_lat", "ll_lon",
    ]

    if all(c in trusted.columns for c in required_geo_cols):
        df = trusted.copy()
        print("Using full geographic metadata directly from trusted CSV.")
    else:
        print("Trusted CSV does not contain patch corners.")
        print("Reading patch geographic metadata directly from HDF5:", args.h5_path)

        patch_indices = trusted["patch_index"].astype(int).to_numpy()

        sort_order = np.argsort(patch_indices)
        patch_indices_sorted = patch_indices[sort_order]
        unsort_order = np.argsort(sort_order)

        df = trusted.copy()

        h5_meta_cols = [
            "product_id",
            "center_lat", "center_lon",
            "ul_lat", "ul_lon", "ur_lat", "ur_lon",
            "lr_lat", "lr_lon", "ll_lat", "ll_lon",
            "date_phi", "date_s2b", "koppen_zone",
        ]

        with h5py.File(args.h5_path, "r") as h5:
            for col in h5_meta_cols:
                if col in df.columns:
                    continue

                h5_key = f"metadata/{col}"
                if h5_key not in h5:
                    print(f"[warn] missing HDF5 metadata field: {h5_key}")
                    continue

                arr = h5[h5_key][patch_indices_sorted]
                arr = arr[unsort_order]

                if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
                    arr = [
                        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in arr
                    ]

                df[col] = arr

    missing = df[["ul_lat", "ul_lon", "ur_lat", "ur_lon", "ll_lat", "ll_lon"]].isna().any(axis=1)
    n_missing = int(missing.sum())
    if n_missing:
        print(f"[warn] dropping {n_missing} patches with missing geographic corners")
    df = df[~missing].copy()

    if args.n and args.n > 0 and args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).sort_values("patch_index").reset_index(drop=True)
    else:
        df = df.sort_values("patch_index").reset_index(drop=True)

    df = df[df["patch_index"] == 390].copy().reset_index(drop=True)

    print("=" * 100)
    print("WorldCover pilot v1 tile-dir")
    print("WorldCover dir:", wc_dir)
    print("HDF5:", args.h5_path)
    print("Trusted manifest:", args.trusted)
    print("Selected patches:", len(df))
    print("Output:", out_dir)
    print("=" * 100)

    labels_h5 = out_dir / "worldcover_pilot_labels_v1.h5"
    manifest_rows = []
    src_cache = {}

    with h5py.File(args.h5_path, "r") as h5, h5py.File(labels_h5, "w") as out_h5:
        ds_labels = out_h5.create_dataset(
            "worldcover/labels",
            shape=(len(df), 256, 256),
            dtype="uint8",
            chunks=(1, 256, 256),
            compression="lzf",
        )
        out_h5.create_dataset("metadata/patch_index", data=df["patch_index"].to_numpy(dtype="int64"))
        out_h5.create_dataset("metadata/product_id", data=df["product_id"].to_numpy(dtype="int64"))

        for i, row in df.iterrows():
            patch_index = int(row["patch_index"])
            product_id = int(row["product_id"])
            tname = tile_name(row["center_lat"], row["center_lon"])
            tpath = wc_dir / tname

            try:
                if not tpath.exists():
                    raise FileNotFoundError(f"missing WorldCover tile: {tpath}")

                if tname not in src_cache:
                    src_cache[tname] = rasterio.open(tpath)

                wc = read_worldcover_patch(src_cache[tname], row)
                ds_labels[i] = wc

                npy_path = mask_dir / f"worldcover_patch{patch_index}_product{product_id}.npy"
                np.save(npy_path, wc)

                stats = class_fractions(wc)
                row_out = row.to_dict()
                row_out.update(stats)
                row_out["worldcover_tile"] = tname
                row_out["label_h5_index"] = i
                row_out["label_h5_path"] = str(labels_h5)
                row_out["mask_npy_path"] = str(npy_path)
                row_out["status"] = "ok"
                row_out["error"] = ""

                if i < args.quicklooks:
                    real = h5["real/images"][patch_index].astype("float32")
                    sim = h5["sim/images"][patch_index].astype("float32")
                    s2 = h5["s2b/images"][patch_index].astype("float32")

                    ql_path = ql_dir / f"worldcover_pilot_patch{patch_index}_product{product_id}.png"
                    qrow = pd.Series(row_out)
                    make_quicklook(real, sim, s2, wc, qrow, ql_path)
                    row_out["quicklook_path"] = str(ql_path)
                else:
                    row_out["quicklook_path"] = ""

            except Exception as e:
                row_out = row.to_dict()
                row_out["worldcover_tile"] = tname
                row_out["label_h5_index"] = -1
                row_out["status"] = "failed"
                row_out["error"] = f"{type(e).__name__}: {e}"
                row_out["quicklook_path"] = ""
                print("[failed]", patch_index, product_id, row_out["error"])

            manifest_rows.append(row_out)

            if (i + 1) % 25 == 0 or (i + 1) == len(df):
                print(f"[{i+1}/{len(df)}]")

    for src in src_cache.values():
        src.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "worldcover_pilot_manifest_v1.csv"
    manifest.to_csv(manifest_path, index=False)

    ok = manifest[manifest["status"] == "ok"].copy()

    class_rows = []
    for cls in CLASS_ORDER:
        col = f"wc_frac_{cls}"
        if col in ok.columns:
            class_rows.append({
                "class_id": cls,
                "class_name": WC_CLASSES.get(cls, f"Unknown {cls}"),
                "mean_patch_fraction": float(ok[col].mean()),
                "median_patch_fraction": float(ok[col].median()),
                "patches_with_class_gt_1pct": int((ok[col] > 0.01).sum()),
                "patches_with_class_gt_10pct": int((ok[col] > 0.10).sum()),
            })
    class_summary = pd.DataFrame(class_rows)
    class_summary.to_csv(out_dir / "worldcover_class_summary_v1.csv", index=False)

    dominant_summary = (
        ok.groupby(["wc_dominant_class", "wc_dominant_name"])
        .size()
        .reset_index(name="n_patches")
        .sort_values("n_patches", ascending=False)
    )
    dominant_summary.to_csv(out_dir / "worldcover_dominant_class_summary_v1.csv", index=False)

    summary = {
        "n_selected": int(len(df)),
        "n_ok": int((manifest["status"] == "ok").sum()),
        "n_failed": int((manifest["status"] == "failed").sum()),
        "worldcover_dir": str(wc_dir),
        "labels_h5": str(labels_h5),
        "manifest": str(manifest_path),
        "quicklook_dir": str(ql_dir),
    }
    (out_dir / "worldcover_pilot_summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Status counts:")
    print(manifest["status"].value_counts(dropna=False))
    print()
    print("Dominant classes:")
    print(dominant_summary.to_string(index=False))
    print()
    print("Saved manifest:", manifest_path)
    print("Saved labels:", labels_h5)
    print("Saved quicklooks:", ql_dir)


if __name__ == "__main__":
    main()
