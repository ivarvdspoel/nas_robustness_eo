#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from typing import Optional


BAND_NAMES = {
    0: "PAN",
    1: "Blue",
    2: "Green",
    3: "Red",
    4: "Red Edge 1",
    5: "Red Edge 2",
    6: "Red Edge 3",
    7: "NIR",
}


BAND_MEAN = np.array(
    [15.0381, 14.5305, 14.4030, 15.4191, 13.6231, 14.2143, 14.7041, 13.1745],
    dtype=np.float32,
)

BAND_STD = np.array(
    [8.2196, 10.6197, 9.4811, 9.0923, 10.5712, 10.4277, 10.3784, 9.7216],
    dtype=np.float32,
)

# Replace with your exact value if different.
SQRT_CLIP = 40.0


def normalize_phisat2(arr: np.ndarray, sqrt_clip: float) -> np.ndarray:
    """
    Apply PhiSat-2 preprocessing to a full tensor of shape (8, H, W).
    """
    arr = arr.astype(np.float32, copy=False)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = np.sqrt(np.clip(arr, 0.0, None))
    arr = np.clip(arr, 0.0, sqrt_clip)

    arr = (arr - BAND_MEAN[:, None, None]) / (BAND_STD[:, None, None] + 1e-6)

    return arr.astype(np.float32)


def gradient_magnitude(x: np.ndarray) -> np.ndarray:
    """
    Use image edges instead of raw spectral intensity.
    This often improves registration between different spectral bands.
    """
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag.astype(np.float32)


def standardize_single_image(x: np.ndarray) -> np.ndarray:
    """
    Zero-center and scale one 2D image.
    This is useful after gradient magnitude.
    """
    x = x.astype(np.float32, copy=False)
    std = float(x.std())

    if std < 1e-6:
        return np.zeros_like(x, dtype=np.float32)

    return ((x - float(x.mean())) / std).astype(np.float32)


def get_hanning_window(h: int, w: int) -> np.ndarray:
    """
    OpenCV phaseCorrelate can accept a window.
    Windowing reduces Fourier edge artifacts.
    """
    return cv2.createHanningWindow((w, h), cv2.CV_32F)


def preprocess_band(
    band: np.ndarray,
    use_gradient: bool,
) -> np.ndarray:
    x = band.astype(np.float32, copy=False)

    if use_gradient:
        x = gradient_magnitude(x)
        x = standardize_single_image(x)

    return x.astype(np.float32)


def estimate_shift_to_red_opencv(
    red: np.ndarray,
    moving: np.ndarray,
    window: Optional[np.ndarray],
) -> tuple[float, float, float, float, float]:
    """
    OpenCV convention:

        cv2.phaseCorrelate(reference, moving)

    returns approximately:

        dx_band_relative_to_red
        dy_band_relative_to_red

    In other words, it estimates the displacement of `moving`
    relative to `reference`.

    Therefore, the shift to apply to the moving band to align it
    with red is the negative:

        dx_apply = -dx_relative
        dy_apply = -dy_relative
    """
    red = red.astype(np.float32, copy=False)
    moving = moving.astype(np.float32, copy=False)

    if window is None:
        shift, response = cv2.phaseCorrelate(red, moving)
    else:
        shift, response = cv2.phaseCorrelate(red, moving, window)

    dx_relative = float(shift[0])
    dy_relative = float(shift[1])

    dx_apply = -dx_relative
    dy_apply = -dy_relative

    return dy_apply, dx_apply, dy_relative, dx_relative, float(response)


def process_file(
    npy_path: Path,
    red_band_idx: int,
    sqrt_clip: float,
    use_gradient: bool,
    use_window: bool,
) -> list[dict]:
    arr = np.load(npy_path)

    if arr.ndim != 3:
        raise ValueError(f"{npy_path} has shape {arr.shape}, expected (8, H, W)")

    if arr.shape[0] != 8:
        raise ValueError(f"{npy_path} has shape {arr.shape}, expected 8 bands")

    arr = normalize_phisat2(arr, sqrt_clip=sqrt_clip)

    n_bands, h, w = arr.shape

    window = get_hanning_window(h, w) if use_window else None

    red = preprocess_band(
        arr[red_band_idx],
        use_gradient=use_gradient,
    )

    rows = []

    for band_idx in range(n_bands):
        if band_idx == red_band_idx:
            dy_apply = 0.0
            dx_apply = 0.0
            dy_relative = 0.0
            dx_relative = 0.0
            response = 1.0
        else:
            moving = preprocess_band(
                arr[band_idx],
                use_gradient=use_gradient,
            )

            (
                dy_apply,
                dx_apply,
                dy_relative,
                dx_relative,
                response,
            ) = estimate_shift_to_red_opencv(
                red=red,
                moving=moving,
                window=window,
            )

        magnitude = float(np.sqrt(dy_relative**2 + dx_relative**2))

        rows.append(
            {
                "image": npy_path.name,
                "band": band_idx,
                "band_name": BAND_NAMES.get(band_idx, f"Band {band_idx}"),
                "reference_band": red_band_idx,
                "reference_band_name": BAND_NAMES.get(red_band_idx, "Red"),
                "dy_apply_to_band_px": dy_apply,
                "dx_apply_to_band_px": dx_apply,
                "dy_band_relative_to_red_px": dy_relative,
                "dx_band_relative_to_red_px": dx_relative,
                "misalignment_magnitude_px": magnitude,
                "phase_corr_response": response,
            }
        )

    return rows


def percentile(x, q):
    return float(np.percentile(np.asarray(x), q))


def summarize(df: pd.DataFrame, red_band_idx: int, include_red: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if include_red:
        d = df.copy()
        scope = "all_bands_including_red"
    else:
        d = df[df["band"] != red_band_idx].copy()
        scope = "all_non_red_bands"

    band_summary = (
        d.groupby(["band", "band_name"])
        .agg(
            n=("misalignment_magnitude_px", "count"),
            mean_dy_relative_px=("dy_band_relative_to_red_px", "mean"),
            mean_dx_relative_px=("dx_band_relative_to_red_px", "mean"),
            mean_abs_dy_px=("dy_band_relative_to_red_px", lambda x: np.mean(np.abs(x))),
            mean_abs_dx_px=("dx_band_relative_to_red_px", lambda x: np.mean(np.abs(x))),
            mean_magnitude_px=("misalignment_magnitude_px", "mean"),
            p90_magnitude_px=("misalignment_magnitude_px", lambda x: percentile(x, 90)),
            p95_magnitude_px=("misalignment_magnitude_px", lambda x: percentile(x, 95)),
            p99_magnitude_px=("misalignment_magnitude_px", lambda x: percentile(x, 99)),
            mean_phase_corr_response=("phase_corr_response", "mean"),
            p10_phase_corr_response=("phase_corr_response", lambda x: percentile(x, 10)),
        )
        .reset_index()
    )

    overall_summary = pd.DataFrame(
        [
            {
                "scope": scope,
                "n": len(d),
                "mean_magnitude_px": float(d["misalignment_magnitude_px"].mean()),
                "p90_magnitude_px": percentile(d["misalignment_magnitude_px"], 90),
                "p95_magnitude_px": percentile(d["misalignment_magnitude_px"], 95),
                "p99_magnitude_px": percentile(d["misalignment_magnitude_px"], 99),
                "mean_abs_dy_px": float(np.mean(np.abs(d["dy_band_relative_to_red_px"]))),
                "mean_abs_dx_px": float(np.mean(np.abs(d["dx_band_relative_to_red_px"]))),
                "mean_phase_corr_response": float(d["phase_corr_response"].mean()),
            }
        ]
    )

    return band_summary, overall_summary


def main():
    parser = argparse.ArgumentParser(
        description="Estimate PhiSat-2 band misalignment relative to red using OpenCV phase correlation."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PhiSat-2 .npy images of shape (8, H, W)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where CSV files will be saved",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="Glob pattern for input files",
    )
    parser.add_argument(
        "--red-band",
        type=int,
        default=3,
        help="Reference red band index. Default: 3",
    )
    parser.add_argument(
        "--sqrt-clip",
        type=float,
        default=SQRT_CLIP,
        help="SQRT_CLIP value used in PhiSat-2 preprocessing",
    )
    parser.add_argument(
        "--no-gradient",
        action="store_true",
        help="Disable gradient-magnitude preprocessing",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable Hanning window in OpenCV phaseCorrelate",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for testing",
    )

    args = parser.parse_args()

    files = sorted(args.input_dir.glob(args.pattern))

    if args.max_files is not None:
        files = files[: args.max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {args.input_dir} with pattern {args.pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for npy_path in tqdm(files, desc="Estimating shifts with OpenCV"):
        rows = process_file(
            npy_path=npy_path,
            red_band_idx=args.red_band,
            sqrt_clip=args.sqrt_clip,
            use_gradient=not args.no_gradient,
            use_window=not args.no_window,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    per_image_csv = args.output_dir / "misalignment_per_image_opencv.csv"
    df.to_csv(per_image_csv, index=False)

    band_summary_non_red, overall_non_red = summarize(
        df,
        red_band_idx=args.red_band,
        include_red=False,
    )

    band_summary_all, overall_all = summarize(
        df,
        red_band_idx=args.red_band,
        include_red=True,
    )

    band_summary_non_red_csv = args.output_dir / "misalignment_band_summary_non_red_opencv.csv"
    overall_non_red_csv = args.output_dir / "misalignment_overall_summary_non_red_opencv.csv"
    band_summary_all_csv = args.output_dir / "misalignment_band_summary_including_red_opencv.csv"
    overall_all_csv = args.output_dir / "misalignment_overall_summary_including_red_opencv.csv"

    band_summary_non_red.to_csv(band_summary_non_red_csv, index=False)
    overall_non_red.to_csv(overall_non_red_csv, index=False)
    band_summary_all.to_csv(band_summary_all_csv, index=False)
    overall_all.to_csv(overall_all_csv, index=False)

    print()
    print("Saved:")
    print(f"  {per_image_csv}")
    print(f"  {band_summary_non_red_csv}")
    print(f"  {overall_non_red_csv}")
    print(f"  {band_summary_all_csv}")
    print(f"  {overall_all_csv}")

    print()
    print("Per-band summary excluding red reference band:")
    print(
        band_summary_non_red[
            [
                "band",
                "band_name",
                "n",
                "mean_magnitude_px",
                "p90_magnitude_px",
                "p95_magnitude_px",
                "p99_magnitude_px",
                "mean_dy_relative_px",
                "mean_dx_relative_px",
                "mean_phase_corr_response",
            ]
        ].to_string(index=False)
    )

    print()
    print("Overall summary excluding red reference band:")
    print(overall_non_red.to_string(index=False))


if __name__ == "__main__":
    main()