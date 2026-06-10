#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

import time

BACKGROUND_CLASSES = [0, 60, 70]
VEGETATION_CLASSES = [10, 20, 30, 40, 90, 95, 100]
BUILT_UP_CLASSES = [50]
WATER_CLASSES = [80]

TASK_CLASSES = {
    "background": BACKGROUND_CLASSES,
    "vegetation": VEGETATION_CLASSES,
    "built_up": BUILT_UP_CLASSES,
    "water": WATER_CLASSES,
}

TASK_COLS = [f"task_frac_{k}" for k in TASK_CLASSES]


def frac_sum(df, classes):
    cols = [f"wc_frac_{c}" for c in classes]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[cols].sum(axis=1)


def score_candidate(current_sums, candidate_fracs, target_share):
    new_sums = current_sums + candidate_fracs
    new_share = new_sums / new_sums.sum()
    return np.abs(new_share - target_share).sum()


def balanced_sample(df, n, seed, candidate_pool_size=500):
    rng = np.random.default_rng(seed)

    fracs = df[TASK_COLS].to_numpy(dtype=float)
    target_share = np.ones(len(TASK_COLS)) / len(TASK_COLS)

    selected_indices = []
    remaining = np.arange(len(df))
    current_sums = np.zeros(len(TASK_COLS), dtype=float)

    start_time = time.time()
    last_print = start_time

    print(
        f"[INFO] Starting balanced sampling "
        f"(target={n}, pool={candidate_pool_size}, available={len(df)})"
    )

    while len(selected_indices) < n and len(remaining) > 0:
        current_iter = len(selected_indices)

        pool_size = min(candidate_pool_size, len(remaining))
        pool = rng.choice(remaining, size=pool_size, replace=False)

        scores = np.array([
            score_candidate(current_sums, fracs[i], target_share)
            for i in pool
        ])

        best_idx = pool[np.argmin(scores)]

        selected_indices.append(best_idx)
        current_sums += fracs[best_idx]

        remaining = remaining[remaining != best_idx]

        # Progress print every 100 iterations
        if (
            current_iter % 100 == 0
            or current_iter == n - 1
        ):
            now = time.time()

            elapsed = now - start_time
            step_time = now - last_print

            samples_done = current_iter + 1
            rate = samples_done / elapsed

            remaining_steps = n - samples_done
            eta_sec = remaining_steps / rate if rate > 0 else 0

            current_share = current_sums / current_sums.sum()

            print(
                f"[{samples_done:>6}/{n}] "
                f"{100*samples_done/n:6.2f}% | "
                f"elapsed={elapsed:8.1f}s | "
                f"rate={rate:7.2f} samples/s | "
                f"ETA={eta_sec:8.1f}s"
            )

            print(
                "           dataset_share="
                f"bg={current_share[0]:.3f}, "
                f"veg={current_share[1]:.3f}, "
                f"built={current_share[2]:.3f}, "
                f"water={current_share[3]:.3f}"
            )

            last_print = now

    print(
        f"[INFO] Finished balanced sampling "
        f"({len(selected_indices)} selected)"
    )

    return df.iloc[selected_indices].copy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-frac", type=float, default=0.80)
    parser.add_argument("--min-present-frac", type=float, default=0.02)

    parser.add_argument("--max-nodata", type=float, default=0.02)
    parser.add_argument("--max-cloud", type=float, default=None)
    parser.add_argument("--status", default="ok")

    parser.add_argument("--candidate-pool-size", type=int, default=5000)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "status" in df.columns:
        df = df[df["status"].astype(str) == args.status].copy()

    if "wc_nodata_frac" in df.columns:
        df = df[df["wc_nodata_frac"] <= args.max_nodata].copy()
    elif "wc_frac_0" in df.columns:
        df = df[df["wc_frac_0"] <= args.max_nodata].copy()

    if args.max_cloud is not None:
        for col in ["s2_cloud_frac", "real_nonclear_frac"]:
            if col in df.columns:
                df = df[df[col] <= args.max_cloud].copy()

    for name, classes in TASK_CLASSES.items():
        df[f"task_frac_{name}"] = frac_sum(df, classes)

    df["task_total_frac"] = df[TASK_COLS].sum(axis=1)

    df = df[df["task_total_frac"] > 0].copy()

    df[TASK_COLS] = df[TASK_COLS].div(df["task_total_frac"], axis=0)

    df["task_dominant_class"] = (
        df[TASK_COLS]
        .idxmax(axis=1)
        .str.replace("task_frac_", "", regex=False)
    )
    df["task_dominant_frac"] = df[TASK_COLS].max(axis=1)

    # Hard patch-level constraint:
    # no patch may be more than max-frac of a single task class.
    df = df[df["task_dominant_frac"] <= args.max_frac].copy()

    # Avoid totally uninformative patches where only one tiny mapped class exists.
    df = df[df[TASK_COLS].max(axis=1) >= args.min_present_frac].copy()

    if len(df) == 0:
        raise ValueError("No patches remain after filtering.")

    selected = balanced_sample(
        df=df,
        n=args.n,
        seed=args.seed,
        candidate_pool_size=args.candidate_pool_size,
    )

    selected = selected.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected.to_csv(out_path, index=False)

    patch_ids_path = out_path.with_name(out_path.stem + "_patch_ids.txt")
    patch_ids_path.write_text(
        "\n".join(selected["patch_index"].astype(int).astype(str)) + "\n",
        encoding="utf-8",
    )

    mean_fracs = {
        c.replace("task_frac_", ""): float(selected[c].mean())
        for c in TASK_COLS
    }

    sum_fracs = {
        c.replace("task_frac_", ""): float(selected[c].sum())
        for c in TASK_COLS
    }

    total = sum(sum_fracs.values())
    dataset_share = {
        k: float(v / total)
        for k, v in sum_fracs.items()
    }

    summary = {
        "task": "landcover_4class",
        "classes": {
            "0_background": BACKGROUND_CLASSES,
            "1_vegetation": VEGETATION_CLASSES,
            "2_built_up": BUILT_UP_CLASSES,
            "3_water": WATER_CLASSES,
        },
        "requested_n": args.n,
        "selected_n": int(len(selected)),
        "available_after_filtering": int(len(df)),
        "constraints": {
            "max_single_class_per_patch": args.max_frac,
            "target_dataset_share_per_class": 0.25,
            "max_nodata": args.max_nodata,
            "max_cloud": args.max_cloud,
        },
        "mean_task_fractions_per_patch": mean_fracs,
        "sum_task_fractions": sum_fracs,
        "dataset_pixel_share": dataset_share,
        "dominant_class_counts": selected["task_dominant_class"].value_counts().to_dict(),
        "max_observed_dominant_frac": float(selected["task_dominant_frac"].max()),
    }

    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()