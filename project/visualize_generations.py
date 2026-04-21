#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from matplotlib.tri import Triangulation


def load_jsonl_files(input_path: Path) -> List[Dict[str, Any]]:
    rows = []

    files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {input_path}")

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                r = obj["results"]

                rows.append(
                    {
                        "file": file.name,
                        "generation": int(obj["generation"]),
                        "miou_clean": float(r["miou_clean"]),
                        "prediction_consistency": float(r["prediction_consistency"]),
                        "std_dev": float(r["std_dev"]),
                    }
                )

    return rows



def make_generation_colors(generations):
    unique = sorted(set(generations))
    n = len(unique)

    cmap = plt.cm.plasma  # yellow → purple
    values = np.linspace(0.95, 0.1, n)

    return {g: cmap(v) for g, v in zip(unique, values)}



def compute_pareto_2d(rows):
    """
    Only uses:
      - miou_clean (maximize)
      - prediction_consistency (maximize)
    """
    pts = sorted(rows, key=lambda r: (-r["miou_clean"], -r["prediction_consistency"]))

    pareto = []
    best_pc = -np.inf

    for p in pts:
        if p["prediction_consistency"] > best_pc:
            pareto.append(p)
            best_pc = p["prediction_consistency"]

    return sorted(pareto, key=lambda r: r["miou_clean"])

def compute_pareto_miou_std(rows):
    """
    Pareto front for:
      - maximize miou_clean
      - minimize std_dev
    """
    # Sort by miou descending, std ascending
    pts = sorted(rows, key=lambda r: (-r["miou_clean"], r["std_dev"]))

    pareto = []
    best_std = float("inf")

    for p in pts:
        if p["std_dev"] < best_std:
            pareto.append(p)
            best_std = p["std_dev"]

    # sort for plotting (left → right)
    return sorted(pareto, key=lambda r: r["miou_clean"])

def plot_2d_miou_std(rows, outpath, title="", show_pareto=False):
    gens = [r["generation"] for r in rows]
    color_map = make_generation_colors(gens)
    unique_gens = sorted(set(gens))

    fig, ax = plt.subplots(figsize=(10, 8))

    for g in unique_gens:
        pts = [r for r in rows if r["generation"] == g]

        ax.scatter(
            [p["miou_clean"] for p in pts],
            [p["std_dev"] for p in pts],
            color=color_map[g],
            s=60,
            alpha=0.9,
            label=f"Gen {g}",
        )

    if show_pareto:
        pareto = compute_pareto_miou_std(rows)

        ax.plot(
            [p["miou_clean"] for p in pareto],
            [p["std_dev"] for p in pareto],
            linestyle="--",
            linewidth=2.5,
            color="black",
            label="Pareto",
        )

        ax.scatter(
            [p["miou_clean"] for p in pareto],
            [p["std_dev"] for p in pareto],
            color="black",
            s=40,
        )

    ax.set_xlabel("miou_clean (higher is better)")
    ax.set_ylabel("std_dev (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def dominates_3d(a, b):
    return (
        (a["miou_clean"] >= b["miou_clean"])
        and (a["prediction_consistency"] >= b["prediction_consistency"])
        and (a["std_dev"] <= b["std_dev"])
        and (
            (a["miou_clean"] > b["miou_clean"])
            or (a["prediction_consistency"] > b["prediction_consistency"])
            or (a["std_dev"] < b["std_dev"])
        )
    )


def compute_pareto_3d(rows):
    pareto = []
    for i, a in enumerate(rows):
        if not any(dominates_3d(b, a) for j, b in enumerate(rows) if i != j):
            pareto.append(a)
    return pareto

def plot_2d(rows, outpath, title="", show_pareto=False):
    gens = [r["generation"] for r in rows]
    color_map = make_generation_colors(gens)
    unique_gens = sorted(set(gens))

    fig, ax = plt.subplots(figsize=(10, 8))

    for g in unique_gens:
        pts = [r for r in rows if r["generation"] == g]

        ax.scatter(
            [p["miou_clean"] for p in pts],
            [p["prediction_consistency"] for p in pts],
            color=color_map[g],
            s=60,
            alpha=0.9,
            label=f"Gen {g}",
        )

    if show_pareto:
        pareto = compute_pareto_2d(rows)

        ax.plot(
            [p["miou_clean"] for p in pareto],
            [p["prediction_consistency"] for p in pareto],
            linestyle="--",
            linewidth=2.5,
            color="black",
            label="Pareto",
        )

        ax.scatter(
            [p["miou_clean"] for p in pareto],
            [p["prediction_consistency"] for p in pareto],
            color="black",
            s=40,
        )

    ax.set_xlabel("miou_clean")
    ax.set_ylabel("prediction_consistency")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
def plot_3d(rows, outpath, title="", show_pareto=False):
    # ---------------------------
    # FILTER DATA
    # ---------------------------
    rows = [
        r for r in rows
        if r["miou_clean"] > 0.4 and r["prediction_consistency"] > 0.4
    ]

    if not rows:
        print("No points passed filtering (miou_clean > 0.4 and prediction_consistency > 0.4)")
        return

    gens = [r["generation"] for r in rows]
    color_map = make_generation_colors(gens)
    unique_gens = sorted(set(gens))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------
    # FIX AXIS LIMITS / SHAPE
    # ---------------------------
    xs = [r["miou_clean"] for r in rows]
    ys = [r["prediction_consistency"] for r in rows]
    zs = [r["std_dev"] for r in rows]

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(min(zs), max(zs))

    # Keep the 3D box itself stable
    ax.set_box_aspect((
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
    ))

    # optional: reduce the "moving panes" feeling
    ax.set_proj_type("persp")
    # or try: ax.set_proj_type("ortho")

    # Keep layout fixed instead of re-tightening every frame
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    # ---------------------------
    # PLOT POINTS
    # ---------------------------
    for g in unique_gens:
        pts = [r for r in rows if r["generation"] == g]

        ax.scatter(
            [p["miou_clean"] for p in pts],
            [p["prediction_consistency"] for p in pts],
            [p["std_dev"] for p in pts],
            color=color_map[g],
            s=55,
            alpha=0.9,
        )

    # ---------------------------
    # PARETO POINTS + SURFACE
    # ---------------------------
    if show_pareto:
        pareto = compute_pareto_3d(rows)

        px = np.array([p["miou_clean"] for p in pareto])
        py = np.array([p["prediction_consistency"] for p in pareto])
        pz = np.array([p["std_dev"] for p in pareto])

        ax.scatter(
            px, py, pz,
            color="black",
            s=90,
            marker="x",
            linewidths=2,
        )

        # Draw a triangulated surface through Pareto points
        # Needs at least 3 points
        if len(pareto) >= 3:
            try:
                tri = Triangulation(px, py)
                ax.plot_trisurf(
                    tri,
                    pz,
                    color="lightgray",
                    alpha=0.35,
                    edgecolor="black",
                    linewidth=0.4,
                )
            except Exception as e:
                print(f"Could not draw Pareto surface: {e}")

    # ---------------------------
    # AXES / TITLE
    # ---------------------------
    ax.set_xlabel("miou_clean")
    ax.set_ylabel("prediction_consistency")
    ax.set_zlabel("std_dev")
    ax.set_title(title)

    # ---------------------------
    # ROTATION ANIMATION
    # ---------------------------
    def update(angle):
        ax.view_init(elev=25, azim=angle)
        return (fig,)

    angles = np.linspace(0, 360, 120)
    anim = FuncAnimation(fig, update, frames=angles, interval=50)

    if not str(outpath).endswith(".gif"):
        outpath = str(outpath).replace(".png", ".gif")

    anim.save(outpath, writer=PillowWriter(fps=20))
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--pareto", action="store_true")

    args = parser.parse_args()

    rows = load_jsonl_files(args.input)
    args.outdir.mkdir(exist_ok=True)

    # group by file
    files = sorted(set(r["file"] for r in rows))

    for file in files:
        file_rows = [r for r in rows if r["file"] == file]
        base = Path(file).stem

        plot_2d(
            file_rows,
            args.outdir / f"{base}_2d.png",
            title=f"{file} — 2D Scatter",
            show_pareto=args.pareto,
        )

        plot_3d(
            file_rows,
            args.outdir / f"{base}_3d.png",
            title=f"{file} — 3D Scatter",
            show_pareto=args.pareto,
        )
        plot_2d_miou_std(
            file_rows,
            args.outdir / f"{base}_miou_std.png",
            title=f"{file} — miou vs std_dev",
            show_pareto=args.pareto,
        )

    print("Done.")

main()