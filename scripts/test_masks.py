# check_s2_dataset.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


image_paths = Path("/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/images_s2_npy")
mask_paths = Path("/shared/home/ivanderspoel/scratch/segmentation_dataset_v1/masks_s2_npy")

NUM_CLASSES = 4
EXPECTED_IMAGE_SHAPE = (7, 128, 128)
EXPECTED_MASK_SHAPE = (4, 128, 128)

TASK_CLASS_NAMES = {
    0: "background",
    1: "vegetation",
    2: "built_up",
    3: "water",
}

# RGB colors in [0, 1]
# 0 background -> black
# 1 vegetation -> green
# 2 built_up -> red
# 3 water -> blue
CLASS_COLORS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.8, 0.0],
        [0.9, 0.1, 0.1],
        [0.1, 0.3, 1.0],
    ],
    dtype=np.float32,
)


def main(max_check=None, save_examples=12):
    images = sorted(image_paths.glob("image_*.npy"))

    print(f"Found {len(images)} images")

    if len(images) == 0:
        raise RuntimeError(f"No images found in {image_paths}")

    total_pixels = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    bad_shapes = []
    missing_masks = []
    non_onehot = []
    multi_hot = []
    zero_hot = []
    weird_values = []

    band_min = np.full(7, np.inf, dtype=np.float64)
    band_max = np.full(7, -np.inf, dtype=np.float64)
    band_mean_sum = np.zeros(7, dtype=np.float64)

    n_checked = 0

    for i, img_path in enumerate(images):
        if max_check is not None and i >= max_check:
            break

        sample_id = img_path.stem.replace("image_", "", 1)
        mask_path = mask_paths / f"mask_{sample_id}.npy"

        if not mask_path.exists():
            missing_masks.append(str(mask_path))
            continue

        x = np.load(img_path)
        y = np.load(mask_path)

        if x.shape != EXPECTED_IMAGE_SHAPE:
            bad_shapes.append((str(img_path), x.shape, "image"))
            continue

        if y.shape != EXPECTED_MASK_SHAPE:
            bad_shapes.append((str(mask_path), y.shape, "mask"))
            continue

        unique_vals = np.unique(y)
        if not np.all(np.isin(unique_vals, [0, 1, 0.0, 1.0])):
            weird_values.append((str(mask_path), unique_vals[:20]))

        # For one-hot masks, sum over class axis should be exactly 1 per pixel
        pixel_class_sum = y.sum(axis=0)

        if np.any(pixel_class_sum > 1):
            multi_hot.append(str(mask_path))

        if np.any(pixel_class_sum == 0):
            zero_hot.append(str(mask_path))

        if not np.all(pixel_class_sum == 1):
            non_onehot.append(str(mask_path))

        class_map = np.argmax(y, axis=0)
        counts = np.bincount(class_map.reshape(-1), minlength=NUM_CLASSES)
        class_counts += counts
        total_pixels += class_map.size

        # Image stats
        x_float = x.astype(np.float32)
        band_min = np.minimum(band_min, x_float.min(axis=(1, 2)))
        band_max = np.maximum(band_max, x_float.max(axis=(1, 2)))
        band_mean_sum += x_float.mean(axis=(1, 2))

        n_checked += 1

        if i < save_examples:
            save_visual_example(x, y, sample_id)

    print("\n=== Basic checks ===")
    print(f"Checked samples: {n_checked}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Bad shapes: {len(bad_shapes)}")
    print(f"Non-one-hot masks: {len(non_onehot)}")
    print(f"Multi-hot masks: {len(multi_hot)}")
    print(f"Zero-hot pixels in masks: {len(zero_hot)}")
    print(f"Masks with values other than 0/1: {len(weird_values)}")

    if missing_masks[:5]:
        print("\nExample missing masks:")
        for item in missing_masks[:5]:
            print(item)

    if bad_shapes[:5]:
        print("\nExample bad shapes:")
        for item in bad_shapes[:5]:
            print(item)

    if non_onehot[:5]:
        print("\nExample non-one-hot masks:")
        for item in non_onehot[:5]:
            print(item)

    if weird_values[:5]:
        print("\nExample weird mask values:")
        for path, vals in weird_values[:5]:
            print(path, vals)

    print("\n=== Class balance ===")
    print("Class counts:", class_counts)

    if total_pixels > 0:
        class_freq = class_counts / total_pixels
        for c, f in enumerate(class_freq):
            print(f"Class {c} ({TASK_CLASS_NAMES[c]}): {f:.6f}")

    print("\n=== Raw image stats ===")
    print("Band min:", band_min)
    print("Band max:", band_max)

    if n_checked > 0:
        print("Band mean:", band_mean_sum / n_checked)

    print("\nSaved visual examples as debug_sample_*_overlay.png")


def make_rgb_s2(x: np.ndarray) -> np.ndarray:
    """
    Convert Sentinel-2B 7-channel image to normalized RGB.

    Band mapping:
    - Band 0: B02 Blue
    - Band 1: B03 Green
    - Band 2: B04 Red

    Therefore:
        RGB = [2, 1, 0]
    """
    rgb = x[[2, 1, 0]].astype(np.float32)  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))     # (H, W, 3)

    # Robust per-channel normalization for visualization
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)

    rgb = (rgb - lo) / (hi - lo + 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def colorize_mask(class_map: np.ndarray) -> np.ndarray:
    """
    Convert class index mask (H, W) to RGB color mask (H, W, 3).
    """
    return CLASS_COLORS[class_map]


def make_overlay(rgb: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    """
    Create RGB + mask overlay with class-specific alpha.

    Background uses lower alpha so the image remains visible.
    """
    color_mask = colorize_mask(class_map)

    alpha_map = np.zeros(class_map.shape, dtype=np.float32)
    alpha_map[class_map == 0] = 0.12  # background
    alpha_map[class_map == 1] = 0.45  # vegetation
    alpha_map[class_map == 2] = 0.45  # built_up
    alpha_map[class_map == 3] = 0.45  # water

    alpha_map = alpha_map[..., None]

    overlay = (1.0 - alpha_map) * rgb + alpha_map * color_mask
    overlay = np.clip(overlay, 0.0, 1.0)

    return overlay


def save_visual_example(x, y, sample_id):
    """
    Save a visual debug plot with:
    - normalized RGB
    - colored class mask
    - RGB + mask overlay

    x shape: (7, 128, 128)
    y shape: (4, 128, 128), one-hot
    """
    class_map = np.argmax(y, axis=0)

    rgb = make_rgb_s2(x)
    overlay = make_overlay(rgb, class_map)

    cmap = ListedColormap(CLASS_COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"RGB - {sample_id}")
    axes[0].axis("off")

    axes[1].imshow(class_map, vmin=0, vmax=NUM_CLASSES - 1, cmap=cmap)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("RGB + Mask Overlay")
    axes[2].axis("off")

    legend_handles = [
        Patch(
            facecolor=CLASS_COLORS[i],
            edgecolor="white",
            label=f"{i}: {TASK_CLASS_NAMES[i]}",
        )
        for i in range(NUM_CLASSES)
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        fontsize=9,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(f"debug_sample_{sample_id}_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(max_check=None, save_examples=12)