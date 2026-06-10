from pathlib import Path
import argparse
import random
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


TASK_CLASS_NAMES = {
    0: 'background',
    1: 'vegetation',
    2: 'built_up',
    3: 'water',
}

TASK_CLASS_COLORS = {
    0: '#808080',
    1: '#2ca02c',
    2: '#d62728',
    3: '#1f77b4',
}

TASK_CLASS_ORDER = [0, 1, 2, 3]
N_CLASSES = 4


def robust_stretch(x, p_low=2, p_high=98):
    x = x.astype('float32')
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return np.zeros_like(x, dtype='float32')
    lo, hi = np.percentile(x[finite], [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype='float32')
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def parse_band_list(s):
    return tuple(int(x) for x in str(s).split(','))


def ensure_chw(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f'Expected 2D or 3D image array, got shape {arr.shape}')

    if arr.shape[0] <= 20 and arr.shape[1] > 20 and arr.shape[2] > 20:
        return arr
    if arr.shape[2] <= 20 and arr.shape[0] > 20 and arr.shape[1] > 20:
        return np.moveaxis(arr, -1, 0)
    return arr


def mask_to_class_index(mask, n_classes=N_CLASSES):
    """Convert masks to 2D class-index format.

    Supports:
      - already indexed masks: (H, W)
      - one-hot/probability CHW: (C, H, W), where C == n_classes
      - one-hot/probability HWC: (H, W, C), where C == n_classes
      - singleton masks: (1, H, W) or (H, W, 1)
    """
    m = np.asarray(mask)

    if m.ndim == 2:
        return m.astype('uint8')

    if m.ndim != 3:
        raise ValueError(f'Expected 2D or 3D mask, got shape {m.shape}')

    if m.shape[0] == n_classes and m.shape[1] > 20 and m.shape[2] > 20:
        return np.argmax(m, axis=0).astype('uint8')

    if m.shape[-1] == n_classes and m.shape[0] > 20 and m.shape[1] > 20:
        return np.argmax(m, axis=-1).astype('uint8')

    if m.shape[0] == 1:
        return m[0].astype('uint8')
    if m.shape[-1] == 1:
        return m[..., 0].astype('uint8')

    if m.shape[0] <= 20 and m.shape[1] > 20 and m.shape[2] > 20:
        return np.argmax(m, axis=0).astype('uint8')
    if m.shape[-1] <= 20 and m.shape[0] > 20 and m.shape[1] > 20:
        return np.argmax(m, axis=-1).astype('uint8')

    raise ValueError(f'Cannot infer mask layout from shape {m.shape}')


def rgb_from_stack(stack, bands):
    stack = ensure_chw(stack)
    n_channels = stack.shape[0]
    rgb_bands = []
    for b in bands:
        if b < 0 or b >= n_channels:
            raise IndexError(f'Band index {b} out of range for stack with {n_channels} channels')
        rgb_bands.append(robust_stretch(stack[b]))
    return np.stack(rgb_bands, axis=-1)


def task_cmap():
    color_list = [TASK_CLASS_COLORS[k] for k in TASK_CLASS_ORDER]
    cmap = ListedColormap(color_list)
    boundaries = [k - 0.5 for k in TASK_CLASS_ORDER] + [TASK_CLASS_ORDER[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def mask_rgba(mask):
    mask = mask_to_class_index(mask)
    cmap, norm = task_cmap()
    return cmap(norm(mask))


def overlay_rgb(image_rgb, mask, alpha=0.45):
    mask = mask_to_class_index(mask)
    rgba = mask_rgba(mask)
    mask_rgb = rgba[..., :3]
    out = image_rgb.copy()
    valid = np.isfinite(mask)
    if valid.shape != out.shape[:2]:
        raise ValueError(f'Mask/image shape mismatch: mask={valid.shape}, image={out.shape[:2]}')
    out[valid] = (1 - alpha) * image_rgb[valid] + alpha * mask_rgb[valid]
    return np.clip(out, 0, 1)


def class_summary(mask):
    mask = mask_to_class_index(mask)
    n = mask.size
    vals, counts = np.unique(mask, return_counts=True)
    fracs = {int(v): float(c / n) for v, c in zip(vals, counts)}
    dominant = int(vals[np.argmax(counts)]) if len(vals) else -1
    return {
        'fractions': fracs,
        'dominant_class': dominant,
        'dominant_name': TASK_CLASS_NAMES.get(dominant, str(dominant)),
        'dominant_frac': float(fracs.get(dominant, 0.0)),
    }


def format_summary_lines(summary, prefix=''):
    parts = [f"{prefix}dominant: {summary['dominant_name']} ({100*summary['dominant_frac']:.1f}%)"]
    for cls in TASK_CLASS_ORDER:
        frac = summary['fractions'].get(cls, 0.0)
        parts.append(f"{prefix}{TASK_CLASS_NAMES[cls]}: {100*frac:.1f}%")
    return parts


def find_phi_pairs(phi_root):
    phi_root = Path(phi_root).expanduser()
    img_dir = phi_root / 'images_npy'
    mask_dir = phi_root / 'masks_npy'
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f'Expected {img_dir} and {mask_dir}')

    mapping = {}
    for img_path in sorted(img_dir.glob('image_*.npy')):
        patch_id = img_path.stem.replace('image_', '', 1)
        candidates = [
            mask_dir / f'mask_{patch_id}.npy',
            mask_dir / f'mask{patch_id}.npy',
            mask_dir / f'mask_patch{patch_id}.npy',
        ]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            hits = sorted(mask_dir.glob(f'*{patch_id}*.npy'))
            if hits:
                mask_path = hits[0]
        if mask_path is not None:
            mapping[int(patch_id)] = {'image_path': img_path, 'mask_path': mask_path}
    return mapping


def load_s2_manifest(s2_root):
    s2_root = Path(s2_root).expanduser()
    manifest_path = s2_root / 'manifest.csv'
    images_path = s2_root / 's2_images.npy'
    masks_path = s2_root / 's2_masks.npy'

    if not manifest_path.exists():
        raise FileNotFoundError(f'Missing manifest: {manifest_path}')
    if not images_path.exists() or not masks_path.exists():
        raise FileNotFoundError(f'Missing s2_images.npy or s2_masks.npy in {s2_root}')

    manifest = pd.read_csv(manifest_path)
    if 'patch_index' not in manifest.columns:
        raise ValueError('manifest.csv must contain patch_index')
    if 'status' in manifest.columns:
        manifest = manifest[manifest['status'] == 'ok'].copy()
    manifest = manifest.reset_index(drop=True)

    images = np.load(images_path, mmap_mode='r')
    masks = np.load(masks_path, mmap_mode='r')

    if len(manifest) != len(images) or len(manifest) != len(masks):
        raise ValueError(f'Length mismatch: manifest={len(manifest)}, images={len(images)}, masks={len(masks)}')

    mapping = {}
    for idx, row in manifest.iterrows():
        mapping[int(row['patch_index'])] = {'index': idx, 'row': row.to_dict()}
    return mapping, images, masks, manifest

def phisat_mask_to_task_index(mask):
    m = mask_to_class_index(mask)

    # ΦSat-2 raw one-hot order:
    # 0=background, 1=vegetation, 2=water, 3=built_up
    # desired:
    # 0=background, 1=vegetation, 2=built_up, 3=water
    lut = np.array([0, 1, 3, 2], dtype=np.uint8)
    return lut[m]

def make_figure(phi_img, phi_mask_raw, s2_img, s2_mask_raw, patch_id, out_path,
                phi_rgb_bands=(2, 1, 0), s2_rgb_bands=(2, 1, 0), title_extra=''):
    phi_img_chw = ensure_chw(phi_img)
    s2_img_chw = ensure_chw(s2_img)
    phi_mask = phisat_mask_to_task_index(phi_mask_raw)
    s2_mask = mask_to_class_index(s2_mask_raw)

    phi_rgb = rgb_from_stack(phi_img_chw, phi_rgb_bands)
    s2_rgb = rgb_from_stack(s2_img_chw, s2_rgb_bands)

    phi_overlay = overlay_rgb(phi_rgb, phi_mask)
    s2_overlay = overlay_rgb(s2_rgb, s2_mask)

    phi_summary = class_summary(phi_mask)
    s2_summary = class_summary(s2_mask)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

    axes[0, 0].imshow(phi_rgb)
    axes[0, 0].set_title(f'ΦSat-2 RGB (5 m)\nimage={tuple(phi_img_chw.shape)}')
    axes[0, 1].imshow(mask_rgba(phi_mask))
    axes[0, 1].set_title(f'ΦSat-2 mask\nraw={tuple(np.asarray(phi_mask_raw).shape)} → {tuple(phi_mask.shape)}')
    axes[0, 2].imshow(phi_overlay)
    axes[0, 2].set_title('ΦSat-2 overlay')
    axes[0, 3].axis('off')
    axes[0, 3].text(0.02, 0.98, '\n'.join(format_summary_lines(phi_summary, prefix='φ  ')),
                    va='top', ha='left', family='monospace', fontsize=10)

    axes[1, 0].imshow(s2_rgb)
    axes[1, 0].set_title(f'Sentinel-2 RGB (10 m)\nimage={tuple(s2_img_chw.shape)}')
    axes[1, 1].imshow(mask_rgba(s2_mask))
    axes[1, 1].set_title(f'Sentinel-2 mask\nraw={tuple(np.asarray(s2_mask_raw).shape)} → {tuple(s2_mask.shape)}')
    axes[1, 2].imshow(s2_overlay)
    axes[1, 2].set_title('Sentinel-2 overlay')
    axes[1, 3].axis('off')
    axes[1, 3].text(0.02, 0.98, '\n'.join(format_summary_lines(s2_summary, prefix='s2 ')),
                    va='top', ha='left', family='monospace', fontsize=10)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xticks([])
            ax.set_yticks([])

    legend_handles = [Patch(facecolor=TASK_CLASS_COLORS[k], label=f'{k}: {TASK_CLASS_NAMES[k]}') for k in TASK_CLASS_ORDER]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False)

    title = f'Patch comparison | patch_id={patch_id}'
    if title_extra:
        title += f' | {title_extra}'
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Create quicklooks comparing ΦSat-2 5 m patches and Sentinel-2 10 m patches.')
    parser.add_argument('--phi-root', required=True, help='Root dir with images_npy/ and masks_npy/ for 256x256 ΦSat-2 dataset.')
    parser.add_argument('--s2-root', required=True, help='Root dir with s2_images.npy, s2_masks.npy, manifest.csv from 128x128 Sentinel-2 dataset.')
    parser.add_argument('--out-dir', required=True, help='Directory where quicklooks and summary files will be written.')
    parser.add_argument('-N', '--n', type=int, default=50, help='Number of common patch ids to render. 0 means all common patch ids.')
    parser.add_argument('--sample', action='store_true', help='Randomly sample common patch ids instead of taking the lowest sorted patch ids.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patch-ids', default='', help='Comma-separated explicit patch ids to render. Overrides -N.')
    parser.add_argument('--phi-rgb-bands', default='2,1,0', help='0-based RGB channel indices for ΦSat-2 display.')
    parser.add_argument('--s2-rgb-bands', default='2,1,0', help='0-based RGB channel indices for Sentinel-2 display.')
    args = parser.parse_args()

    phi_rgb_bands = parse_band_list(args.phi_rgb_bands)
    s2_rgb_bands = parse_band_list(args.s2_rgb_bands)

    out_dir = Path(args.out_dir).expanduser()
    ql_dir = out_dir / 'quicklooks'
    out_dir.mkdir(parents=True, exist_ok=True)
    ql_dir.mkdir(parents=True, exist_ok=True)

    print('Scanning ΦSat-2 per-patch npy files...')
    phi_map = find_phi_pairs(args.phi_root)
    print(f'Found {len(phi_map)} ΦSat-2 image/mask pairs')

    print('Loading Sentinel-2 arrays + manifest...')
    s2_map, s2_images, s2_masks, manifest = load_s2_manifest(args.s2_root)
    print(f'Found {len(s2_map)} Sentinel-2 manifest rows with arrays')

    common_ids = sorted(set(phi_map.keys()) & set(s2_map.keys()))
    print(f'Common patch ids: {len(common_ids)}')

    if args.patch_ids.strip():
        requested = [int(x.strip()) for x in args.patch_ids.split(',') if x.strip()]
        common_set = set(common_ids)
        selected_ids = [pid for pid in requested if pid in common_set]
        missing = sorted(set(requested) - set(selected_ids))
        if missing:
            print(f'[warn] these requested patch ids were not found in both datasets: {missing}')
    else:
        selected_ids = common_ids.copy()
        if args.n > 0 and args.n < len(selected_ids):
            if args.sample:
                rng = random.Random(args.seed)
                selected_ids = sorted(rng.sample(selected_ids, args.n))
            else:
                selected_ids = selected_ids[:args.n]

    print(f'Selected {len(selected_ids)} patch ids for rendering')

    summary_rows = []
    failures = []

    for i, patch_id in enumerate(selected_ids, start=1):
        try:
            phi_img = np.load(phi_map[patch_id]['image_path'])
            phi_mask_raw = np.load(phi_map[patch_id]['mask_path'])
            s2_idx = s2_map[patch_id]['index']
            s2_img = np.asarray(s2_images[s2_idx])
            s2_mask_raw = np.asarray(s2_masks[s2_idx])

            ql_path = ql_dir / f'compare_patch{patch_id}.png'
            title_extra = f'φ_raw_mask={tuple(np.asarray(phi_mask_raw).shape)} | s2_raw_mask={tuple(np.asarray(s2_mask_raw).shape)}'
            make_figure(
                phi_img=phi_img,
                phi_mask_raw=phi_mask_raw,
                s2_img=s2_img,
                s2_mask_raw=s2_mask_raw,
                patch_id=patch_id,
                out_path=ql_path,
                phi_rgb_bands=phi_rgb_bands,
                s2_rgb_bands=s2_rgb_bands,
                title_extra=title_extra,
            )

            summary_rows.append({
                'patch_id': patch_id,
                'phi_image_path': str(phi_map[patch_id]['image_path']),
                'phi_mask_path': str(phi_map[patch_id]['mask_path']),
                's2_index': int(s2_idx),
                'quicklook_path': str(ql_path),
                'phi_image_shape': str(tuple(np.asarray(phi_img).shape)),
                'phi_mask_raw_shape': str(tuple(np.asarray(phi_mask_raw).shape)),
                'phi_mask_index_shape': str(tuple(mask_to_class_index(phi_mask_raw).shape)),
                's2_image_shape': str(tuple(np.asarray(s2_img).shape)),
                's2_mask_raw_shape': str(tuple(np.asarray(s2_mask_raw).shape)),
                's2_mask_index_shape': str(tuple(mask_to_class_index(s2_mask_raw).shape)),
                'status': 'ok',
                'error': '',
            })
        except Exception as e:
            failures.append((patch_id, f'{type(e).__name__}: {e}'))
            print(f'[failed] patch_id={patch_id}: {type(e).__name__}: {e}')
            summary_rows.append({
                'patch_id': patch_id,
                'phi_image_path': str(phi_map.get(patch_id, {}).get('image_path', '')),
                'phi_mask_path': str(phi_map.get(patch_id, {}).get('mask_path', '')),
                's2_index': int(s2_map.get(patch_id, {}).get('index', -1)) if patch_id in s2_map else -1,
                'quicklook_path': '',
                'status': 'failed',
                'error': f'{type(e).__name__}: {e}',
            })

        if i % 25 == 0 or i == len(selected_ids):
            print(f'[{i}/{len(selected_ids)}]')

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'comparison_manifest.csv', index=False)

    summary = {
        'phi_root': str(args.phi_root),
        's2_root': str(args.s2_root),
        'out_dir': str(out_dir),
        'n_phi_pairs': int(len(phi_map)),
        'n_s2_rows': int(len(s2_map)),
        'n_common_patch_ids': int(len(common_ids)),
        'n_selected': int(len(selected_ids)),
        'n_ok': int((summary_df['status'] == 'ok').sum()) if len(summary_df) else 0,
        'n_failed': int((summary_df['status'] == 'failed').sum()) if len(summary_df) else 0,
        'sample': bool(args.sample),
        'seed': int(args.seed),
        'phi_rgb_bands': list(phi_rgb_bands),
        's2_rgb_bands': list(s2_rgb_bands),
    }
    (out_dir / 'comparison_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('\nDone.')
    print(f"Saved quicklooks: {ql_dir}")
    print(f"Saved manifest: {out_dir / 'comparison_manifest.csv'}")
    print(f"Saved summary: {out_dir / 'comparison_summary.json'}")
    if failures:
        print('\nFailures:')
        for patch_id, err in failures[:20]:
            print(f'  patch_id={patch_id}: {err}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')


if __name__ == '__main__':
    main()
