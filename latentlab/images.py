"""Paired *pre-treatment* MAMA-MIA NIfTI phase + expert-mask measurements.

No resampling or registration is silently performed. All three arrays must have
aligned orientations, shape and affine. Output is imaging-only, not outcome data.
"""
import numpy as np
from scipy import ndimage as ndi

CHEAP_FEATURES = ["tumor_volume_ml", "tumor_bbox_fill", "tumor_bbox_z_mm",
                  "tumor_bbox_y_mm", "tumor_bbox_x_mm", "tumor_face_area_mm2",
                  "post_tumor_mean_z", "post_tumor_std_z",
                  "post_tumor_p10_z", "post_tumor_p90_z", "enhancement_median_z",
                  "enhancement_iqr_z"]
TARGET = "spatial_neighbor_difference_q90"


def _robust_pre_reference(pre, post):
    # Share a baseline phase reference; do NOT z-score phases independently,
    # otherwise physically interesting enhancement gets subtracted away.
    sample = pre[np.isfinite(pre)]
    if not sample.size:
        raise ValueError("No finite MRI voxels")
    low, med, high = np.percentile(sample, [1, 50, 99])
    scale = max(float((high - low) / 2.0), 1e-6)
    return (np.clip((pre.astype(np.float32) - med) / scale, -10, 10),
            np.clip((post.astype(np.float32) - med) / scale, -10, 10))


def extract_arrays(pre, post, mask, spacing=(1., 1., 1.)):
    pre, post, mask = np.asarray(pre), np.asarray(post), np.asarray(mask)
    spacing = np.asarray(spacing, dtype=float)
    if pre.ndim != 3 or pre.shape != post.shape or pre.shape != mask.shape:
        raise ValueError("Pre, post and expert mask must be aligned 3D arrays")
    if len(spacing) != 3 or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Spacing must have 3 finite positive values in mm")
    region = mask > 0
    if region.sum() < 20:
        raise ValueError("Tumor mask has fewer than 20 voxels")
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("Image contains invalid values")
    p0, p1 = _robust_pre_reference(pre, post)
    coords = np.argwhere(region)
    sizes = np.ptp(coords, axis=0) + 1
    vol_vox = int(region.sum())
    bbox_vox = int(np.prod(sizes))
    # Voxel-face area approximation, not a triangulated medical surface area.
    area = 0.0
    padded = np.pad(region.astype(np.int8), 1)
    for axis in range(3):
        nfaces = np.abs(np.diff(padded, axis=axis)).sum()
        area += float(nfaces * np.prod(spacing[np.arange(3) != axis]))
    vals = p1[region]
    delta = (p1 - p0)[region]
    feat = {
        "tumor_volume_ml": float(vol_vox * np.prod(spacing) / 1000.0),
        "tumor_bbox_fill": float(vol_vox / bbox_vox),
        "tumor_face_area_mm2": area,
        "post_tumor_mean_z": float(np.mean(vals)),
        "post_tumor_std_z": float(np.std(vals)),
        "post_tumor_p10_z": float(np.percentile(vals, 10)),
        "post_tumor_p90_z": float(np.percentile(vals, 90)),
        "enhancement_median_z": float(np.median(delta)),
        "enhancement_iqr_z": float(np.percentile(delta,75)-np.percentile(delta,25)),
    }
    for axis, nm in enumerate("zyx"):
        feat[f"tumor_bbox_{nm}_mm"] = float(sizes[axis] * spacing[axis])
    # Target: expensive relative to summary quantiles because spatial topology
    # and neighboring voxel intensities are needed. Never put into VAE input.
    diffs = []
    for axis in range(3):
        sl0 = [slice(None)]*3; sl1 = [slice(None)]*3
        sl0[axis] = slice(None, -1); sl1[axis] = slice(1, None)
        sl0, sl1 = tuple(sl0),tuple(sl1)
        valid = region[sl0] & region[sl1]
        if valid.any():
            diffs.append(np.abs(p1[sl0]-p1[sl1])[valid])
    if not diffs:
        raise ValueError("No spatially adjacent tumor voxels")
    feat[TARGET] = float(np.percentile(np.concatenate(diffs), 90))
    feat["mask_voxels"] = vol_vox
    return feat


def extract_nifti(pre_path, post_path, mask_path):
    import nibabel as nib
    files = [nib.load(str(p)) for p in (pre_path, post_path, mask_path)]
    for f in files[1:]:
        if f.shape != files[0].shape or not np.allclose(f.affine, files[0].affine, atol=1e-3):
            raise ValueError("NIfTI shape/affine mismatch; register externally and review alignment")
    arrays = [f.get_fdata(dtype=np.float32) for f in files]
    result = extract_arrays(*arrays, spacing=files[0].header.get_zooms()[:3])
    return result
