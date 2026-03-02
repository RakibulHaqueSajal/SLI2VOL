# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.measure import label
from skimage.morphology import convex_hull_image

from model import Correspondence_Flow_Net
from model import *  # weight_init if needed
from dataset import edge_profile  # your existing edge_profile
from data_creator.pancreas_dataset import (
    build_multi_source_loaders,
    collect_source_items_mixed,  # used to create a labeled-only eval set from train sources
    base_img_lab_tf,
    my_collate_drop_none,
)
from monai.data import CacheDataset
from torch.utils.data import DataLoader


# ============================================================
# 1) Verification module (ported)
# ============================================================
def verification_module(mask_img, mask_collection, img, img_collection, template):
    def getLargestCC(segmentation):
        labels_cc = label(segmentation)
        if labels_cc.max() == 0:
            return segmentation
        largestCC = labels_cc == (np.argmax(np.bincount(labels_cc.flat)[1:]) + 1)
        return (largestCC > 0).astype(int)

    def refine(segmentation, original):
        _ = convex_hull_image(segmentation)
        return segmentation

    num_class = int(np.max(mask_collection[-1]))
    mask_refine = np.zeros(mask_img.shape, dtype=np.int32)

    for nc in range(1, num_class + 1):
        mask_history = (mask_collection[-1] == nc).astype(np.float32)
        mask_img_temp = (mask_img == nc).astype(np.float32)

        mask_neg = (binary_erosion(mask_history, iterations=0) * 0).astype(np.float32)
        # Use the exact ring logic you had:
        from scipy.ndimage import binary_dilation, binary_fill_holes
        mask_neg = binary_dilation(mask_history, iterations=5).astype(np.float32) - mask_history
        mask_neg = (mask_neg > 0).astype(np.float32)

        mask_neg_c = np.tile(mask_neg[None, ...], (img.shape[0], 1, 1))
        mask_pos_c = np.tile(mask_history[None, ...], (img.shape[0], 1, 1))

        pos_means, neg_means = [], []
        for ci in range(mask_pos_c.shape[0]):
            pos_vals = img_collection[-1, ci][mask_pos_c[ci] == 1]
            neg_vals = img_collection[-1, ci][mask_neg_c[ci] == 1]
            pos_means.append(np.mean(pos_vals) if pos_vals.size > 0 else np.nan)
            neg_means.append(np.mean(neg_vals) if neg_vals.size > 0 else np.nan)

        pos_means = np.array(pos_means, dtype=np.float32)
        neg_means = np.array(neg_means, dtype=np.float32)

        feature_positive = np.ones_like(img, dtype=np.float32) * pos_means[:, None, None]
        feature_negative = np.ones_like(img, dtype=np.float32) * neg_means[:, None, None]

        feature_positive = np.sum((img - feature_positive) ** 2, axis=0)
        feature_negative = np.sum((img - feature_negative) ** 2, axis=0)

        positive_likelihood = np.zeros_like(feature_positive, dtype=np.float32)
        positive_likelihood[feature_positive < feature_negative] = 1.0

        positive_likelihood = positive_likelihood * mask_img_temp
        positive_likelihood[positive_likelihood > 0] = 1.0
        positive_likelihood[~np.isfinite(positive_likelihood)] = 0.0

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)
        positive_likelihood = binary_dilation(positive_likelihood, iterations=2)
        positive_likelihood = binary_erosion(positive_likelihood, iterations=2)
        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)

        mask_refine[positive_likelihood > 0] = nc

    return mask_refine, template


# ============================================================
# 2) Dice + Surface distance helpers
# ============================================================

def symmetric_hausdorff_distance(pred_hw: np.ndarray, gt_hw: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance in PIXELS between binary surfaces.
    Uses distance transforms of the *surface* sets.

    Edge cases:
      - both empty: 0
      - one empty: NaN (so it won't poison averages/boxplots)
    """
    pred_fg = (pred_hw > 0)
    gt_fg   = (gt_hw > 0)

    if (not pred_fg.any()) and (not gt_fg.any()):
        return 0.0
    if (pred_fg.any() and (not gt_fg.any())) or ((not pred_fg.any()) and gt_fg.any()):
        return np.nan

    pred_s = surface_mask(pred_fg)
    gt_s   = surface_mask(gt_fg)

    # if surface extraction collapsed (tiny objects), fall back to fg as surface
    if not pred_s.any():
        pred_s = pred_fg
    if not gt_s.any():
        gt_s = gt_fg

    dt_pred = distance_transform_edt(~pred_s)  # dist to nearest pred surface
    dt_gt   = distance_transform_edt(~gt_s)    # dist to nearest gt surface

    # directed Hausdorff: max distance from points of A to nearest point in B
    d_gt_to_pred = dt_pred[gt_s].max() if gt_s.any() else np.nan
    d_pred_to_gt = dt_gt[pred_s].max() if pred_s.any() else np.nan

    return float(max(d_gt_to_pred, d_pred_to_gt))


def compute_slicewise_hausdorff_in_range(pred_zhw, gt_zhw, z_start, z_end):
    z_start = max(0, int(z_start))
    z_end   = min(gt_zhw.shape[0] - 1, int(z_end))
    zs = np.arange(z_start, z_end + 1, dtype=np.int32)
    vals = np.zeros((len(zs),), dtype=np.float32)

    for j, z in enumerate(zs):
        vals[j] = symmetric_hausdorff_distance(pred_zhw[z], gt_zhw[z])

    return zs, vals


def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (den + eps))

def dice_multiclass_slice(pred_hw: np.ndarray, gt_hw: np.ndarray, classes=None) -> float:
    if classes is None:
        cls = np.unique(gt_hw)
        cls = cls[cls != 0]
    else:
        cls = np.array(list(classes), dtype=np.int32)

    if cls.size == 0:
        return 1.0 if np.max(pred_hw) == 0 else 0.0

    ds = []
    for c in cls:
        ds.append(dice_binary(pred_hw == c, gt_hw == c))
    return float(np.mean(ds))

def compute_slicewise_dice_in_range(pred_zhw, gt_zhw, z_start, z_end, mode="multiclass_mean"):
    z_start = max(0, int(z_start))
    z_end = min(gt_zhw.shape[0] - 1, int(z_end))
    zs = np.arange(z_start, z_end + 1, dtype=np.int32)
    vals = np.zeros((len(zs),), dtype=np.float32)

    if mode == "binary_union":
        for j, z in enumerate(zs):
            vals[j] = dice_binary(pred_zhw[z] > 0, gt_zhw[z] > 0)
    elif mode == "multiclass_mean":
        for j, z in enumerate(zs):
            vals[j] = dice_multiclass_slice(pred_zhw[z], gt_zhw[z], classes=None)
    else:
        raise ValueError(f"Unknown dice mode: {mode}")

    return zs, vals

def surface_mask(binary_hw: np.ndarray) -> np.ndarray:
    """
    Surface pixels = foreground - eroded(foreground).
    """
    fg = binary_hw.astype(bool)
    if not fg.any():
        return np.zeros_like(fg, dtype=bool)
    er = binary_erosion(fg, iterations=1, border_value=0)
    surf = np.logical_and(fg, np.logical_not(er))
    return surf

def symmetric_mean_surface_distance(pred_hw: np.ndarray, gt_hw: np.ndarray) -> float:
    """
    ASSD-like slice metric in PIXELS:
      mean( dist(GT_surface -> pred) ) and mean( dist(Pred_surface -> gt) ), averaged.
    Edge cases:
      - both empty: 0
      - one empty: NaN (so you can ignore in averages/boxplots if you want)
    """
    pred_fg = (pred_hw > 0)
    gt_fg = (gt_hw > 0)

    if (not pred_fg.any()) and (not gt_fg.any()):
        return 0.0
    if (pred_fg.any() and (not gt_fg.any())) or ((not pred_fg.any()) and gt_fg.any()):
        return np.nan

    pred_s = surface_mask(pred_fg)
    gt_s = surface_mask(gt_fg)

    if not pred_s.any() or not gt_s.any():
        # fallback: if erosion ate everything (tiny objects), treat fg as surface
        pred_s = pred_fg
        gt_s = gt_fg

    # distance to nearest surface
    dt_pred = distance_transform_edt(~pred_s)  # distance to pred surface
    dt_gt = distance_transform_edt(~gt_s)      # distance to gt surface

    d_gt_to_pred = dt_pred[gt_s].mean() if gt_s.any() else np.nan
    d_pred_to_gt = dt_gt[pred_s].mean() if pred_s.any() else np.nan

    return float(0.5 * (d_gt_to_pred + d_pred_to_gt))

def compute_slicewise_surface_distance_in_range(pred_zhw, gt_zhw, z_start, z_end):
    z_start = max(0, int(z_start))
    z_end = min(gt_zhw.shape[0] - 1, int(z_end))
    zs = np.arange(z_start, z_end + 1, dtype=np.int32)
    vals = np.zeros((len(zs),), dtype=np.float32)

    for j, z in enumerate(zs):
        vals[j] = symmetric_mean_surface_distance(pred_zhw[z], gt_zhw[z])

    return zs, vals


# ============================================================
# 3) NIfTI savers
# ============================================================
def save_nifti(vol_zhw: np.ndarray, out_path: str, dtype=np.int16):
    nii = nib.Nifti1Image(vol_zhw.astype(dtype), np.eye(4))
    nib.save(nii, out_path)

def save_overlap_3d_nifti(gt_zhw, pred_zhw, out_path, organ_only=False, z_start=None, z_end=None):
    gt_fg = (gt_zhw > 0).astype(np.uint8)
    pr_fg = (pred_zhw > 0).astype(np.uint8)
    overlay = (gt_fg * 1) + (pr_fg * 2)  # 0,1,2,3

    if organ_only and (z_start is not None) and (z_end is not None):
        z_start = max(0, int(z_start))
        z_end = min(overlay.shape[0] - 1, int(z_end))
        tmp = np.zeros_like(overlay, dtype=np.uint8)
        tmp[z_start:z_end + 1] = overlay[z_start:z_end + 1]
        overlay = tmp

    nii = nib.Nifti1Image(overlay.astype(np.uint8), np.eye(4))
    nib.save(nii, out_path)


# ============================================================
# 4) Plotting
# ============================================================
def plot_curve(case_tag: str, zs: np.ndarray, ys: np.ndarray, seed_idx: int, ylabel: str, title: str, out_path: str):
    plt.figure()
    plt.plot(zs, ys, marker="o", markersize=3)
    plt.axvline(seed_idx, linestyle="--", label="Seed slice")
    plt.xlabel("Slice index (z)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def boxplot_metrics(dice_list, sd_list, hd_list,out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    

    dice_arr = np.asarray(dice_list, dtype=np.float32)

    sd_arr = np.asarray(sd_list, dtype=np.float32)
    sd_clean = sd_arr[np.isfinite(sd_arr)]

    hd_arr = np.asarray(hd_list, dtype=np.float32)
    hd_clean = hd_arr[np.isfinite(hd_arr)]

    # Save overall mean metrics arrays
    np.save(os.path.join(out_dir, f"mean_dice_{tag}.npy"), dice_arr)
    np.save(os.path.join(out_dir, f"mean_surface_distance_{tag}.npy"), sd_arr)
    np.save(os.path.join(out_dir, f"mean_hausdorff_{tag}.npy"), hd_arr)

    # Dice boxplot
    plt.figure()
    plt.boxplot(dice_list, vert=True, showfliers=False)
    plt.ylabel("Mean Dice (organ z-range)")
    plt.title(f"Evaluation Set Boxplot: Dice ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"boxplot_dice_{tag}.png"), dpi=150)
    plt.close()

    # Surface distance boxplot (ignore NaNs)
    sd_clean = [x for x in sd_list if np.isfinite(x)]
    plt.figure()
    plt.boxplot(sd_clean, vert=True, showfliers=False)
    plt.ylabel("Mean Surface Distance (pixels, organ z-range)")
    plt.title(f"Evaluation Set Boxplot: Surface Distance ({tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"boxplot_surface_distance_{tag}.png"), dpi=150)
    plt.close()


# ============================================================
# 5) Build a labeled-only eval loader from specific sources
#    (so you can make boxplots for “train sources evaluation set”)
# ============================================================
class EvalSingleSliceFromCache(torch.utils.data.Dataset):
    """
    Cached ds yields dict with:
      image [1,H,W,Z], label_full [1,H,W,Z], source, image_meta_dict
    Produces:
      img_vol [Z,H,W], lab_full [Z,H,W], lab_seed [Z,H,W], seed_idx, z_start, z_end, case_id, source
    """
    def __init__(self, cached_ds: CacheDataset):
        self.cached_ds = cached_ds

    def __len__(self):
        return len(self.cached_ds)

    def __getitem__(self, idx):
        d = self.cached_ds[idx]
        img = d["image"].squeeze(0)         # [H,W,Z]
        lab = d["label_full"].squeeze(0)    # [H,W,Z]

        img_np = img.cpu().numpy().astype(np.float32)
        lab_np = (lab.cpu().numpy() > 0).astype(np.uint8)

        areas = lab_np.sum(axis=(0, 1))
        if areas.max() == 0:
            return None

        seed_idx = int(np.argmax(areas))
        pos = np.nonzero(areas > 0)[0]
        z_start, z_end = int(pos[0]), int(pos[-1])

        lab_seed = np.zeros_like(lab_np, dtype=np.uint8)
        lab_seed[:, :, seed_idx] = lab_np[:, :, seed_idx]

        # [Z,H,W]
        img_z = np.moveaxis(img_np, -1, 0)
        lab_full_z = np.moveaxis(lab_np, -1, 0)
        lab_seed_z = np.moveaxis(lab_seed, -1, 0)

        # case_id from meta dict
        case_id = "unknown_case"
        meta = d.get("image_meta_dict", None)
        if isinstance(meta, dict):
            f = meta.get("filename_or_obj", None)
            if f is not None:
                fname = os.path.basename(str(f))
                # strip .nii.gz
                case_id = os.path.splitext(os.path.splitext(fname)[0])[0]

        source = d.get("source", "unknown_source")

        return (
            torch.from_numpy(img_z),
            torch.from_numpy(lab_full_z),
            torch.from_numpy(lab_seed_z),
            seed_idx,
            z_start,
            z_end,
            case_id,
            source,
        )

def build_labeled_eval_loader(source_roots, eval_sources, num_workers=4, cache_rate=1.0):
    """
    Creates a test-style loader (batch_size=1) but only for labeled cases in eval_sources.
    """
    all_items = []
    for src_name, root_dir in source_roots:
        all_items.extend(collect_source_items_mixed(src_name, root_dir))

    labeled = [it for it in all_items if it.get("labeled", False)]
    eval_set = set(eval_sources)
    pool = [it for it in labeled if it["source"] in eval_set]
    if len(pool) == 0:
        raise RuntimeError(f"No labeled eval items found for sources={eval_sources}")

    test_lab = [{"image": it["image"], "label_full": it["label_full"], "source": it["source"]} for it in pool]
    cached = CacheDataset(test_lab, base_img_lab_tf, cache_rate=cache_rate, num_workers=num_workers)
    ds = EvalSingleSliceFromCache(cached)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # keep deterministic / avoid MONAI meta issues
        collate_fn=my_collate_drop_none,
        drop_last=False,
    )
    return loader


# ============================================================
# 6) Inference + per-case analysis + dataset boxplots
# ============================================================
def run_eval(
    eval_loader,
    model,
    out_root,
    dice_mode="multiclass_mean",
    overlay_organ_only=False,
    save_img_gt_pred=True,
    use_verification=True,
    apply_edge_profile=True,
    tag="eval",
):
    os.makedirs(out_root, exist_ok=True)
    per_case_dir = os.path.join(out_root, f"per_case_{tag}")
    plots_dir = os.path.join(per_case_dir, "plots")
    nifti_dir = os.path.join(per_case_dir, "niftis")
    box_dir = os.path.join(out_root, f"boxplots_{tag}")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(nifti_dir, exist_ok=True)

    model.eval()
    summary = {}

    all_case_mean_dice = []
    all_case_mean_sd = []
    all_case_mean_hd = []

    with torch.no_grad():
        for bidx, batch in enumerate(eval_loader):
            (
                img_vol, lab_full, lab_seed,
                seed_idx, z_start, z_end, case_id, source
            ) = batch

            # unwrap case/source
            case_id = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
            source = source[0] if isinstance(source, (list, tuple)) else source
            if isinstance(case_id, np.ndarray): case_id = case_id.item()
            if isinstance(source, np.ndarray): source = source.item()

            seed = int(seed_idx.item()) if torch.is_tensor(seed_idx) else int(seed_idx)
            z0 = int(z_start.item()) if torch.is_tensor(z_start) else int(z_start)
            z1 = int(z_end.item()) if torch.is_tensor(z_end) else int(z_end)

            # numpy volumes (batch=1)
            img_zhw = img_vol[0].cpu().numpy().astype(np.float32)      # [Z,H,W]
            gt_zhw  = lab_full[0].cpu().numpy().astype(np.int32)       # [Z,H,W]
            seed_zhw = lab_seed[0].cpu().numpy().astype(np.int32)      # [Z,H,W]

            Zi, Zg = img_zhw.shape[0], gt_zhw.shape[0]
            Z = min(Zi, Zg)
            if Zi != Zg:
                img_zhw = img_zhw[:Z]
                gt_zhw = gt_zhw[:Z]
                seed_zhw = seed_zhw[:Z]
                seed = min(seed, Z - 1)
                z0 = min(z0, Z - 1)
                z1 = min(z1, Z - 1)

            H, W = img_zhw.shape[1], img_zhw.shape[2]

            # IMPORTANT: build GPU tensor from the *cropped* volume
            img_gpu = torch.from_numpy(img_zhw[None, ...]).float().cuda()  # [1,Z,H,W]

            pred_zhw = np.zeros_like(gt_zhw, dtype=np.int32)
            pred_zhw[seed] = seed_zhw[seed].copy()

            # verification buffers
            template = []
            mask_collection = pred_zhw[seed:seed+1].copy()              # [F,H,W]
            img_collection  = img_zhw[seed:seed+1][:, None, ...].copy() # [F,1,H,W]

            # backward
            for i in range(seed, 0, -1):
                frame1 = img_gpu[:, i:i+1]
                frame2 = img_gpu[:, i-1:i]

                if apply_edge_profile:
                    frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                mask1 = torch.from_numpy(pred_zhw[i:i+1]).unsqueeze(0).float().cuda()
                logits = model(frame1, frame2, mask1)
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).cpu().numpy().astype(np.int32)

                if use_verification:
                    img_next = img_zhw[i-1:i]  # [1,H,W]
                    out, template = verification_module(out, mask_collection, img_next, img_collection, template)

                pred_zhw[i-1] = out
                mask_collection = np.concatenate([mask_collection, out[None, ...]], axis=0)
                img_collection  = np.concatenate([img_collection, img_zhw[i-1:i][:, None, ...]], axis=0)

                if np.max(out) == 0:
                    break

            # forward
            template = []
            mask_collection = pred_zhw[seed:seed+1].copy()
            img_collection  = img_zhw[seed:seed+1][:, None, ...].copy()

            for i in range(seed, Z - 1):
                frame1 = img_gpu[:, i:i+1]
                frame2 = img_gpu[:, i+1:i+2]

                if apply_edge_profile:
                    frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                mask1 = torch.from_numpy(pred_zhw[i:i+1]).unsqueeze(0).float().cuda()
                logits = model(frame1, frame2, mask1)
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).cpu().numpy().astype(np.int32)

                if use_verification:
                    img_next = img_zhw[i+1:i+2]
                    out, template = verification_module(out, mask_collection, img_next, img_collection, template)

                # guard: avoid index error if Z changes unexpectedly
                if (i + 1) < pred_zhw.shape[0]:
                    pred_zhw[i+1] = out
                else:
                    break

                mask_collection = np.concatenate([mask_collection, out[None, ...]], axis=0)
                img_collection  = np.concatenate([img_collection, img_zhw[i+1:i+2][:, None, ...]], axis=0)

                if np.max(out) == 0:
                    break

            # slice-wise curves only in organ range
            zs_d, dice_vals = compute_slicewise_dice_in_range(pred_zhw, gt_zhw, z0, z1, mode=dice_mode)
            zs_s, sd_vals = compute_slicewise_surface_distance_in_range(pred_zhw, gt_zhw, z0, z1)
            zs_h, hd_vals   = compute_slicewise_hausdorff_in_range(pred_zhw, gt_zhw, z0, z1)

            mean_dice = float(np.nanmean(dice_vals)) if dice_vals.size else 0.0
            mean_sd = float(np.nanmean(sd_vals)) if sd_vals.size else np.nan
            mean_hd = float(np.nanmean(hd_vals)) if hd_vals.size else np.nan

            all_case_mean_dice.append(mean_dice)
            all_case_mean_sd.append(mean_sd)
            all_case_mean_hd.append(mean_hd)

            # save per-case plots
            case_tag = f"{source}_{case_id}"
            
            plot_curve(
                case_tag=case_tag,
                zs=zs_d,
                ys=dice_vals,
                seed_idx=seed,
                ylabel="Slice-wise Dice",
                title=f"{case_tag}: Dice in organ range",
                out_path=os.path.join(plots_dir, f"{case_tag}_dice.png"),
            )
            plot_curve(
                case_tag=case_tag,
                zs=zs_s,
                ys=sd_vals,
                seed_idx=seed,
                ylabel="Mean Surface Distance (pixels)",
                title=f"{case_tag}: Surface distance in organ range",
                out_path=os.path.join(plots_dir, f"{case_tag}_surface_distance.png"),
            )
            plot_curve(
                case_tag=case_tag,
                zs=zs_h,
                ys=hd_vals,
                seed_idx=seed,
                ylabel="Hausdorff Distance (pixels)",
                title=f"{case_tag}: Hausdorff in organ range",
                out_path=os.path.join(plots_dir, f"{case_tag}_hausdorff.png"),
            )


            # save niftis
            src_out = os.path.join(nifti_dir, str(source))
            os.makedirs(src_out, exist_ok=True)

            if save_img_gt_pred:
                save_nifti(gt_zhw, os.path.join(src_out, f"{case_id}_gt.nii.gz"), dtype=np.int16)
                save_nifti(pred_zhw, os.path.join(src_out, f"{case_id}_pred.nii.gz"), dtype=np.int16)
                save_nifti(seed_zhw, os.path.join(src_out, f"{case_id}_seed.nii.gz"), dtype=np.int16)

            save_overlap_3d_nifti(
                gt_zhw=gt_zhw,
                pred_zhw=pred_zhw,
                out_path=os.path.join(src_out, f"{case_id}_overlap.nii.gz"),
                organ_only=overlay_organ_only,
                z_start=z0,
                z_end=z1,
            )

            summary[case_tag] = {
                "source": str(source),
                "case_id": str(case_id),
                "seed_idx": int(seed),
                "z_start": int(z0),
                "z_end": int(z1),
                "dice_mode": dice_mode,
                "mean_dice_organ_range": mean_dice,
                "mean_surface_distance_organ_range_pixels": mean_sd,
                "plots": {
                    "dice": os.path.join(plots_dir, f"{case_tag}_dice.png"),
                    "surface_distance": os.path.join(plots_dir, f"{case_tag}_surface_distance.png"),
                    "hausdorff": os.path.join(plots_dir, f"{case_tag}_hausdorff.png"),
                },
            }

            print(f"[{tag}][{bidx}] {case_tag} | mean_dice={mean_dice:.4f} | mean_sd(px)={mean_sd:.4f} | mean_hd(px)={mean_hd:.4f}")

    # dataset-level boxplots
    boxplot_metrics(all_case_mean_dice, all_case_mean_sd, all_case_mean_hd, out_dir=box_dir, tag=tag)

    # dump json
    with open(os.path.join(out_root, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# 7) CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Sli2Vol evaluation + visualization (Dice + surface distance)")

    # sources: arbitrary count
    p.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help='Space-separated list like SourceA=/pathA SourceB=/pathB ...'
    )

    # split setting
    p.add_argument("--setting", type=str, default="B", choices=["A", "B"])
    p.add_argument("--train_sources", type=str, default="SourceA",
                   help='Comma-separated train sources (for setting B)')
    p.add_argument("--test_sources", type=str, default="SourceB",
                   help='Comma-separated test sources (for setting B)')

    # model
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--in_channels", type=int, default=48)
    p.add_argument("--R", type=int, default=6)

    # loader params
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=10)

    # eval selection for boxplots
    p.add_argument(
        "--boxplot_on",
        type=str,
        default="train",
        choices=["train", "test"],
        help='If "train": boxplots on labeled cases from train_sources. If "test": boxplots on test_loader.'
    )

    # options
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--dice_mode", type=str, default="multiclass_mean", choices=["multiclass_mean", "binary_union"])
    p.add_argument("--overlay_organ_only", action="store_true")
    p.add_argument("--no_verification", action="store_true")
    p.add_argument("--no_edge_profile", action="store_true")

    return p.parse_args()

def parse_sources(sources_list):
    out = []
    for s in sources_list:
        if "=" not in s:
            raise ValueError(f"Bad --sources entry: {s} (expected Name=/path)")
        name, path = s.split("=", 1)
        name = name.strip()
        path = path.strip()
        out.append((name, path))
    return out


# ============================================================
# 8) Main
# ============================================================
def main():
    p = argparse.ArgumentParser("Sli2Vol evaluation + visualization (Dice + surface distance)")

    # data
    p.add_argument("--sourceA_root", type=str, required=True,
                   help="Root for SourceA (must contain imagesTr/ and optionally labelsTr/)")
    p.add_argument("--sourceB_root", type=str, required=True,
                   help="Root for SourceB (must contain imagesTr/ and optionally labelsTr/)")
    p.add_argument("--sourceC_root", type=str, required=False,
                   help="Root for SourceC (must contain imagesTr/ and optionally labelsTr/)")
    p.add_argument("--sourceD_root", type=str, required=False,
                   help="Root for SourceD (must contain imagesTr/ and optionally labelsTr/)")
    p.add_argument("--sourceE_root", type=str, required=False,
                     help="Root for SourceE (must contain imagesTr/ and optionally labelsTr/)")
    p.add_argument("--sourceF_root", type=str, required=False,
                        help="Root for SourceF (must contain imagesTr/ and optionally labelsTr/)")

    p.add_argument("--setting", type=str, default="B", choices=["A", "B"],
                   help="Split setting A or B")
    p.add_argument("--train_sources", type=str, default="SourceA",
                   help='Comma-separated train sources for setting B, e.g. "SourceA" or "SourceA,SourceB"')
    p.add_argument("--test_sources", type=str, default="SourceB",
                   help='Comma-separated test sources for setting B, e.g. "SourceB"')
    
    p.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--in_channels", type=int, default=48)
    p.add_argument("--R", type=int, default=6)

    # loader params
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=10)

    # eval selection for boxplots
    p.add_argument(
        "--boxplot_on",
        type=str,
        default="test",
        choices=["train", "test"],
        help='If "train": boxplots on labeled cases from train_sources. If "test": boxplots on test_loader.'
    )

    # options
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--dice_mode", type=str, default="multiclass_mean", choices=["multiclass_mean", "binary_union"])
    p.add_argument("--overlay_organ_only", action="store_true")
    p.add_argument("--no_verification", action="store_true")
    p.add_argument("--no_edge_profile", action="store_true")

    args = p.parse_args()

    setting = args.setting.upper()

    #Parse Sources
    source_roots = []
    if args.sourceA_root is not None:
        source_roots.append(("SourceA", args.sourceA_root))
    if args.sourceB_root is not None:
        source_roots.append(("SourceB", args.sourceB_root))
    if args.sourceC_root is not None:
        source_roots.append(("SourceC", args.sourceC_root))
    if args.sourceD_root is not None:
        source_roots.append(("SourceD", args.sourceD_root))
    if args.sourceE_root is not None:
        source_roots.append(("SourceE", args.sourceE_root))
    if args.sourceF_root is not None:
        source_roots.append(("SourceF", args.sourceF_root))

    if not source_roots:
        raise ValueError("At least one source root must be provided")
    
    source_roots = [(name, path) for name, path in source_roots if path is not None]

    train_sources = [x.strip() for x in args.train_sources.split(",") if x.strip()]
    test_sources = [x.strip() for x in args.test_sources.split(",") if x.strip()]

    # Build your normal loaders (gives you a real test_loader)
    if setting == "A":
        train_loader, val_loader, test_loader = build_multi_source_loaders(
            source_roots=source_roots,
            setting="A",
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_cache_rate=0.1,
            test_cache_rate=0.0
        )
    else:
        train_loader, val_loader, test_loader = build_multi_source_loaders(
            source_roots=source_roots,
            setting="B",
            train_sources=train_sources,
            test_sources=test_sources,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_cache_rate=0.1,
            test_cache_rate=0.0
        )


    # Load model
    model = Correspondence_Flow_Net(in_channels=args.in_channels, is_training=False, R=args.R).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()

    os.makedirs(args.out_root, exist_ok=True)

    use_verification = not args.no_verification
    use_edge = not args.no_edge_profile

    # Per-case + boxplots on the requested evaluation set
    if args.boxplot_on == "test":
        eval_loader = test_loader
        tag = "test_loader"
    else:
        # build labeled-only eval loader from train_sources
        eval_loader = train_loader

    run_eval(
        eval_loader=eval_loader,
        model=model,
        out_root=args.out_root,
        dice_mode=args.dice_mode,
        overlay_organ_only=args.overlay_organ_only,
        save_img_gt_pred=True,
        use_verification=use_verification,
        apply_edge_profile=use_edge,
        tag=tag,
    )

    print(f"\nDone. Outputs at: {args.out_root}\n")

if __name__ == "__main__":
    main()
