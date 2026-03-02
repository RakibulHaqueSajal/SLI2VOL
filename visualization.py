import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib

from dataset import *
from model import *
from data_creator.pancreas_dataset import build_multi_source_loaders
from val_pancreas import verification_module  # your existing verification module

# ============================================================
# Dice helpers
# ============================================================
def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return float((2.0 * inter + eps) / (den + eps))

def dice_multiclass_slice(pred_hw: np.ndarray, gt_hw: np.ndarray, classes=None) -> float:
    """
    Single dice value for a slice by averaging per-class dice.
    If classes=None: uses GT-present classes (excluding 0) on that slice.
    """
    if classes is None:
        cls = np.unique(gt_hw)
        cls = cls[cls != 0]
    else:
        cls = np.array(list(classes), dtype=np.int32)

    if cls.size == 0:
        # if GT empty on this slice: return 1 if pred also empty else 0
        return 1.0 if np.max(pred_hw) == 0 else 0.0

    ds = []
    for c in cls:
        ds.append(dice_binary(pred_hw == c, gt_hw == c))
    return float(np.mean(ds))

def compute_slicewise_dice_in_range(
    pred_zhw: np.ndarray,
    gt_zhw: np.ndarray,
    z_start: int,
    z_end: int,
    mode: str = "multiclass_mean",
):
    """
    Compute dice only for z in [z_start, z_end].
    Returns:
      zs: np.ndarray of slice indices
      dice_vals: np.ndarray dice per slice (aligned with zs)
    """
    z_start = int(z_start)
    z_end = int(z_end)
    z_start = max(0, z_start)
    z_end = min(gt_zhw.shape[0] - 1, z_end)

    zs = np.arange(z_start, z_end + 1, dtype=np.int32)
    dice_vals = np.zeros((len(zs),), dtype=np.float32)

    if mode == "binary_union":
        for j, z in enumerate(zs):
            dice_vals[j] = dice_binary(pred_zhw[z] > 0, gt_zhw[z] > 0)

    elif mode == "multiclass_mean":
        for j, z in enumerate(zs):
            dice_vals[j] = dice_multiclass_slice(pred_zhw[z], gt_zhw[z], classes=None)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return zs, dice_vals

# ============================================================
# Plotting
# ============================================================
def plot_slicewise_dice_in_range(
    case_id: str,
    zs: np.ndarray,
    dice_vals: np.ndarray,
    seed_idx: int,
    out_path: str,
):
    plt.figure()
    plt.plot(zs, dice_vals, marker="o")
    plt.axvline(seed_idx, linestyle="--", label="Seed slice")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Slice index (z)")
    plt.ylabel("Slice-wise Dice")
    plt.title(f"{case_id}: Slice-wise Dice (organ range)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
# ============================================================
# 3D overlay saver (overlap code volume)
# ============================================================
def save_overlap_3d_nifti(
    gt_zhw: np.ndarray,
    pred_zhw: np.ndarray,
    out_path: str,
    organ_only: bool = False,
    z_start: int = None,
    z_end: int = None,
):
    """
    Saves a 3D overlap-coded label volume:
      0 = background
      1 = GT only
      2 = Pred only
      3 = Both (GT & Pred)

    If organ_only=True, slices outside [z_start,z_end] are zeroed.
    """
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

def save_nifti(vol_zhw: np.ndarray, out_path: str, dtype=np.int16):
    nii = nib.Nifti1Image(vol_zhw.astype(dtype), np.eye(4))
    nib.save(nii, out_path)

# ============================================================
# Main visualization driver
# ============================================================
def analyze_testset_slicewise_and_3doverlays(
    test_loader,
    model,
    out_root: str,
    dice_mode: str = "multiclass_mean",  # or "binary_union"
    overlay_organ_only: bool = False,
    save_img_gt_pred: bool = True,
    use_verification: bool = True,
):
    """
    For each test sample:
      1) Runs seed-propagation inference (backward + forward) with optional verification_module.
      2) Computes slice-wise dice ONLY in [z_start, z_end] and saves a plot.
      3) Saves a 3D overlap-coded overlay volume as .nii.gz.
      4) Optionally saves img/pred/gt volumes as .nii.gz for 3D Slicer.

    Requirements:
      - model(frame1, frame2, mask1) returns logits
      - edge_profile exists (imported from your codebase)
      - verification_module imported from val_pancreas
    """
    os.makedirs(out_root, exist_ok=True)
    plots_dir = os.path.join(out_root, "slicewise_dice_plots_organ_range")
    overlay_dir = os.path.join(out_root, "overlays_3d")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    model.eval()
    results = {}

    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            (
                img_vol,      # [1,Z,H,W] float
                lab_full,     # [1,Z,H,W] int
                lab_seed,     # [1,Z,H,W] int (seed slice only)
                seed_idx,
                z_start,
                z_end,
                case_id,
                source,
            ) = batch

            # unwrap strings
            case_id = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
            source = source[0] if isinstance(source, (list, tuple)) else source
            if isinstance(case_id, np.ndarray): case_id = case_id.item()
            if isinstance(source, np.ndarray): source = source.item()

            seed = int(seed_idx.item()) if torch.is_tensor(seed_idx) else int(seed_idx)
            z0 = int(z_start.item()) if torch.is_tensor(z_start) else int(z_start)
            z1 = int(z_end.item()) if torch.is_tensor(z_end) else int(z_end)

            img_zhw = img_vol[0].cpu().numpy().astype(np.float32)      # [Z,H,W] in [0,1]
            gt_zhw  = lab_full[0].cpu().numpy().astype(np.int32)       # [Z,H,W]
            seedmask_zhw = lab_seed[0].cpu().numpy().astype(np.int32)  # [Z,H,W]
            Z, H, W = img_zhw.shape
            Zi = img_zhw.shape[0]
            Zg = gt_zhw.shape[0]

            if Zi != Zg:
                Zm = min(Zi, Zg)
                print(f"[WARN] Z mismatch for {source}/{case_id}: img Z={Zi}, gt Z={Zg}. Cropping both to Z={Zm}.")
                img_zhw = img_zhw[:Zm]
                gt_zhw = gt_zhw[:Zm]
                seedmask_zhw = seedmask_zhw[:Zm]
                Z = Zm
            else:
                Z = Zi


            # -------------------------
            # Inference: seed propagation
            # -------------------------
            pred_zhw = np.zeros_like(gt_zhw, dtype=np.int32)
            pred_zhw[seed] = seedmask_zhw[seed].copy()

            img_gpu = img_vol.float().cuda()  # [1,Z,H,W]

            # verification history buffers (match old pipeline expectations)
            template = []
            mask_collection = pred_zhw[seed:seed+1].copy()              # [F,H,W]
            img_collection  = img_zhw[seed:seed+1][:, None, ...].copy() # [F,1,H,W]

            # backward
            for i in range(seed, 0, -1):
                frame1 = img_gpu[:, i:i+1]        # [1,1,H,W]
                frame2 = img_gpu[:, i-1:i]        # [1,1,H,W]

                frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                mask1 = torch.from_numpy(pred_zhw[i:i+1]).unsqueeze(0).float().cuda()  # [1,1,H,W]
                logits = model(frame1, frame2, mask1)
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).cpu().numpy().astype(np.int32)  # [H,W]

                if use_verification:
                    img_next = img_zhw[i-1:i]  # [1,H,W] (C=1)
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

                frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                mask1 = torch.from_numpy(pred_zhw[i:i+1]).unsqueeze(0).float().cuda()
                logits = model(frame1, frame2, mask1)
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).cpu().numpy().astype(np.int32)

                if use_verification:
                    img_next = img_zhw[i+1:i+2]
                    out, template = verification_module(out, mask_collection, img_next, img_collection, template)

                pred_zhw[i+1] = out

                mask_collection = np.concatenate([mask_collection, out[None, ...]], axis=0)
                img_collection  = np.concatenate([img_collection, img_zhw[i+1:i+2][:, None, ...]], axis=0)

                if np.max(out) == 0:
                    break

            # -------------------------
            # (1) Slice-wise dice plot ONLY in organ range [z0,z1]
            # -------------------------
            zs, dice_vals = compute_slicewise_dice_in_range(
                pred_zhw=pred_zhw,
                gt_zhw=gt_zhw,
                z_start=z0,
                z_end=z1,
                mode=dice_mode,
            )

            plot_path = os.path.join(plots_dir, f"{source}_{case_id}_dice_organ_range.png")
            plot_slicewise_dice_in_range(
                case_id=f"{source}/{case_id}",
                zs=zs,
                dice_vals=dice_vals,
                seed_idx=seed,
                out_path=plot_path,
            )

            # -------------------------
            # (2) 3D overlay-coded volume
            # -------------------------
            src_dir = os.path.join(overlay_dir, str(source))
            os.makedirs(src_dir, exist_ok=True)

            overlay_path = os.path.join(src_dir, f"{case_id}_overlap.nii.gz")
            save_overlap_3d_nifti(
                gt_zhw=gt_zhw,
                pred_zhw=pred_zhw,
                out_path=overlay_path,
                organ_only=overlay_organ_only,
                z_start=z0,
                z_end=z1,
            )

            # Optional: save image/pred/gt too (for 3D Slicer)
            if save_img_gt_pred:
                save_nifti(gt_zhw,  os.path.join(src_dir, f"{case_id}_gt.nii.gz"),  dtype=np.int16)
                save_nifti(pred_zhw, os.path.join(src_dir, f"{case_id}_pred.nii.gz"), dtype=np.int16)

            # bookkeeping
            results[case_id] = {
                "source": str(source),
                "seed_idx": int(seed),
                "z_start": int(z0),
                "z_end": int(z1),
                "dice_mode": dice_mode,
                "mean_dice_organ_range": float(np.mean(dice_vals)) if len(dice_vals) else 0.0,
                "plot_path": plot_path,
                "overlay_path": overlay_path,
            }

            print(f"[{bidx}] {source}/{case_id} -> plot: {plot_path} | overlay: {overlay_path}")

    with open(os.path.join(out_root, "analysis_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    source_roots = [
        ("SourceA", "/uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas"),
        ("SourceB", "/uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas"),
    ]

    # pick A or B depending on your experiment
    # train_loader, val_loader, test_loader = build_multi_source_loaders(
    #     source_roots,
    #     setting="A",
    #     seed=10,
    #     batch_size=4,
    #     num_workers=4,
    # )
    # If you want domain shift:
    train_loader, val_loader, test_loader = build_multi_source_loaders(
        source_roots,
        setting="B",
        train_sources=["SourceA"],
        test_sources=["SourceB"],
        seed=10,
        batch_size=4,
        num_workers=4,
    )

    # Load model
    mother_path = "/uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Trial_B_Final_Increased_Epoch/"
    model_path = os.path.join(mother_path, "best_model.pth")

    in_channels = 48
    model = Correspondence_Flow_Net(in_channels=in_channels, is_training=False, R=6).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    out_root = os.path.join(mother_path, "predicitions_analysis_3d_overlay")
    os.makedirs(out_root, exist_ok=True)

    analyze_testset_slicewise_and_3doverlays(
        test_loader=test_loader,
        model=model,
        out_root=out_root,
        dice_mode="multiclass_mean",     # or "binary_union"
        overlay_organ_only=False,        # True -> zero out overlay outside z_start..z_end
        save_img_gt_pred=True,           # save img/gt/pred along with overlap
        use_verification=True,           # keep verification_module refinement
    )
