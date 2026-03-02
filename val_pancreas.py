import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

from scipy.spatial.distance import dice as dice_distance
from scipy.ndimage import binary_dilation, binary_fill_holes, binary_erosion
from skimage.measure import label
from skimage.morphology import convex_hull_image
from model import *
from data_creator.pancreas_dataset import build_multi_source_loaders
from dataset import *
# ------------------------------------------------------------
# 1) Verification module (ported from your old script)
# ------------------------------------------------------------
def verification_module(mask_img, mask_collection, img, img_collection, template):
    """
    mask_img:        (H, W) predicted label map for next slice (ints)
    mask_collection: (F, H, W) history of masks (ints), last element is most recent history
    img:             (C, H, W) current image slice (C usually 1)
    img_collection:  (F, C, H, W) history of image slices aligned with mask_collection
    template:        unused in your refine(), kept for compatibility

    Returns:
      mask_refine: (H, W) refined mask (ints)
      template: passthrough
    """
    def getLargestCC(segmentation):
        labels_cc = label(segmentation)
        if labels_cc.max() == 0:
            return segmentation
        largestCC = labels_cc == (np.argmax(np.bincount(labels_cc.flat)[1:]) + 1)
        return (largestCC > 0).astype(int)

    def refine(segmentation, original):
        # Your old code returns segmentation directly (no convex hull applied)
        _ = convex_hull_image(segmentation)  # kept to match the original structure
        return segmentation

    num_class = int(np.max(mask_collection[-1]))  # IMPORTANT: this assumes history has class labels
    mask_refine = np.zeros(mask_img.shape, dtype=np.int32)

    for nc in range(1, num_class + 1):
        mask_history = (mask_collection[-1] == nc).astype(np.float32)  # (H,W)
        mask_img_temp = (mask_img == nc).astype(np.float32)            # (H,W)

        # Negative ring around history mask
        mask_neg = binary_dilation(mask_history, iterations=5).astype(np.float32) - mask_history
        mask_neg = (mask_neg > 0).astype(np.float32)

        # Tile to channels
        mask_neg_c = np.tile(mask_neg[None, ...], (img.shape[0], 1, 1))     # (C,H,W)
        mask_pos_c = np.tile(mask_history[None, ...], (img.shape[0], 1, 1)) # (C,H,W)

        # Compute per-channel mean intensity over positive and negative regions from LAST history frame
        # img_collection[-1, i] is (H,W) for channel i
        # Handle edge cases: if mask is empty, mean() becomes nan -> we’ll guard later.
        pos_means = []
        neg_means = []
        for ci in range(mask_pos_c.shape[0]):
            pos_vals = img_collection[-1, ci][mask_pos_c[ci] == 1]
            neg_vals = img_collection[-1, ci][mask_neg_c[ci] == 1]
            pos_means.append(np.mean(pos_vals) if pos_vals.size > 0 else np.nan)
            neg_means.append(np.mean(neg_vals) if neg_vals.size > 0 else np.nan)

        pos_means = np.array(pos_means, dtype=np.float32)
        neg_means = np.array(neg_means, dtype=np.float32)

        feature_positive = np.ones_like(img, dtype=np.float32) * pos_means[:, None, None]
        feature_negative = np.ones_like(img, dtype=np.float32) * neg_means[:, None, None]

        # Squared distance to pos/neg prototypes
        feature_positive = np.sum((img - feature_positive) ** 2, axis=0)  # (H,W)
        feature_negative = np.sum((img - feature_negative) ** 2, axis=0)  # (H,W)

        positive_likelihood = np.zeros_like(feature_positive, dtype=np.float32)
        positive_likelihood[feature_positive < feature_negative] = 1.0

        # constrain to predicted region for this class
        positive_likelihood = positive_likelihood * mask_img_temp
        positive_likelihood[positive_likelihood > 0] = 1.0
        positive_likelihood[~np.isfinite(positive_likelihood)] = 0.0

        # original had these lines commented; keep behavior
        # positive_likelihood = getLargestCC(positive_likelihood)
        # positive_likelihood = refine(positive_likelihood, mask_img_temp)

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)

        positive_likelihood = binary_dilation(positive_likelihood, iterations=2)
        positive_likelihood = binary_erosion(positive_likelihood, iterations=2)

        positive_likelihood = binary_fill_holes(positive_likelihood).astype(np.int32)

        mask_refine[positive_likelihood > 0] = nc

    return mask_refine, template


# ------------------------------------------------------------
# 2) Metrics + I/O helpers
# ------------------------------------------------------------
def dice_score_binary(pred: np.ndarray, gt: np.ndarray, eps=1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return (2.0 * inter + eps) / (den + eps)

def dice_per_class(mask_pred: np.ndarray, mask_gt: np.ndarray) -> dict:
    # both [Z,H,W] int
    n_cls = int(mask_gt.max())
    out = {}
    for c in range(1, n_cls + 1):
        out[c] = float(dice_score_binary(mask_pred == c, mask_gt == c))
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_nifti(vol_zhw: np.ndarray, path: str):
    nii = nib.Nifti1Image(vol_zhw.astype(np.int16), np.eye(4))
    nib.save(nii, path)


# ------------------------------------------------------------
# 3) Test runner using YOUR new test_loader
# ------------------------------------------------------------
def run_test_with_verification(
    test_loader,
    model,
    out_root: str,
    apply_edge_profile=True,
    verification=True,
    verbose=True,
):
    """
    test_loader yields (from your PancreasTestSingleSliceFromCache):
      img_vol:   [1, Z, H, W] float32 in [0,1]
      lab_full:  [1, Z, H, W] uint8/int (multi-class)
      lab_seed:  [1, Z, H, W] uint8/int (only seed slice non-zero)
      seed_idx, z_start, z_end, case_id, source

    This replicates your old test pipeline logic:
      - seed slice is provided
      - propagate backward and forward using CorrFlowNet
      - refine each step using verification_module (optional but enabled here)
      - save pred/gt/seed
      - report dice per class
    """
    model.eval()
    ensure_dir(out_root)

    results = {}

    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            (
                img_vol,     # [1,Z,H,W]
                lab_full,    # [1,Z,H,W]
                lab_seed,    # [1,Z,H,W]
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

            # tensors -> numpy for bookkeeping
            img_np = img_vol[0].cpu().numpy().astype(np.float32)      # [Z,H,W] in [0,1]
            gt = lab_full[0].cpu().numpy().astype(np.int32)           # [Z,H,W]
            seed_mask_vol = lab_seed[0].cpu().numpy().astype(np.int32)# [Z,H,W]

            Z, H, W = img_np.shape

            # init pred volume with only the seed slice
            pred = np.zeros_like(gt, dtype=np.int32)
            pred[seed] = seed_mask_vol[seed].copy()

            # push image volume to GPU for model
            img_gpu = img_vol.float().cuda()  # [1,Z,H,W]

            # Build initial collections for verification module
            # Old code uses img_collection/mask_collection growing over time;
            # We'll keep the same pattern.
            template = []

            # history starts with the SEED slice
            # mask_collection: (F,H,W)
            mask_collection = pred[seed:seed+1].copy()

            # img_collection: (F,C,H,W) ; here C=1
            img_collection = img_np[seed:seed+1][..., None]           # [1,H,W,1]  (wrong)
            # fix to [F,C,H,W]
            img_collection = np.transpose(img_np[seed:seed+1][None, ...], (0, 1, 2, 3))  # [1,Z?] nope

            # Correct: img_np slice is [H,W], want [F,C,H,W] where C=1
            img_collection = img_np[seed:seed+1]                      # [1,H,W]
            img_collection = img_collection[:, None, ...]             # [1,1,H,W]

            # -------------------------
            # BACKWARD propagation
            # -------------------------
            backward_times = []
            for i in range(seed, 0, -1):
                # frame1 = i, frame2 = i-1
                frame1 = img_gpu[:, i:i+1]   # [1,1,H,W]
                frame2 = img_gpu[:, i-1:i]   # [1,1,H,W]

                t0 = time.time()
                if apply_edge_profile:
                    frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                # mask of current slice i
                mask1 = torch.from_numpy(pred[i:i+1]).unsqueeze(0).float().cuda()  # [1,1,H,W]

                logits = model(frame1, frame2, mask1)  # [1,C,h,w]
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).detach().cpu().numpy().astype(np.int32)  # [H,W]

                if verification:
                    # img for next slice: old code passed (img_vol)[i-1:i] which is [1,H,W] as "C=1"
                    img_next = img_np[i-1:i]       # [1,H,W]
                    out, template = verification_module(
                        out,
                        mask_collection,          # [F,H,W]
                        img_next,                 # [C,H,W] where C=1
                        img_collection,           # [F,1,H,W]
                        template
                    )

                pred[i-1] = out

                # update collections
                mask_collection = np.concatenate([mask_collection, out[None, ...]], axis=0)              # (F+1,H,W)
                img_collection  = np.concatenate([img_collection, img_np[i-1:i][:, None, ...]], axis=0)  # (F+1,1,H,W)

                backward_times.append(time.time() - t0)

                if np.max(out) == 0:
                    break

            # -------------------------
            # FORWARD propagation
            # -------------------------
            template = []
            mask_collection = pred[seed:seed+1].copy()
            img_collection  = img_np[seed:seed+1][:, None, ...].copy()

            forward_times = []
            for i in range(seed, Z - 1):
                # frame1 = i, frame2 = i+1
                frame1 = img_gpu[:, i:i+1]
                frame2 = img_gpu[:, i+1:i+2]

                t0 = time.time()
                if apply_edge_profile:
                    frame1, frame2 = edge_profile([frame1, frame2], False, 3, 1)

                mask1 = torch.from_numpy(pred[i:i+1]).unsqueeze(0).float().cuda()

                logits = model(frame1, frame2, mask1)
                logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
                out = torch.argmax(logits, 1).squeeze(0).detach().cpu().numpy().astype(np.int32)

                if verification:
                    img_next = img_np[i+1:i+2]  # [1,H,W] as C=1
                    out, template = verification_module(out, mask_collection, img_next, img_collection, template)
            
                pred[i+1] = out

                mask_collection = np.concatenate([mask_collection, out[None, ...]], axis=0)
                img_collection  = np.concatenate([img_collection, img_np[i+1:i+2][:, None, ...]], axis=0)

                forward_times.append(time.time() - t0)

                if np.max(out) == 0:
                    break

            # -------------------------
            # Metrics + Save
            # -------------------------
            per_cls = dice_per_class(pred, gt)
            mean_dice = float(np.mean(list(per_cls.values()))) if len(per_cls) else 0.0

            if verbose:
                print("=" * 90)
                print(f"[{bidx}] case={case_id} source={source} Z={Z} seed={seed} z_range=[{z0},{z1}]")
                print(f"  mean_dice={mean_dice:.4f} | "
                      f"avg_step_time backward={np.mean(backward_times) if backward_times else 0:.4f}s "
                      f"forward={np.mean(forward_times) if forward_times else 0:.4f}s")
                for c, dsc in per_cls.items():
                    print(f"   class {c}: dice={dsc:.4f}")

            results[case_id] = {
                "source": source,
                "seed_idx": seed,
                "z_start": z0,
                "z_end": z1,
                "dice_per_class": per_cls,
                "mean_dice": mean_dice,
            }

            # Save nifti: pred + gt + seed-only
            case_out = os.path.join(out_root, str(source))
            ensure_dir(case_out)
            save_nifti(pred, os.path.join(case_out, f"{case_id}_pred.nii.gz"))
            save_nifti(gt,   os.path.join(case_out, f"{case_id}_gt.nii.gz"))
            save_nifti(seed_mask_vol, os.path.join(case_out, f"{case_id}_seed.nii.gz"))

    # Optional: dump a JSON summary
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ------------------------------------------------------------
# 4) Example: glue it to your loader + model
# ------------------------------------------------------------
if __name__ == "__main__":
    # You already have build_multi_source_loaders, source_roots, etc.
    source_roots = [
        ("SourceA", "/uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas"),
        ("SourceB", "/uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas"),
    ]

    train_loader, val_loader, test_loader = build_multi_source_loaders(
        source_roots,
        setting="B",
        train_sources=["SourceA"],
        test_sources=["SourceB"],
        seed=10,
        batch_size=4,
        num_workers=4,
    )
    # train_loader, val_loader, test_loader = build_multi_source_loaders(
    #     source_roots,
    #     setting="A",
    #     seed=10,
    #     batch_size=4,
    #     num_workers=4,
    # )


    # Load your model
    mother_path = "/uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Trial_B_Final_Increased_Epoch/"
    model_path = os.path.join(mother_path, "best_model.pth")

    in_channels = 48
    model = Correspondence_Flow_Net(in_channels=in_channels, is_training=False, R=6).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    out_root = os.path.join(mother_path, "predictions_new_loader_verified")

    results = run_test_with_verification(
        test_loader=test_loader,
        model=model,
        out_root=out_root,
        apply_edge_profile=True,
        verification=True,
        verbose=True,
    )
