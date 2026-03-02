import os, glob
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
)

# ============================================================
# Collate
# ============================================================
def my_collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)

# ============================================================
# Deterministic cached transforms ONLY (safe for CacheDataset)
# ============================================================
# image -> torch [1,H,W,Z] in [0,1]
base_img_tf = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-300, a_max=400,
        b_min=0.0, b_max=1.0, clip=True
    ),
    EnsureTyped(keys=["image"]),
])

# image + label_full -> torch [1,H,W,Z], label typed
base_img_lab_tf = Compose([
    LoadImaged(keys=["image", "label_full"]),
    EnsureChannelFirstd(keys=["image", "label_full"]),
    Orientationd(keys=["image", "label_full"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-300, a_max=400,
        b_min=0.0, b_max=1.0, clip=True
    ),
    EnsureTyped(keys=["image", "label_full"]),
])

# ============================================================
# Basic intensity helpers (non-cached, random happens later)
# ============================================================
def cap_to_255(img_hu: np.ndarray, low: float, high: float) -> np.ndarray:
    img = np.clip(img_hu, low, high)
    img = (img - low) / (high - low + 1e-8)
    return (img * 255.0).astype(np.float32)

def gamma_contrast_255(img_255: np.ndarray, g: float) -> np.ndarray:
    x = np.clip(img_255 / 255.0, 0, 1)
    y = np.power(x, g)
    return (y * 255.0).astype(np.float32)

# ============================================================
# Case collection (supports MIXED labeled/unlabeled per source)
# ============================================================
def collect_source_items_mixed(
    source_name: str,
    root_dir: str,
    image_glob: str = "*.nii.gz",
) -> List[Dict[str, Any]]:
    """
    root_dir expected to contain imagesTr/ and (optionally) labelsTr/
    Mixed case supported: some images have labels, some don't.
    """
    imagesTr = os.path.join(root_dir, "imagesTr")
    labelsTr = os.path.join(root_dir, "labelsTr")

    imgs = sorted(glob.glob(os.path.join(imagesTr, image_glob)))
    if len(imgs) == 0:
        raise RuntimeError(f"[{source_name}] No images found in {imagesTr}/{image_glob}")

    has_labels_dir = os.path.isdir(labelsTr)

    items: List[Dict[str, Any]] = []
    for img_path in imgs:
        d: Dict[str, Any] = {"image": img_path, "source": source_name}

        lab_path = None
        if has_labels_dir:
            cand = os.path.join(labelsTr, os.path.basename(img_path))
            if os.path.exists(cand):
                lab_path = cand

        if lab_path is not None:
            d["labeled"] = True
            d["label_full"] = lab_path
        else:
            d["labeled"] = False

        items.append(d)

    n_lab = sum(int(it["labeled"]) for it in items)
    print(f"[collect|{source_name}] total={len(items)} labeled={n_lab} unlabeled={len(items)-n_lab}")
    return items

# ============================================================
# Splits
# ============================================================
def make_split(items: List[Dict[str, Any]], train_ratio=0.9, seed=10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_train = int(round(train_ratio * len(items)))
    n_train = max(1, min(n_train, len(items) - 1))
    train = [items[i] for i in idx[:n_train]]
    rest  = [items[i] for i in idx[n_train:]]
    return train, rest

def split_within_source(items: List[Dict[str, Any]], train_ratio=0.9, seed=10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split within each source, then merge. Assumes each item has 'source'.
    """
    rng = np.random.RandomState(seed)
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        by_src.setdefault(it["source"], []).append(it)

    train, test = [], []
    for src, lst in by_src.items():
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        n_train = int(round(train_ratio * len(lst)))
        n_train = max(1, min(n_train, len(lst) - 1))
        tr = [lst[i] for i in idx[:n_train]]
        te = [lst[i] for i in idx[n_train:]]
        print(f"[split|{src}] total={len(lst)} train={len(tr)} test={len(te)}")
        train += tr
        test  += te

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
##Getting the image name 
def _stem_niigz(p: str) -> str:
    base = os.path.basename(p)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]
# ============================================================
# Cached dataset wrappers (random sampling happens here)
# ============================================================
class PancreasSlicePairFromCache(torch.utils.data.Dataset):
    """
    cached_ds returns dict: {"image": torch [1,H,W,Z], "source": ...}
    Output: (frame1_input, frame2_input, frame1_target, frame2_target) each [1,256,256]
    """
    def __init__(
        self,
        cached_ds: CacheDataset,
        interval_choices=(2, 3, 4),
        crop_size=256,
        seed=10,
    ):
        self.cached_ds = cached_ds
        self.interval_choices = interval_choices
        self.crop_size = crop_size
        self.rng = np.random.RandomState(seed)

        import cv2
        self._cv2 = cv2

    def __len__(self):
        return len(self.cached_ds)

    def _random_crop_2d(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = a.shape
        ch = cw = self.crop_size
        if H < ch or W < cw:
            pad_h = max(0, ch - H)
            pad_w = max(0, cw - W)
            a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="edge")
            b = np.pad(b, ((0, pad_h), (0, pad_w)), mode="edge")
            H, W = a.shape
        y = self.rng.randint(0, H - ch + 1)
        x = self.rng.randint(0, W - cw + 1)
        return a[y:y+ch, x:x+cw], b[y:y+ch, x:x+cw]

    def __getitem__(self, idx: int):
        try:
            d = self.cached_ds[idx]
            img = d["image"]                       # torch [1,H,W,Z] in [0,1]
            img_np = img.squeeze(0).cpu().numpy().astype(np.float32)  # [H,W,Z]

            Z = img_np.shape[-1]
            if Z < 5:
                return None

            dz = int(self.rng.choice(self.interval_choices))
            z1 = int(self.rng.randint(0, Z - dz))
            z2 = z1 + dz

            f1 = img_np[:, :, z1]
            f2 = img_np[:, :, z2]

            # random resize [256..300]
            rs = int(self.rng.randint(256, 301))
            f1r = self._cv2.resize(f1, (rs, rs), interpolation=self._cv2.INTER_LINEAR)
            f2r = self._cv2.resize(f2, (rs, rs), interpolation=self._cv2.INTER_LINEAR)

            # random crop 256
            f1c, f2c = self._random_crop_2d(f1r, f2r)

            # map [0,1] back to HU-ish range used in base window [-300,400]
            f1_hu = f1c * (400 - (-300)) + (-300)
            f2_hu = f2c * (400 - (-300)) + (-300)

            low  = int(self.rng.randint(-400, -199))
            high = int(self.rng.randint(300, 501))
            f1_in = cap_to_255(f1_hu, low, high)
            f2_in = cap_to_255(f2_hu, low, high)

            g1 = float(self.rng.uniform(0.5, 2.0))
            g2 = float(self.rng.uniform(0.5, 2.0))
            f1_in = gamma_contrast_255(f1_in, g1)
            f2_in = gamma_contrast_255(f2_in, g2)

            # targets: clean [0,1]
            f1_t = f1c.astype(np.float32)
            f2_t = f2c.astype(np.float32)

            # torch [1,256,256]
            frame1_input = torch.from_numpy(f1_in / 255.0).unsqueeze(0)
            frame2_input = torch.from_numpy(f2_in / 255.0).unsqueeze(0)
            frame1_t = torch.from_numpy(f1_t).unsqueeze(0)
            frame2_t = torch.from_numpy(f2_t).unsqueeze(0)

            return frame1_input, frame2_input, frame1_t, frame2_t

        except Exception as e:
            print(f"[PancreasSlicePairFromCache] __getitem__ error: {e}")
            return None

class PancreasTestSingleSliceFromCache(torch.utils.data.Dataset):
    """
    cached_ds returns dict: {"image": [1,H,W,Z], "label_full":[1,H,W,Z], "source":...}
    Output:
      img_vol:  [Z,H,W] float32
      lab_full: [Z,H,W] uint8 (0/1)
      lab_seed: [Z,H,W] uint8 (only one slice)
      seed_idx, z_start, z_end, case_id, source
    """
    def __init__(self, cached_ds: CacheDataset):
        self.cached_ds = cached_ds

    def __len__(self):
        return len(self.cached_ds)

    def __getitem__(self, idx: int):
        try:
            d = self.cached_ds[idx]
            img = d["image"].squeeze(0)      # [H,W,Z]
            lab = d["label_full"].squeeze(0) # [H,W,Z]

            img_np = img.cpu().numpy().astype(np.float32)
            lab_np = (lab.cpu().numpy() > 0).astype(np.uint8)

            areas = lab_np.sum(axis=(0, 1))  # [Z]
            if areas.max() == 0:
                return None

            seed_idx = int(np.argmax(areas))
            pos = np.nonzero(areas > 0)[0]
            z_start, z_end = int(pos[0]), int(pos[-1])

            lab_seed = np.zeros_like(lab_np, dtype=np.uint8)
            lab_seed[:, :, seed_idx] = lab_np[:, :, seed_idx]

            # to [Z,H,W]
            img_z = np.moveaxis(img_np, -1, 0)
            lab_full_z = np.moveaxis(lab_np, -1, 0)
            lab_seed_z = np.moveaxis(lab_seed, -1, 0)

            case_id = d.get("case_id", "unknown_case")

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

        except Exception as e:
            print(f"[PancreasTestSingleSliceFromCache] __getitem__ error: {e}")
            return None

# ============================================================
# Builder: supports two settings
#   A) labeled cases split within-source; unlabeled cases train-only
#   B) train on specified sources; test on specified sources (labeled only)
# ============================================================
def build_multi_source_loaders(
    source_roots: List[Tuple[str, str]],  # [(source_name, root_dir), ...]
    setting: str = "A",                  # "A" or "B"
    train_sources: Optional[List[str]] = None,
    test_sources: Optional[List[str]] = None,
    seed: int = 10,
    batch_size: int = 5,
    num_workers: int = 4,
    train_cache_rate: float = 1.0,
    test_cache_rate: float = 1.0,
    labeled_train_ratio: float = 0.9,    # for labeled split within-source
):
    # 1) collect mixed labeled/unlabeled from all sources
    all_items: List[Dict[str, Any]] = []
    for src_name, root_dir in source_roots:
        all_items.extend(collect_source_items_mixed(src_name, root_dir))

    labeled_items   = [it for it in all_items if it.get("labeled", False)]
    unlabeled_items = [it for it in all_items if not it.get("labeled", False)]

    # 2) choose split protocol
    setting = setting.upper()
    if setting == "A":
        # split only labeled items within each source for train/test
        labeled_train, labeled_test = split_within_source(
            labeled_items, train_ratio=labeled_train_ratio, seed=seed
        )
        train_pool = labeled_train + unlabeled_items
        test_pool  = labeled_test

        print(f"[Setting A] train_pool={len(train_pool)} (labeled_train={len(labeled_train)} + unlabeled={len(unlabeled_items)})")
        print(f"[Setting A] test_pool={len(test_pool)} (labeled_test only)")

    elif setting == "B":
        if train_sources is None or test_sources is None:
            raise ValueError("Setting B requires train_sources and test_sources lists.")

        train_sources_set = set(train_sources)
        test_sources_set  = set(test_sources)

        train_pool = [it for it in all_items if it["source"] in train_sources_set]

        # IMPORTANT: test must be labeled
        test_pool  = [it for it in labeled_items if it["source"] in test_sources_set]

        if len(train_pool) == 0:
            raise RuntimeError("Setting B: train_pool is empty. Check train_sources.")
        if len(test_pool) == 0:
            raise RuntimeError("Setting B: test_pool is empty. Are test_sources unlabeled or labels missing?")

        print(f"[Setting B] train_sources={sorted(list(train_sources_set))} train_pool={len(train_pool)}")
        print(f"[Setting B] test_sources={sorted(list(test_sources_set))} test_pool={len(test_pool)}")

    else:
        raise ValueError("setting must be 'A' or 'B'")

    # 3) HARD GUARANTEE: no unlabeled in test
    assert all(it.get("labeled", False) for it in test_pool), "BUG: Unlabeled sample found in test_pool!"
    assert all(("label_full" in it) and os.path.exists(it["label_full"]) for it in test_pool), \
        "BUG: test_pool has missing label_full paths!"

    # 4) split train_pool into train/val (self-supervised, labels not required)
    train_items, val_items = make_split(train_pool, train_ratio=0.9, seed=seed + 1)
    print(f"[train/val] train={len(train_items)} val={len(val_items)}")

    # 5) build CacheDatasets
    # train/val cache: images only
    train_img = [{"image": it["image"], "source": it["source"]} for it in train_items]
    val_img   = [{"image": it["image"], "source": it["source"]} for it in val_items]
    # test cache: image + label_full
    test_lab  = [{"image": it["image"], "label_full": it["label_full"], "source": it["source"], "case_id": _stem_niigz(it["image"])} for it in test_pool]

    train_cached = CacheDataset(train_img, base_img_tf, cache_rate=train_cache_rate, num_workers=num_workers)
    val_cached   = CacheDataset(val_img,   base_img_tf, cache_rate=train_cache_rate, num_workers=num_workers)
    test_cached  = CacheDataset(test_lab,  base_img_lab_tf, cache_rate=test_cache_rate, num_workers=num_workers)

    # 6) wrap into task datasets
    train_ds = PancreasSlicePairFromCache(train_cached, seed=seed)
    val_ds   = PancreasSlicePairFromCache(val_cached, seed=seed + 1)
    test_ds  = PancreasTestSingleSliceFromCache(test_cached)

    # 7) DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=my_collate_drop_none,
        drop_last=False, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=my_collate_drop_none,
        drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=my_collate_drop_none,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

# ============================================================
# Debug utilities (batch content sanity)
# ============================================================
def debug_train_batch(train_loader):
    batch = next(iter(train_loader))
    f1_in, f2_in, f1_t, f2_t = batch
    print("TRAIN batch:")
    print(" f1_in:", tuple(f1_in.shape), float(f1_in.min()), float(f1_in.max()))
    print(" f2_in:", tuple(f2_in.shape), float(f2_in.min()), float(f2_in.max()))
    print(" f1_t :", tuple(f1_t.shape),  float(f1_t.min()),  float(f1_t.max()))
    print(" f2_t :", tuple(f2_t.shape),  float(f2_t.min()),  float(f2_t.max()))

def debug_test_batch(test_loader):
    batch = next(iter(test_loader))
    img_vol, lab_full, lab_seed, seed_idx, z_start, z_end, case_id, source = batch
    print("TEST sample:")
    print(" case:", case_id[0], "source:", source[0])
    print(" img_vol:", tuple(img_vol.shape), "range:", float(img_vol.min()), float(img_vol.max()))
    print(" lab_full sum:", int(lab_full.sum()), "lab_seed sum:", int(lab_seed.sum()))
    print(" seed_idx:", int(seed_idx), "z_range:", int(z_start), int(z_end))
