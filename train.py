# -*- coding: utf-8 -*-
import os
import json
import glob
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Correspondence_Flow_Net
from model import *  # weight_init etc.
from dataset import my_collate, edge_profile
from data_creator.pancreas_dataset import build_multi_source_loaders

try:
    from comet_ml import Experiment
except Exception:
    Experiment = None


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=10):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# Early stopping
# ============================================================
class EarlyStopping:
    """
    Early stop if validation loss does not improve by min_delta for 'patience' validations.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_count = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns True if should stop.
        """
        if self.best is None:
            self.best = val_loss
            self.bad_count = 0
            return False

        if val_loss < (self.best - self.min_delta):
            self.best = val_loss
            self.bad_count = 0
            return False

        self.bad_count += 1
        return self.bad_count >= self.patience


# ============================================================
# Validation loop
# ============================================================
@torch.no_grad()
def run_validation(model, val_loader, device="cuda"):
    model.eval()
    total = 0.0
    n = 0

    for frame1_input, frame2_input, frame1, frame2 in val_loader:
        frame1_input = frame1_input.float().to(device, non_blocking=True)
        frame2_input = frame2_input.float().to(device, non_blocking=True)
        frame1 = frame1.float().to(device, non_blocking=True)
        frame2 = frame2.float().to(device, non_blocking=True)

        [frame1_input, frame2_input] = edge_profile([frame1_input, frame2_input], False, 3, 1)
        _, _, h, w = frame1_input.size()

        outputs = model(frame1_input, frame2_input, frame1)
        outputs = F.interpolate(outputs, (h, w), mode="bilinear", align_corners=False)

        loss = F.smooth_l1_loss(outputs, frame2, reduction="mean")
        total += float(loss.item())
        n += 1

    return total / max(1, n)


# ============================================================
# Train loop (one epoch)
# ============================================================
def run_train_epoch(model, train_loader, optimizer, device="cuda"):
    model.train()
    running = 0.0
    n = 0

    for frame1_input, frame2_input, frame1, frame2 in train_loader:
        frame1_input = frame1_input.float().to(device, non_blocking=True)
        frame2_input = frame2_input.float().to(device, non_blocking=True)
        frame1 = frame1.float().to(device, non_blocking=True)
        frame2 = frame2.float().to(device, non_blocking=True)

        [frame1_input, frame2_input] = edge_profile([frame1_input, frame2_input], False, 3, 1)
        
    
        _, _, h, w = frame1_input.size()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(frame1_input, frame2_input, frame1)
        outputs = F.interpolate(outputs, (h, w), mode="bilinear", align_corners=False)

        loss = F.smooth_l1_loss(outputs, frame2, reduction="mean")
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n += 1

    return running / max(1, n)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("Pancreas training (sli2vol)")

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

    # training hyperparams
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--max_epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--step_size", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.5)

    p.add_argument("--in_channels", type=int, default=48)
    p.add_argument("--R", type=int, default=6)

    # validation schedule
    p.add_argument("--val_every", type=int, default=1)

    # early stopping
    p.add_argument("--early_stop", action="store_true",
                   help="Enable early stopping")
    p.add_argument("--patience", type=int, default=50,
                   help="Patience for early stopping (in validation checks)")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum improvement required to reset patience")

    # checkpointing
    p.add_argument("--save_dir", type=str, required=True,
                   help="Directory to save checkpoints")
    p.add_argument("--run_name", type=str, default="Pancreas_Run")

    # comet
    p.add_argument("--use_comet", type=bool, default=False)
    p.add_argument("--comet_api_key", type=str, default=None)
    p.add_argument("--comet_project", type=str, default="sli2vol")
    p.add_argument("--comet_workspace", type=str, default="rakibulhaq56")

    # device
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    
    if args.sourceC_root is None:
        source_roots = [
            ("SourceA", args.sourceA_root),
            ("SourceB", args.sourceB_root),
        ]
    else:
        source_roots = [
            ("SourceA", args.sourceA_root),
            ("SourceB", args.sourceB_root),
            ("SourceC", args.sourceC_root),
            ("SourceD", args.sourceD_root),
            # ("SourceC", args.sourceC_root),
            # ("SourceD", args.sourceD_root),
        ]
    if args.sourceE_root is not None:
        source_roots.append( ("SourceE", args.sourceE_root) )
    if args.sourceF_root is not None:
        source_roots.append( ("SourceF", args.sourceF_root) )

    setting = args.setting.upper()

    train_sources = [s.strip() for s in args.train_sources.split(",") if s.strip()]
    test_sources = [s.strip() for s in args.test_sources.split(",") if s.strip()]

    # loaders
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

    # checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    current_model_path = os.path.join(args.save_dir, "current_model.pth")
    meta_path = os.path.join(args.save_dir, "run_config.json")

    with open(meta_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # comet
    
    if args.use_comet:
        comet_api_key = "Vr3vky03wyHWTXJTZ6phb4zEF"
        comet_project = "sli2vol"
        comet_workspace = "rakibulhaq56"
        experiment = Experiment(
            api_key=comet_api_key,
            project_name=comet_project,
            workspace=comet_workspace,
            log_code=True,
            disabled=False,
        )
        experiment.set_name(args.run_name)

    # model
    model = Correspondence_Flow_Net(in_channels=args.in_channels, is_training=True, R=args.R).to(device)
    model.apply(weight_init)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta) if args.early_stop else None

    best_val = float("inf")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device=device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        do_val = (epoch % args.val_every == 0) or (epoch == args.max_epochs)

        if do_val:
            val_loss = run_validation(model, val_loader, device=device)
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}")

            if experiment is not None:
                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("val_loss", val_loss, step=epoch)
                experiment.log_metric("lr", lr_now, step=epoch)

            # save best
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_model_path)

            # early stopping check
            if early is not None:
                should_stop = early.step(val_loss)
                if should_stop:
                    print(f"[EARLY STOP] epoch={epoch} best_val={early.best:.6f} patience={args.patience}")
                    break

        else:
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val=SKIP | lr={lr_now:.2e}")
            if experiment is not None:
                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("lr", lr_now, step=epoch)

        # always save current
        torch.save(model.state_dict(), current_model_path)

    print(f"Done. Best val: {best_val:.6f}")
    print(f"Saved best:    {best_model_path}")
    print(f"Saved current: {current_model_path}")

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
