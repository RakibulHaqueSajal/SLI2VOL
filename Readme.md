### Sli2Vol — Liver & Pancreas Segmentation (Setting B)

This repository documents the exact training and testing configuration used for multi-organ (Liver + Pancreas) 3D segmentation using Sli2Vol under a source-based split (Setting B).

### Overview

This experiment evaluates propagation-based 3D segmentation under a multi-dataset configuration. The structural assumption tested in this setup is local compositional continuity, where segmentation is propagated slice-by-slice from a single annotated slice.

The model is trained jointly on liver and pancreas datasets using a predefined source split and evaluated on disjoint sources to assess robustness.

### Dataset Structure

The following datasets are used:

Liver Datasets

CHAOS

SLiver07

Medical Segmentation Decathlon (Liver)

Pancreas Datasets

CT-Pancreas

Medical Segmentation Decathlon (Pancreas)

Each dataset is assigned a source label internally (SourceA–SourceF) for flexible train/test splitting.

### Source-Based Split (Setting B)

Training Sources:

CHAOS (Liver)

Decathlon Liver (Test directory subset)

CT-Pancreas

Testing Sources:

SLiver07 (Liver)

Decathlon Liver (Test directory subset)

Decathlon Pancreas

This configuration introduces dataset-level variability in scanner characteristics, acquisition protocols, and annotation styles.

### Training
```
python train.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train \
  --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test \
  --sourceE_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  --sourceF_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  --setting B \
  --use_comet True \
  --train_sources SourceA,SourceD,SourceE \
  --test_sources SourceB,SourceD,SourceF \
  --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver_Pancreas/Liver_Pancreas_B_Run_Final \
  --run_name Liver_Pancreas_B_Run_Final \
  --batch_size 16 \
  --num_workers 4 \
  --max_epochs 350 \
  --lr 1e-4 \
  --early_stop \
  --patience 50 \
  --min_delta 1e-4
```

### Output

Best model checkpoint saved to:
```
.../checkpoint/Liver_Pancreas/Liver_Pancreas_B_Run_Final/best_model.pth
```

### Testing / Evaluation
```
python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train \
  --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test \
  --sourceE_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  --sourceF_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  --setting B \
  --train_sources SourceA,SourceD,SourceE \
  --test_sources SourceB,SourceD,SourceF \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver_Pancreas/Liver_Pancreas_B_Run_Final/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Liver_Pancreas-B \
  --boxplot_on test \
  --batch_size 16 \
  --num_workers 4

```
### Output Includes:

Dice scores

Hausdorff distances

Slice-wise performance curves


### Evaluation Protocol

Seed slice selected as the slice with the largest organ cross-sectional area

Metrics:

Dice coefficient (volumetric overlap)

Hausdorff distance (boundary deviation)

Slice-wise analysis performed to evaluate distance-dependent degradation

### Reproducibility Notes

To reproduce results:

Ensure dataset paths match those specified.

Use identical source split (--setting B).

Use the saved checkpoint for testing.

Do not modify hyperparameters or train/test source configuration.

Any deviation from these parameters will produce non-comparable results.

The conda environment is also provided here