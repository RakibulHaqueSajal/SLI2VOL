#!/bin/bash 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH --open-mode=append 
#SBATCH --time=0-23:50:00 
#SBATCH --mem=100G 
#SBATCH --partition=medvic
#SBATCH --output=log_%J.txt

##/uufs/sci.utah.edu/projects/medvic-lab/Rakib/miniconda3/etc/profile.d/conda.sh 

#pancreas only


#SettingA
# python train.py \
#   --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
#   --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
#   --setting A \
#   --use_comet True \
#   --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Pancrease_A_RUN_Final\
#   --run_name Pancreas_Only_A_Run_Final\
#   --batch_size 16 --num_workers 4 \
#   --max_epochs 300 --lr 1e-4 \
#   --early_stop --patience 50 --min_delta 1e-4 \



#SettingB
# python train.py \
#   --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
#   --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
#   --setting B \
#   --train_sources SourceA \
#   --test_sources SourceB \
#   --use_comet True \
#   --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Pancreas_B_Run_Final\
#   --run_name Pancreas_B_Run_Final \
#   --batch_size 16 --num_workers 4 \
#   --max_epochs 250 --lr 1e-4 \
#   --early_stop --patience 50 --min_delta 1e-4 \
 

##Liver only





# python train.py \
#   --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
#   --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
#   --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train\
#   --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test\
#   --setting A \
#   --use_comet True \
#   --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver/Liver_A_Run_Final \
#   --run_name Liver_Only_A_Run_Final \
#   --batch_size 16 --num_workers 4 \
#   --max_epochs 250 --lr 1e-4 \
#   --early_stop --patience 50 --min_delta 1e-4 \


#SettingB

# python train.py \
#   --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
#   --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
#   --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train\
#   --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test\
#   --setting B \
#   --use_comet True \
#   --train_sources SourceA,SourceD \
#   --test_sources SourceB,SourceC \
#   --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver/Liver_B_Run_Final \
#   --run_name Liver_Only_B_Run_Final \
#   --batch_size 16 --num_workers 4 \
#   --max_epochs 250 --lr 1e-4 \
#   --early_stop --patience 50 --min_delta 1e-4 \


#lIVER+PANCREAS


# python train.py \
#   --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
#   --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
#   --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train \
#   --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test \
#   --sourceE_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
#   --sourceF_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
#   --setting B \
#   --use_comet True \
#   --train_sources SourceA,SourceD,SourceE \
#   --test_sources SourceB,SourceD,SourceF \
#   --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver_Pancreas/Liver_Pancreas_B_Run_Final \
#   --run_name Liver_Pancreas_B_Run_Final \
#   --batch_size 16 \
#   --num_workers 4 \
#   --max_epochs 350 \
#   --lr 1e-4 \
#   --early_stop \
#   --patience 50 \
#   --min_delta 1e-4


  # python train.py \
  # --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  # --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  # --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train \
  # --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test \
  # --sourceE_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  # --sourceF_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  # --setting A \
  # --use_comet True \
  # --save_dir /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver_Pancreas/Liver_Pancreas_A_Run_Final \
  # --run_name Liver_Pancreas_A_Run_Final \
  # --batch_size 16 \
  # --num_workers 4 \
  # --max_epochs 350 \
  # --lr 1e-4 \
  # --early_stop \
  # --patience 50 \
  # --min_delta 1e-4



##Evaluation Only Script on Pancreas Only Model

#Pancreas Setting A

python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  --setting A \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Pancrease_A_RUN/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Pancreas-A\
  --boxplot_on test \
  --batch_size 16 --num_workers 4 \

#Pancreas Setting B

python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  --setting B \
  --train_sources SourceA \
  --test_sources SourceB \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/pancreas/Pancreas_B_Run_Final/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Pancreas-B\
  --boxplot_on test \
  --batch_size 16 --num_workers 4 \

# #Liver Setting A 

 python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train\
  --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test\
  --setting A \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver/Liver_A_Run_Final/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Liver-A\
  --boxplot_on test \
  --batch_size 16 --num_workers 4 \

# #Liver Setting B

python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train\
  --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test\
  --setting B \
  --train_sources SourceA,SourceD \
  --test_sources SourceB,SourceC \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver/Liver_B_Run_Final/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Liver-B\
  --boxplot_on test \
  --batch_size 16 --num_workers 4 \

# #Liver+Pancreas Setting A

  python test.py \
  --sourceA_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/Chaos \
  --sourceB_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/SLiver07 \
  --sourceC_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Decatholon_Train \
  --sourceD_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Liver/DecathLiver/Deactholon_Test \
  --sourceE_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/CT-Pancreas \
  --sourceF_root /uufs/sci.utah.edu/scratch/Rakib_data/3D-Segmentation/Pancreas/DecathPancreas \
  --setting A \
  --ckpt /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/checkpoint/Liver_Pancreas/Liver_Pancreas_A_Run_Final/best_model.pth \
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Liver_Pancreas-A\
  --boxplot_on test \
  --batch_size 16 \
  --num_workers 4 \


# #Liver+Pancreas Setting B
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
  --out_root /uufs/sci.utah.edu/projects/medvic-lab/Rakib/Sli2Vol/Results/Liver_Pancreas-B\
  --boxplot_on test \
  --batch_size 16 \
  --num_workers 4 \
 