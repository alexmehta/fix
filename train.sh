#!/bin/bash -l

#$ -l h_rt=18:00:00

#$ -m ea

#$ -N DEQ-Prox-Denoising-Full
# note do -l buyin
#$ -j y
#$ -pe omp 4
#$ -l gpus=4
#$ -l gpu_c=7.0
module load python3/3.10.12
module load torch/2.1
# pip3 install -r requirements.txt --no-cache-dir
python3 -m deep_equilibrium_inverse.scripts.fixedpoint.deblur_proxgrad_fixedeta_pre --batch_size 256 --savepath models/blur.ckpt  --loadpath models/blur.ckpt --datapath ../img_align_celeba