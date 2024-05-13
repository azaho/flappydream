#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --mem=11GB
#SBATCH -t 3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-239
#SBATCH --output /om2/user/zaho/flappydream/reports/slurm-%A_%a.out # STDOUT
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

A=$((SLURM_ARRAY_TASK_ID%2))
B=$(((SLURM_ARRAY_TASK_ID/2)%40))
C=$(((SLURM_ARRAY_TASK_ID/2)/40))
arrA=("0" "64")
arrC=(64 128 256)
A=${arrA[$A]}
C=${arrC[$C]}

echo $A $B $C
nvidia-smi
python train_rnn.py -t 40 -lr 0.001 -ecuda -lsv $A -sv 9 10 11 -nh $C -r n$B