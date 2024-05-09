#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity,11GB
#SBATCH --mem=11GB
#SBATCH -t 3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-239
#SBATCH --output /om2/user/zaho/flappydream/reports/slurm-%A_%a.out # STDOUT
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

A=$((SLURM_ARRAY_TASK_ID%6))
B=$(((SLURM_ARRAY_TASK_ID/6)%20))
C=$(((SLURM_ARRAY_TASK_ID/6)/20))
arrA=("0" "4" "8" "16" "32" "64")
arrB=(20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39)
arrC=("9 10 11" "3 4 5 9 10 11")
A=${arrA[$A]}
B=${arrB[$B]}
C=${arrC[$C]}

echo $A $B $C
nvidia-smi
python train_rnn.py -t 40 -lr 0.001 -ecuda -lsv $A -sv $C -r $B