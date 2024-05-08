#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity,11GB
#SBATCH --mem=11GB
#SBATCH -t 3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-49
#SBATCH --output /om2/user/zaho/flappydream/reports/slurm-%A_%a.out # STDOUT
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

C=$(((SLURM_ARRAY_TASK_ID/5)/5))
A=$((SLURM_ARRAY_TASK_ID%5))
B=$(((SLURM_ARRAY_TASK_ID/5)%5))
arrC=("9 10 11" "3 4 5 9 10 11")
arrA=("0" "1" "10" "20" "40")
arrB=(0 1 2 3 4)
A=${arrA[$A]}
B=${arrB[$B]}

echo $A $B
nvidia-smi
python train_rnn.py -t 40 -lr 0.001 -ecuda -lsv $A -sv 9 10 11 -r $B