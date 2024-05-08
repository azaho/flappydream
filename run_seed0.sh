#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity,11GB
#SBATCH --mem=11GB
#SBATCH -t 3:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-19
#SBATCH --output /om2/user/zaho/flappydream/reports/slurm-%A_%a.out # STDOUT
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

A=$((SLURM_ARRAY_TASK_ID))

echo $A $B
nvidia-smi
python train_rnn.py -t 40 -lr 0.001 -ecuda -lsv 0.0 -sv 9 10 11 -r $A