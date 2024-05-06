#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --constraint=11GB
#SBATCH -t 10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-19
#SBATCH --output /om2/user/zaho/flappydream/reports/slurm-%A_%a.out # STDOUT
#SBATCH -p yanglab
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

A=$((SLURM_ARRAY_TASK_ID/5))
B=$((SLURM_ARRAY_TASK_ID%5))
arrA=("0" "5" "10" "20")
arrB=(0 1 2 3 4)
A=${arrA[$A]}

echo $A $B
python train_rnn.py -t 40 -lr 0.001 -ln -lsv $A -sv 9 10 11 -r $B