#!/bin/bash
#SBATCH -n 1                # node count
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH -t 02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-239
#SBATCH --output /om2/user/zaho/ccn_story3/reports/slurm-%A_%a.out # STDOUT
#SBATCH -p yanglab
export PATH="/om2/user/zaho/anaconda3/bin:$PATH"

A=$((SLURM_ARRAY_TASK_ID/30))
B=$((SLURM_ARRAY_TASK_ID%30))
arrA=("hdgating.py" "hdinversion.py" "hdratio.py" "hdreshuffle_fixed.py" "hdgating_and_inversion.py" "hdgating_and_reshuffle_fixed.py" "backprop.py" "backprop_nodistractor.py")
arrB=(0 1 2 3 4)
A=${arrA[$A]}

echo $A $B
python train_${A} --random $B