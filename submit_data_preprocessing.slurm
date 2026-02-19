#!/bin/bash -l
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCH --mem=2gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dimml002@umn.edu
#SBATCH --job-name RF.part1
#SBATCH -o ./jobs/%j.RF_part1.out
#SBATCH -e ./jobs/%j.RF_part1.err
#SBATCH -p msismall,agsmall

set -e

cd $SLURM_SUBMIT_DIR

source /users/7/dimml002/miniforge3/etc/profile.d/conda.sh
conda activate sklearn

snakemake -s data_preprocessing.smk \
    --profile slurm.RF \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --configfile configfile.yaml
