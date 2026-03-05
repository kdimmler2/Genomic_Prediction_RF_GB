#!/bin/bash -l
#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dimml002@umn.edu
#SBATCH --job-name validate.slurm
#SBATCH -o ./jobs/%j.validate.out
#SBATCH -e ./jobs/%j.validate.err
#SBATCH -p agsmall,aglarge

set -e

cd $SLURM_SUBMIT_DIR

micromamba activate sklearn

snakemake -s validate.smk \
    --profile slurm.validate \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --configfile configfile.yaml
