#!/bin/bash -l
#SBATCH -t 16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCH --mem=6gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dimml002@umn.edu
#SBATCH --job-name RF.slurm
#SBATCH -o ./jobs/%j.sklearn_RF.out
#SBATCH -e ./jobs/%j.sklearn_RF.err
#SBATCH -p agsmall,aglarge

set -e

cd $SLURM_SUBMIT_DIR

micromamba activate sklearn

snakemake -s sklearn_RF.smk \
    --profile slurm.RF \
    --keep-going \
    --rerun-incomplete \
    --rerun-triggers mtime \
    --configfile configfile.yaml

#snakemake -s BackElim.smk --configfile configfile.yaml --unlock

# Define the number of times to run
num_runs=45

# Loop to run  multiple times
for ((i=1; i<=$num_runs; i++))
do
	echo "Running iteration $i"
	snakemake -s BackElim.smk \
	    --profile slurm.RF \
	    --keep-going \
	    --rerun-incomplete \
	    --rerun-triggers mtime \
	    --configfile configfile.yaml \
	    --restart-times 3
	
	python3 rename.py

	wait
done

python3 rename.py
