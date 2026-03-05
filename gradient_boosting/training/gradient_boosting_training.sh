snakemake -s random_forest_training.smk \
    --cores 1 \
    --printshellcmds \
    --rerun-incomplete \
    --configfile config.yaml \
    --keep-going \
    --rerun-triggers mtime
