snakemake -s random_forest_training.smk \
    --cores 1 \
    --printshellcmds \
    --rerun-incomplete \
    --configfile config.yaml \
    --keep-going \
    --rerun-triggers mtime \
    > logs/random_forest_training.out 2> logs/random_forest_training.err
