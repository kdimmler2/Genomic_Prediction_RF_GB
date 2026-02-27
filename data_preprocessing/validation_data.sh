#!/usr/bin/env bash

snakemake -s validation_data.smk \
    --cores 1 \
    --printshellcmds \
    --rerun-incomplete \
    --configfile config.yaml \
    --keep-going \
    --rerun-triggers mtime

