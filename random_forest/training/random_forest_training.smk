########################################
# RF_driver.smk
########################################

import os

# Use 4 cores if number of cores is not defined in the environment
SUBCORES = int(os.environ.get("SUBCORES", "4"))

DONE_DIR = ".done_rf"
os.makedirs(DONE_DIR, exist_ok=True)

rule all:
    input:
        DONE_DIR + "/RF_final_training.done"


rule RF_hyperparameter_tuning:
    output:
        DONE_DIR + "/RF_hyperparameter_tuning.done"
    log:
        "logs/RF_hyperparameter_tuning.log"
    shell:
        r"""
        mkdir -p logs
        snakemake -s pipelines/RF_hyperparameter_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime \
                  > {log} 2>&1
        touch {output}
        """


rule RF_threshold_tuning:
    input:
        DONE_DIR + "/RF_hyperparameter_tuning.done"
    output:
        DONE_DIR + "/RF_threshold_tuning.done"
    log:
        "logs/RF_threshold_tuning.log"
    shell:
        r"""
        mkdir -p logs
        snakemake -s pipelines/RF_threshold_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime \
                  > {log} 2>&1
        touch {output}
        """


rule RF_test_metrics:
    input:
        DONE_DIR + "/RF_threshold_tuning.done"
    output:
        DONE_DIR + "/RF_test_metrics.done"
    log:
        "logs/RF_test_metrics.log"
    shell:
        r"""
        mkdir -p logs
        snakemake -s pipelines/RF_test_metrics.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime \
                  > {log} 2>&1
        touch {output}
        """


rule RF_final_training:
    input:
        DONE_DIR + "/RF_test_metrics.done"
    output:
        DONE_DIR + "/RF_final_training.done"
    log:
        "logs/RF_final_training.log"
    shell:
        r"""
        mkdir -p logs
        snakemake -s pipelines/RF_final_training.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime \
                  > {log} 2>&1
        touch {output}
        """
