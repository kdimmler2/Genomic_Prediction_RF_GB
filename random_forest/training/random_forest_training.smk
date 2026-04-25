########################################
# RF_driver.smk
########################################
import os

# use 4 cores if number of cores is not defined in the rule
SUBCORES = int(os.environ.get("SUBCORES", "4"))

DONE_DIR = ".done_rf"
os.makedirs(DONE_DIR, exist_ok=True)

rule all:
    input:
        f"{DONE_DIR}/RF_final_training.done"


rule RF_hyperparameter_tuning:
    output:
        f"{DONE_DIR}/RF_hyperparameter_tuning.done"
    shell:
        r"""
        snakemake -s pipelines/RF_hyperparameter_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule RF_threshold_tuning:
    input:
        f"{DONE_DIR}/RF_hyperparameter_tuning.done"
    output:
        f"{DONE_DIR}/RF_threshold_tuning.done"
    shell:
        r"""
        snakemake -s pipelines/RF_threshold_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule RF_test_metrics:
    input:
        f"{DONE_DIR}/RF_threshold_tuning.done"
    output:
        f"{DONE_DIR}/RF_test_metrics.done"
    shell:
        r"""
        snakemake -s pipelines/RF_test_metrics.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule RF_final_training:
    input:
        f"{DONE_DIR}/RF_test_metrics.done"
    output:
        f"{DONE_DIR}/RF_final_training.done"
    shell:
        r"""
        snakemake -s pipelines/RF_final_training.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """
