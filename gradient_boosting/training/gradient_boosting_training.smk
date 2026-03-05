########################################
# GB_driver.smk
########################################
import os

SUBCORES = int(os.environ.get("SUBCORES", "4"))

DONE_DIR = ".done_gb"
os.makedirs(DONE_DIR, exist_ok=True)

rule all:
    input:
        f"{DONE_DIR}/GB_final_training.done"


rule GB_hyperparameter_tuning:
    output:
        f"{DONE_DIR}/GB_hyperparameter_tuning.done"
    shell:
        r"""
        snakemake -s pipelines/GB_hyperparameter_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule GB_threshold_tuning:
    input:
        f"{DONE_DIR}/GB_hyperparameter_tuning.done"
    output:
        f"{DONE_DIR}/GB_threshold_tuning.done"
    shell:
        r"""
        snakemake -s pipelines/GB_threshold_tuning.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule GB_test_metrics:
    input:
        f"{DONE_DIR}/GB_threshold_tuning.done"
    output:
        f"{DONE_DIR}/GB_test_metrics.done"
    shell:
        r"""
        snakemake -s pipelines/GB_test_metrics.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """


rule GB_final_training:
    input:
        f"{DONE_DIR}/GB_test_metrics.done"
    output:
        f"{DONE_DIR}/GB_final_training.done"
    shell:
        r"""
        snakemake -s pipelines/GB_final_training.smk \
                  --cores {SUBCORES} --rerun-incomplete --nolock \
                  --configfile config.yaml --rerun-triggers mtime
        touch {output}
        """
