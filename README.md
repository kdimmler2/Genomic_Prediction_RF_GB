# Creation of Genomic Prediction Models using Ensemble Machine Learning and Feature Reduction with Elastic Net

This workflow builds genomic prediction models from variant data in a VCF file to classify a binary phenotype. Random Forest and Gradient Boosting models are trained using genome-wide variant features, and feature importance scores are used for feature reduction. Elastic Net regression provides an additional layer of feature selection and produces a more interpretable predictive model.

---

## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/kdimmler2/Genomic_Prediction_RF_GB.git
cd Genomic_Prediction_RF_GB

conda env create -f conda_env.yaml
conda activate Genomic_Prediction_RF_GB
```

---

## Phenotype File Format

A phenotype file with the following format is required:

```bash
0 HG00096 0
```

The columns represent:

1. **Binary phenotype** (0 or 1)  
2. **Sample ID** as it appears in the VCF  
3. **Covariates**

Additional covariates can be included by adding extra columns.

An example file is provided in:

```
demo_data/training_phenos.txt
```

---

## Data Preprocessing

Training and validation datasets are prepared using the following scripts:

```bash
bash training_data.sh
bash validation_data.sh
```

Processed data will be written to:

```
data_preprocessing/training_data/
data_preprocessing/validation_data/
```

---

## Random Forest and Gradient Boosting Training

Separate workflows are provided for Random Forest and Gradient Boosting models.  
The pipeline structure is the same for both algorithms.

The workflow stages are:

1. **Hyperparameter tuning**
2. **Threshold tuning**
3. **Test metrics**
4. **Final training**

It is recommended to run each stage sequentially. Example Snakemake commands are provided in:

```
random_forest/training/snakemake_commands.txt
```

### Hyperparameter Selection

After hyperparameter tuning, results can be analyzed using the R script:

```
RF_RandomGS_Run1.qmd
```

The selected hyperparameters can then be entered into:

```
config.yaml
```

### Threshold Selection

After threshold tuning, a classification threshold can be selected based on model performance and added to `config.yaml`.

### Test Metrics

This stage evaluates model performance on a subset of the training data.

### Final Training

After confirming acceptable performance, the final model is trained using the full training dataset.

---

## Validation

The trained model can then be applied to the validation dataset to generate predictions.

All variants present in the training data must also be present in the validation data.

---

## Feature Reduction

Feature importance scores generated during final training can be used to select a reduced feature set.

One common approach is to rank features by importance and include features until approximately **80% of the cumulative importance score** is reached.

---

## Elastic Net

Elastic Net models are implemented in R using the same training and validation data splits.

This provides an additional level of feature selection and produces a more interpretable predictive model.
