# Genomic Prediction Models with Ensemble Machine Learning

This Snakemake workflow trains Random Forest and Gradient Boosting models from genomic variants in a VCF file to predict a binary phenotype. Feature importance scores are generated through permutation testing and can be used for feature reduction. Elastic Net regression (implimented through R) provides additional feature selection and a more interpretable predictive model.

First clone the repo and create the conda environment

```bash
git clone https://github.com/kdimmler2/Genomic_Prediction_RF_GB.git
cd Genomic_Prediction_RF_GB

conda env create -f conda.env.yaml
conda activate sklearn
```

