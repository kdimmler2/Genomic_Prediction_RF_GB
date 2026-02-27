import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    average_precision_score,
    matthews_corrcoef,
    recall_score,
)

rule all:
    input:
        'results/hyperparameter_tuning/all_models.csv',

rule random_RF_grid_search:
    input:
        modeling_input = config['input_data'],
    output:
        all_models       = 'results/hyperparameter_tuning/all_models.csv',
        splits_npz       = 'results/hyperparameter_tuning/splits.npz'
    resources:
        time=360,
        mem_mb=8000,
        nodes=1,
        cpus_per_task=4,
        ntasks=1
    threads: 4
    run:
        # Load final modeling matrix (rows=samples). Expect columns:
        #   - y (binary label)
        #   - optional covariates (e.g., sex)
        #   - genotype features
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index('sample', inplace=True)

        # Separate features (X) and label (y). Keep y out of all feature transforms
        # to avoid leakage.
        X = data.drop('y', axis=1)
        y = data['y']

        # Fixed seed for reproducibility across runs / environments
        SEED = config['seed']

        # Create fixed Train/Test/Validation splits:
        #   - Train (60%): used for RandomizedSearchCV fitting/selection
        #   - Test  (20%): held out for unbiased final performance reporting
        #   - Evaluation   (20%): reserved for downstream calibration/thresholding steps
        #
        # Stratification preserves class balance in each split.
        X_train_full, X_eval, y_train_full, y_eval = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=SEED, stratify=y_train_full
        )

        # Quick sanity check: class counts per split (helps catch accidental leakage / imbalance)
        print('Training set size:', len(y_train), 'Positive cases:', np.sum(y_train))
        print('Test set size:', len(y_test), 'Positive cases:', np.sum(y_test))
        print('Validation set size:', len(y_eval), 'Positive cases:', np.sum(y_eval))

        # Random Forest hyperparameter search space (intentionally broad).
        # NOTE: These ranges should reflect expected feature dimensionality and sample size.
        param_dist = {
            'n_estimators': [500, 1000],
            'max_depth': [10, 12, 16, 20, None],
            'min_samples_split': [5, 10, 20, 50],
            'min_samples_leaf': [1, 5, 10, 20],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'criterion': ['gini', 'entropy'],
        }

        # Number of sampled hyperparameter configurations
        n_iter = 5

        # Multi-metric scoring:
        # - Use AP (average precision) as the selection/refit metric because the label is imbalanced.
        # - Collect other metrics for a fuller error profile.
        # - Specificity is computed as recall with pos_label=0 (TNR).
        #
        # IMPORTANT: metrics using probabilities require needs_proba=True.
        scoring = {
            'ap':           make_scorer(average_precision_score, needs_proba=True),
            'roc_auc':      'roc_auc',
            'f1':           'f1',
            'precision':    'precision',
            'recall':       'recall',                       # sensitivity (TPR)
            'balanced_acc': 'balanced_accuracy',
            'mcc':          make_scorer(matthews_corrcoef),
            'specificity':  make_scorer(recall_score, pos_label=0),  # TNR
        }

        # Base model + CV scheme. Use stratified folds for class balance.
        # n_jobs=-1 uses all available cores; consider capping on local or underpowered computers.
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        # Randomized search on TRAIN only (test/eval are untouched).
        # refit='ap' ensures best_estimator_ is selected by AP.
        random_search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            refit='ap',
            cv=cv,
            n_jobs=-1,
            random_state=SEED,
            verbose=2,
            return_train_score=True
        )

        # Fit hyperparameter search on the TRAIN split only.
        # Test and evaluation splits are reserved for downstream evaluation steps.
        random_search.fit(X_train, y_train)

        # Flatten cv_results_ into a long table so we can inspect tradeoffs and stability
        # across metrics (mean/std across folds) for each sampled configuration.
        all_results = random_search.cv_results_
        all_model_performance = []

        for i, params in enumerate(all_results['params']):
            row = {**params, 'iteration': i + 1}

            # CV test metrics (means / stds)
            if 'mean_test_ap' in all_results:               row['cv_ap'] = all_results['mean_test_ap'][i]
            if 'std_test_ap' in all_results:                row['cv_ap_std'] = all_results['std_test_ap'][i]

            if 'mean_test_roc_auc' in all_results:          row['cv_auc'] = all_results['mean_test_roc_auc'][i]
            if 'std_test_roc_auc' in all_results:           row['cv_auc_std'] = all_results['std_test_roc_auc'][i]

            if 'mean_test_f1' in all_results:               row['cv_f1'] = all_results['mean_test_f1'][i]
            if 'std_test_f1' in all_results:                row['cv_f1_std'] = all_results['std_test_f1'][i]

            if 'mean_test_precision' in all_results:        row['cv_precision'] = all_results['mean_test_precision'][i]
            if 'std_test_precision' in all_results:         row['cv_precision_std'] = all_results['std_test_precision'][i]

            if 'mean_test_recall' in all_results:           row['cv_recall'] = all_results['mean_test_recall'][i]
            if 'std_test_recall' in all_results:            row['cv_recall_std'] = all_results['std_test_recall'][i]

            if 'mean_test_specificity' in all_results:      row['cv_specificity'] = all_results['mean_test_specificity'][i]
            if 'std_test_specificity' in all_results:       row['cv_specificity_std'] = all_results['std_test_specificity'][i]

            if 'mean_test_balanced_acc' in all_results:     row['cv_bal_acc'] = all_results['mean_test_balanced_acc'][i]
            if 'std_test_balanced_acc' in all_results:      row['cv_bal_acc_std'] = all_results['std_test_balanced_acc'][i]

            if 'mean_test_mcc' in all_results:              row['cv_mcc'] = all_results['mean_test_mcc'][i]
            if 'std_test_mcc' in all_results:               row['cv_mcc_std'] = all_results['std_test_mcc'][i]

            # Optional: train-side metrics can help diagnose overfitting
            if 'mean_train_ap' in all_results:              row['train_ap'] = all_results['mean_train_ap'][i]
            if 'mean_train_roc_auc' in all_results:         row['train_auc'] = all_results['mean_train_roc_auc'][i]
            if 'mean_train_f1' in all_results:              row['train_f1'] = all_results['mean_train_f1'][i]
            if 'mean_train_precision' in all_results:       row['train_precision'] = all_results['mean_train_precision'][i]
            if 'mean_train_recall' in all_results:          row['train_recall'] = all_results['mean_train_recall'][i]
            if 'mean_train_specificity' in all_results:     row['train_specificity'] = all_results['mean_train_specificity'][i]
            if 'mean_train_balanced_acc' in all_results:    row['train_bal_acc'] = all_results['mean_train_balanced_acc'][i]
            if 'mean_train_mcc' in all_results:             row['train_mcc'] = all_results['mean_train_mcc'][i]

            # Rank by AP (useful for sorting / filtering later)
            if 'rank_test_ap' in all_results:               row['rank_ap'] = all_results['rank_test_ap'][i]

            all_model_performance.append(row)

        # Save full search results so we can reproduce/inspect the hyperparameter landscape
        all_performance_df = pd.DataFrame(all_model_performance)
        all_performance_df.to_csv(output.all_models, index=False)

        np.savez(
            output.splits_npz,
            train_idx=X_train.index.values,
            eval_idx=X_eval.index.values,
            test_idx=X_test.index.values
        )

