import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, average_precision_score

rule all:
    input:
        'results/final_training/model_refit_ALL.pkl',
        'results/final_training/importance_scores.csv'

rule rf_final_refit_all_internal:
    threads: 4
    input:
        modeling_input = config['input_data'],
        splits_npz     = 'results/hyperparameter_tuning/splits.npz'
    output:
        model_pkl      = 'results/final_training/model_refit_ALL.pkl',
        refit_ids_tsv  = 'results/final_training/refit_train_ids.tsv'
    run:
           # ---- Locked hyperparameters (no further tuning) ----
        # These were selected in the hyperparameter search step and stored in config.
        # This rule performs the final refit using all available training data.
        best_params = config['best_params']

        # ---- Load modeling matrix ----
        # Expect rows = samples and a binary label column y.
        data = pd.read_csv(input.modeling_input, sep='\t')
        data.set_index('sample', inplace=True)
        X = data.drop('y', axis=1)
        y = data['y']

        # ---- Load split indices and recombine FIT + CAL + INTERNAL-TEST ----
        # After threshold selection and internal evaluation, we refit using
        # all available labeled samples to produce the final deployable model.
        splits = np.load(input.splits_npz, allow_pickle=True)

        fit_idx          = splits['train_idx']   # original FIT
        calibration_idx  = splits['eval_idx']    # original CAL
        internaltest_idx = splits['test_idx']    # original INTERNAL-TEST

        # Combine all indices (unique to guard against accidental duplication)
        all_idx = pd.Index(
            np.concatenate([fit_idx, calibration_idx, internaltest_idx])
        ).unique()

        X_all, y_all = X.loc[all_idx], y.loc[all_idx]

        # ---- Final refit on full dataset ----
        # No validation splits remain — this produces the final trained model artifact.
        SEED = config['seed']
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_params)
        model.fit(X_all, y_all)

        # ---- Save model artifact + record training sample IDs ----
        # The pickle represents the final trained classifier.
        # Saving the sample IDs ensures full reproducibility of what data were used.
        with open(output.model_pkl, 'wb') as f:
            pickle.dump(model, f)

        pd.DataFrame({'sample': X_all.index}).to_csv(
            output.refit_ids_tsv, sep='\t', index=False
        )

rule rf_feature_importance:
    input:
        modeling_input = config['input_data'] 
    output:
        importance_scores = 'results/final_training/importance_scores.csv'
    resources:
        time=60,
        mem_mb=12000,
        nodes=1,
        cpus_per_task=1,
        ntasks=1
    threads: 1
    run:
        # ---- Load modeling matrix ----
        # Expect rows = samples and a binary label column y.
        data = pd.read_csv(input.modeling_input, sep='\t')
        data.set_index('sample', inplace=True)
        X = data.drop('y', axis=1)
        y = data['y']

        # ---- Load final hyperparameters ----
        # These were selected during the hyperparameter search stage and are now fixed.
        best_params = config['best_params']
        SEED = config['seed']

        # ---- Cross-validated permutation importance (AP scoring) ----
        # We compute permutation importance using Average Precision (AP),
        # which is more informative than ROC AUC under class imbalance.
        ap_scorer = make_scorer(average_precision_score, needs_proba=True)

        # Use stratified CV so each fold preserves class balance.
        # Importance is computed out-of-sample on held-out folds to reduce optimism bias.
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

        all_importances = []

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            # Guard: AP/ROC metrics require both classes in the evaluation fold.
            if len(np.unique(y_te)) < 2:
                print(f"Skip fold {fold_idx}: only one class in validation.")
                continue

            # Fit model using final hyperparameters on training portion of fold
            base_model = RandomForestClassifier(
                n_jobs=-1,
                random_state=SEED + fold_idx,  # vary seed slightly per fold
                **best_params
            )
            base_model.fit(X_tr, y_tr)

            # Permutation importance computed on held-out fold (true out-of-sample scoring)
            perm = permutation_importance(
                estimator=base_model,
                X=X_te,
                y=y_te,
                scoring=ap_scorer,     # Average Precision scorer
                n_repeats=1,           # modest repeats for demo speed; increase for stability
                n_jobs=-1,
                random_state=SEED + fold_idx,
            )

            # Store fold-level importance statistics
            fold_df = pd.DataFrame({
                "feature": X_te.columns,
                "importance_mean": perm.importances_mean,
                "importance_std":  perm.importances_std,
                "fold": fold_idx
            })

            all_importances.append(fold_df)

        # ---- Aggregate importance across folds ----
        # Summarize central tendency and stability across CV folds.
        combined = pd.concat(all_importances, ignore_index=True)

        summary = (
            combined
            .groupby("feature", as_index=False)
            .agg(
                mean_importance=("importance_mean", "mean"),
                median_importance=("importance_mean", "median"),
                iqr_importance=("importance_mean", lambda s: s.quantile(0.75) - s.quantile(0.25)),
                mean_std=("importance_std", "mean"),
                n_folds=("fold", "nunique"),
            )
            .sort_values("median_importance", ascending=False)
        )

        # Save fold-aggregated permutation importance scores
        summary.to_csv(output.importance_scores, index=False)
        print("Saved CV-based permutation importances (AP scorer).")

