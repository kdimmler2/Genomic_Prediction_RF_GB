import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)

rule all:
    input:
        '../results/test_set_metrics/test_probs.tsv',

rule rf_internal_test_fixed:
    threads: 4
    input:
        modeling_input = config['input_data'],
        splits_npz     = '../results/hyperparameter_tuning/splits.npz'
    output:
        test_probs_tsv   = '../results/test_set_metrics/test_probs.tsv',
        test_metrics_tsv = '../results/test_set_metrics/test_metrics.tsv'
    run:
        # ---- Locked configuration (manual handoff) ----
        # Hyperparameters are selected manually from the randomized search ../results
        # (tradeoffs across AP/MCC/specificity, etc.) and defined in the config file
        best_params = config['best_params']

        # Decision threshold chosen from the calibration/threshold-sweep step.
        # This encodes the desired operating point (e.g., bias toward sensitivity vs specificity).
        chosen_threshold = config['chosen_threshold']

        # ---- Load modeling matrix ----
        # Expect rows = samples and a binary label column y.
        data = pd.read_csv(input.modeling_input, sep='\t')
        data.set_index('sample', inplace=True)
        X = data.drop('y', axis=1)
        y = data['y']

        # ---- Load fixed split indices ----
        # Reuse the exact same split indices produced earlier to guarantee reproducibility.
        # Here we treat:
        #   - train_idx as the FIT set used to train the locked model
        #   - test_idx  as an internal held-out test set for final evaluation
        splits = np.load(input.splits_npz, allow_pickle=True)
        fit_idx          = splits['train_idx']   # FIT
        internaltest_idx = splits['test_idx']    # INTERNAL-TEST

        X_fit,  y_fit  = X.loc[fit_idx], y.loc[fit_idx]
        X_test, y_test = X.loc[internaltest_idx], y.loc[internaltest_idx]

        # ---- Fit final model on FIT only ----
        # No further tuning is done here; evaluation is performed on INTERNAL-TEST.
        SEED = config['seed']
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_params)
        model.fit(X_fit, y_fit)

        # ---- Predict probabilities + apply fixed threshold on INTERNAL-TEST ----
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= chosen_threshold).astype(int)

        # ---- Evaluate performance ----
        # Report both threshold-free metrics (ROC AUC, AP) and threshold-based metrics
        # at the chosen operating point (precision/recall/specificity/MCC, etc.).
        test_auc = roc_auc_score(y_test, test_probs)
        test_ap  = average_precision_score(y_test, test_probs)

        prec = precision_score(y_test, test_preds, zero_division=0)
        rec  = recall_score(y_test, test_preds)
        f1   = f1_score(y_test, test_preds, zero_division=0)
        bal  = balanced_accuracy_score(y_test, test_preds)

        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
        sens = tp / (tp + fn)  # sensitivity (same as recall)
        spec = tn / (tn + fp)  # specificity (TNR)

        # MCC can be undefined if predictions collapse to a single class
        if test_preds.sum() == 0 or test_preds.sum() == len(test_preds):
            mcc = 0.0
        else:
            mcc = matthews_corrcoef(y_test, test_preds)

        # (Optional) Explicit label order for consistency in any downstream parsing
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()

        # ---- Write outputs ----
        # 1) Per-sample probabilities and predicted classes on INTERNAL-TEST
        pd.DataFrame({
            'sample': X_test.index,
            'y_true': y_test.values,
            'prob':   test_probs,
            'pred':   test_preds
        }).to_csv(output.test_probs_tsv, sep='\t', index=False)

        # 2) Summary metric table for easy reporting/plotting
        pd.DataFrame([
            ('threshold',           float(chosen_threshold)),
            ('n_test',              int(len(y_test))),
            ('n_pos_test',          int(y_test.sum())),
            ('n_pred_pos',          int(test_preds.sum())),
            ('n_pred_neg',          int(len(test_preds) - test_preds.sum())),
            ('tp',                  int(tp)),
            ('fp',                  int(fp)),
            ('tn',                  int(tn)),
            ('fn',                  int(fn)),
            ('precision',           float(prec)),
            ('recall',              float(rec)),
            ('f1',                  float(f1)),
            ('balanced_accuracy',   float(bal)),
            ('mcc',                 float(mcc)),
            ('roc_auc',             float(test_auc)),
            ('average_precision',   float(test_ap)),
            ('sensitivity',         float(sens)),
            ('specificity',         float(spec)),
        ], columns=['metric', 'value']).to_csv(output.test_metrics_tsv, sep='\t', index=False)
