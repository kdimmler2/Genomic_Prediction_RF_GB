import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for batch/threaded execution
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)

rule all:
    input:
        'results/threshold_testing/threshold_table.tsv',

rule gb_threshold_prep:
    input:
        modeling_input   = config['input_data'],
        splits_npz       = 'results/hyperparameter_tuning/splits.npz'
    output:
        cal_probs_tsv      = 'results/threshold_testing/cal_probs.tsv',
        cal_pr_png         = 'results/threshold_testing/cal_pr_curve.png',
        cal_roc_png        = 'results/threshold_testing/cal_roc_curve.png',
        threshold_table_tsv = 'results/threshold_testing/threshold_table.tsv'
    resources:
        time=60,
        mem_mb=12000,
        nodes=1,
        cpus_per_task=4,
        ntasks=1
    threads: 4
    run:
        # ---- Load modeling matrix ----
        # Expect rows = samples, columns = features + binary outcome y.
        data = pd.read_csv(input.modeling_input, sep='\t')
        data.set_index('sample', inplace=True)
        X = data.drop('y', axis=1)
        y = data['y']

        # ---- Load fixed splits + manually selected hyperparameters ----
        # Hyperparameters were selected after inspecting CV results from the search step.
        # Splits are reused to guarantee identical train/cal partitions across downstream rules.
        best_params = config['best_params']

        splits = np.load(input.splits_npz, allow_pickle=True)
        fit_idx = splits['train_idx']   # training split used to fit final model
        cal_idx = splits['eval_idx']    # held-out calibration split

        X_fit, y_fit = X.loc[fit_idx], y.loc[fit_idx]
        X_cal, y_cal = X.loc[cal_idx], y.loc[cal_idx]

        # ---- Fit final Random Forest with locked hyperparameters ----
        # No further tuning is performed here; this step evaluates the chosen configuration.
        SEED = config['seed']
        model = HistGradientBoostingClassifier(random_state=SEED, **best_params)
        model.fit(X_fit, y_fit)

        # ---- Predict probabilities on calibration split ----
        # Probabilities are used for threshold-free metrics (PR/ROC)
        # and to evaluate candidate decision thresholds.
        cal_probs = model.predict_proba(X_cal)[:, 1]
        cal_df = pd.DataFrame({'sample': X_cal.index, 'y': y_cal.values, 'prob': cal_probs})
        cal_df.to_csv(output.cal_probs_tsv, sep='\t', index=False)

        # ---- Precision–Recall curve (threshold-free evaluation) ----
        # AP is emphasized because it is more informative under class imbalance.
        prec, rec, _ = precision_recall_curve(y_cal, cal_probs)
        pr_auc = auc(rec, prec)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('CAL PR curve (AP = ' + str(round(pr_auc, 3)) + ')')
        plt.grid(True, ls='--', alpha=0.4)
        plt.savefig(output.cal_pr_png, bbox_inches='tight', dpi=200)
        plt.close()

        # ---- ROC curve (threshold-free evaluation) ----
        # Included for completeness and comparability with common benchmarks.
        fpr, tpr, _ = roc_curve(y_cal, cal_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('CAL ROC curve (AUC = ' + str(round(roc_auc, 3)) + ')')
        plt.grid(True, ls='--', alpha=0.4)
        plt.savefig(output.cal_roc_png, bbox_inches='tight', dpi=200)
        plt.close()

        # ---- Threshold sweep (0.00–1.00 step 0.05) ----
        # Evaluate classification metrics across candidate probability cutoffs.
        # This enables selecting an operating point aligned with domain priorities
        # (e.g., favor sensitivity vs. specificity).
        grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
        rows = []
        n = len(y_cal)

        for t in grid:
            preds = (cal_probs >= t).astype(int)

            # Robust metric computation (avoid undefined precision/F1)
            p_ = precision_score(y_cal, preds, zero_division=0)
            r_ = recall_score(y_cal, preds)
            f1 = f1_score(y_cal, preds, zero_division=0)
            bal = balanced_accuracy_score(y_cal, preds)

            tn, fp, fn, tp = confusion_matrix(y_cal, preds).ravel()
            sens = tp / (tp + fn)  # sensitivity
            spec = tn / (tn + fp)  # specificity

            # MCC undefined if predictions collapse to a single class
            if preds.sum() == 0 or preds.sum() == n:
                mcc = 0.0
            else:
                mcc = matthews_corrcoef(y_cal, preds)

            rows.append([
                float(t), p_, r_, f1, bal, mcc, sens, spec,
                int(preds.sum()), int(n - preds.sum())
            ])

        thresh_table = pd.DataFrame(rows, columns=[
            'threshold', 'precision', 'recall', 'f1',
            'balanced_accuracy', 'mcc', 'sensitivity', 'specificity',
            'n_predicted_positive', 'n_predicted_negative'
        ])

        # Save full threshold-performance landscape for manual inspection
        thresh_table.to_csv(output.threshold_table_tsv, sep='\t', index=False)
