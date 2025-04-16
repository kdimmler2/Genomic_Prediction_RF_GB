import pandas as pd
import numpy as np
import random
from pathlib import Path
from joblib import dump, load
import statistics
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statistics import mean

from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

include: 'src/utils.py'

rule all:
    input:
        'results/iteration1/inputs/modeling.table',
        'results/iteration1/inputs/raw_tables/',
        'results/iteration1/inputs/modeling_input.txt',
        'results/iteration1/random_forest/best_random_params.json',
        'results/iteration1/gradient_boosting/best_random_params.json',
        'results/iteration1/random_forest/feature_importance_scores.txt',
        'results/iteration1/gradient_boosting/evaluation_metrics.tsv',
#        'results/iteration1/random_forest/best_params.json',
#        'results/iteration1/random_forest/evaluation_metrics.tsv',
#        'results/iteration1/random_forest/gradient_boosting/evaluation_metrics.tsv',
#        'results/iteration1/random_forest/random_forest_model.pkl',
#        'results/iteration1/random_forest/feature_importance_scores.txt',
#        'results/iteration1/random_forest/top_80_features.txt',
#        'results/iteration1/random_forest/BackElim.vcf.gz',

rule gatk_table:
    input:
        training_vcf = config['modeling_vcf'], 
    output:
        table = 'results/iteration1/inputs/modeling.table', 
    resources:
        time    = 60,
        mem_mb  = 24000,
        cpus    = 4,
    shell:
        '''
            gatk VariantsToTable \
                -V {input.training_vcf} \
                -F ID -F CHROM -F POS -F REF -F ALT -GF GT \
                --split-multi-allelic true \
                -O {output.table}
        '''

checkpoint split_table:
    input:
        table = 'results/iteration1/inputs/modeling.table',
    output:
        split_tables = directory('results/iteration1/inputs/raw_tables/'),
    resources:
        time    = 10,
        mem_mb  = 24000,
        cpus    = 4,
    run:
        def split_file_by_lines(input_file, output_dir, lines_per_file):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(input_file, 'r') as f:
                header = next(f)  # Read the header
                lines_written = 0
                file_num = 1
                output_file = output_dir + '/part_' + str(file_num) + '.txt'
                with open(output_file, 'w') as out:
                    out.write(header)
                    for line in f:
                        out.write(line)
                        lines_written += 1
                        if lines_written >= lines_per_file:
                            lines_written = 0
                            file_num += 1
                            output_file = output_dir + '/part_' + str(file_num) + '.txt'
                            out.close()  # Close the previous file
                            out = open(output_file, 'w')
                            out.write(header)
        
        input_file = input.table
        output_dir = 'results/iteration1/inputs/raw_tables'
        lines_per_file = 1000
        split_file_by_lines(input_file, output_dir, lines_per_file)


rule convert_tables:
    input:
        table = 'results/iteration1/inputs/raw_tables/part_{num}.txt'
    output:
        converted_table = 'results/iteration1/inputs/converted_tables/part_{num}.txt',
    params:
        phenofile = config['modeling_pheno_file'], 
    resources:
        time    = 30,
        mem_mb  = 24000,
        cpus    = 4,
    run:
        # Read in the table
        df = pd.read_csv(input.table, delimiter='\t')

        # Make sample ID the index
        df.set_index('ID', inplace=True)

        # Transpose the dataframe so that sample are the rows
        df2 = df.transpose()

        # Convert each genotype to a dummy variable
        for c in range(len(df2.columns)):
            for r in range(4, len(df2)):
                ref = df2.iat[2, c]
                alt = df2.iat[3, c]
                gt = df2.iat[r,c]
                if df2.iat[r,c] == ref + '/' + ref or df2.iat[r,c] == ref + '|' + ref:
                    df2.iat[r, c] = 0
                elif df2.iat[r,c] == ref + '/' + alt or df2.iat[r,c] == ref + '|' + alt:
                    df2.iat[r, c] = 1
                elif df2.iat[r,c] == alt + '/' + alt or df2.iat[r,c] == alt + '|' + alt:
                    df2.iat[r, c] = 2
                elif alt in str(df2.iat[r, c]):
                    df2.iat[r, c] = 1
                else:
                    df2.iat[r, c] = 0

        # Remove CHROM, POS, REF, ALT
        df3 = df2[4:]

        # Make samples the index and variant IDs the header
        df4 = df3.copy()
        df4.reset_index(inplace=True)
        df4.rename(columns={'index': 'sample'}, inplace=True)

        # Add phenos column and remove .GT from sample IDs
        for r in range(len(df4)):
            split = df4.at[r, 'sample'].split('.')
            df4.at[r, 'sample'] = split[0]
        df4.insert(1, 'phenos', 5)

        # Read in pheno file generated by get_phenos.py
        infile = open(params.phenofile, 'rt')

        # Create dictionaries of each phenotype of interest
        phenos = {}
        sexes = {}
        gaits = {}

        for line in infile:
            line = line.rstrip()
            split = line.split(' ')
            phenos[split[1]] = split[0]
            sexes[split[1]] = split[2]
            gaits[split[1]] = split[3]

        # Add columns for each phenotype of interest
        df5 = df4
        df5.insert(2, 'sex', 5)
        df5.insert(3, 'gait', 5)

        # Use dictionaries to replace phenotype values for each sample
        for r in range(len(df5)):
            if df5.at[r, 'sample'] in phenos:
                df5.at[r, 'phenos'] = int(phenos[df5.at[r, 'sample']])
            if df5.at[r, 'sample'] in sexes:
                df5.at[r, 'sex'] = int(sexes[df5.at[r, 'sample']])
            if df5.at[r, 'sample'] in gaits:
                df5.at[r, 'gait'] = int(gaits[df5.at[r, 'sample']])

        df5.rename(columns={'phenos': 'y'}, inplace=True)
        df5.set_index('sample', inplace=True)

        df5.to_csv(output.converted_table, header=True, index=True, sep='\t')

def get_file_nums(wildcards):
    ck_output = checkpoints.split_table.get(**wildcards).output[0]
    parts = str(Path(ck_output) / 'part_{num}.txt')
    PARTS, = glob_wildcards(parts)
    return sorted(expand(
        'results/iteration1/inputs/converted_tables/part_{num}.txt',
        num=PARTS
    ))
    
rule combine_tables:
    input:
        tables = get_file_nums,
    output:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
    params:
        directory = 'results/iteration1/inputs/converted_tables/',
    resources:
        time    = 120,
        mem_mb  = 24000,
        cpus    = 4,
    run:
        # List all files in the directory
        #files = os.listdir(params.directory)

        files = input.tables

        # Initialize an empty DataFrame to store the combined data
        combined_df = pd.DataFrame()

        # Read each file and merge based on the common column
        for file in files:
            if file.endswith('.txt'):
                df = pd.read_csv(file, sep='\t')
                if combined_df.empty:
                    labels = df.iloc[:, :4]
                    combined_df = df.iloc[:, 4:]
                else:
                    combined_df = pd.merge(combined_df, df.iloc[:, 4:], left_index=True, right_index=True, how='outer')

        combined_df = pd.concat([labels, combined_df], axis=1)
        combined_df.set_index('sample', inplace=True)

        # Save the combined DataFrame to a new file
        combined_df.to_csv(output.modeling_input, index=True, header=True, sep='\t')

rule random_RF_grid_search:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
    output:
        best_params = 'results/iteration1/random_forest/best_random_params.json',
        all_models = 'results/iteration1/random_forest/all_models.csv',
        top_models = 'results/iteration1/random_forest/top_models.csv',
    resources:
        time=5760,  # Extend if needed (20 min instead of 10)
        mem_mb=64000,  # Reduce to 64GB unless testing needs more
        nodes=1,  # Keep to 1 node
        cpus_per_task=64,  # Reduce to 64 for efficiency
        ntasks=1
    run:
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
        )

        # Read in the data
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index("sample", inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop("y", axis=1)
        y = data["y"]

        # Split into Train (60%) / Test (20%) / Validation (20%)
        X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1, stratify=y_train_full)

        print("Training set size:", len(y_train), "Positive cases:", np.sum(y_train))
        print("Test set size:", len(y_test), "Positive cases:", np.sum(y_test))
        print("Validation set size:", len(y_val), "Positive cases:", np.sum(y_val))

        param_dist = {
            "n_estimators": [50, 100, 500, 1000, 1500],
            "max_depth": [3, 7, 12, 20, 50, 100],
            "max_features": ["sqrt", "log2", 0.5],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [20, 50, 100, 200, 500],
            "min_samples_leaf": [1, 5, 10, 20, 50]
        }

        n_iter = 500
        rf = RandomForestClassifier(random_state=1)

        random_search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            random_state=1,
            verbose=3,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        all_results = random_search.cv_results_
        all_model_performance = []

        best_overall_score = -np.inf
        best_model = None
        best_params = None
        best_threshold = None

        for i in range(n_iter):
            params = all_results["params"][i]
            print(f"Evaluating Model {i + 1}/{n_iter}: {params}")

            model = RandomForestClassifier(**params, random_state=1)
            model.fit(X_train, y_train)

            test_probs = model.predict_proba(X_test)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 20)
            test_f1_scores = [f1_score(y_test, (test_probs >= t).astype(int)) for t in thresholds]
            best_t = thresholds[np.argmax(test_f1_scores)]
            test_preds = (test_probs >= best_t).astype(int)

            test_f1 = f1_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds)
            test_bal_acc = balanced_accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)

            tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

            model_val = RandomForestClassifier(**params, random_state=1)
            model_val.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
            val_probs = model_val.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= best_t).astype(int)
            val_f1 = f1_score(y_val, val_preds)

            print(f"Test F1: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}, Test Balanced Accuracy: {test_bal_acc}")
            print(f"Test TP: {tp}, Test FP: {fp}, Test TN: {tn}, Test FN: {fn}")
            print(f"Validation F1 (Fixed Threshold {best_t}): {val_f1}")

            if test_f1 > best_overall_score:
                best_overall_score = test_f1
                best_model = model
                best_params = params
                best_threshold = best_t

            all_model_performance.append({
                **params,
                "iteration": i + 1,
                "cv_f1": all_results["mean_test_score"][i],
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "test_bal_acc": test_bal_acc,
                "test_tp": tp,
                "test_fp": fp,
                "test_tn": tn,
                "test_fn": fn,
                "val_f1": val_f1,
                "best_threshold": best_t,
                "balanced_score": val_f1
            })

        output_data = {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "balanced_score": best_overall_score
        }

        with open(output.best_params, "w") as f:
            json.dump(output_data, f)

        print("Best Model Selected:", best_params)
        print("Best Balanced Score (Test F1):", best_overall_score)

        all_performance_df = pd.DataFrame(all_model_performance)
        all_performance_df.to_csv(output.all_models, index=False)

        top_10_performance_df = all_performance_df.nlargest(10, "val_f1")
        top_10_performance_df.to_csv(output.top_models, index=False)

        print(f"Randomized search and evaluation complete. {n_iter} models evaluated and saved.")

rule random_GB_grid_search:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
    output:
        best_params = 'results/iteration1/gradient_boosting/best_random_params.json',
        all_models = 'results/iteration1/gradient_boosting/all_models.csv',
        top_models = 'results/iteration1/gradient_boosting/top_models.csv',
    resources:
        time=5760,  # Extend if needed (20 min instead of 10)
        mem_mb=64000,  # Reduce to 64GB unless testing needs more
        nodes=1,  # Keep to 1 node
        cpus_per_task=64,  # Reduce to 64 for efficiency
        ntasks=1
    run:
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
        )

        # Read in the data
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index("sample", inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop("y", axis=1)
        y = data["y"]

        # Split into Train (60%) / Test (20%) / Validation (20%)
        X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1, stratify=y_train_full)

        print("Training set size:", len(y_train), "Positive cases:", np.sum(y_train))
        print("Test set size:", len(y_test), "Positive cases:", np.sum(y_test))
        print("Validation set size:", len(y_val), "Positive cases:", np.sum(y_val))

        # Define hyperparameter search space
        param_dist = {
            'learning_rate': [0.1, 0.3, 0.5, 0.7, 1.0],
            'max_iter': [100, 300, 500, 1000],
            'max_leaf_nodes': [31, 63, 127, 255],
            'max_depth': [3, 7, 12, 20, 50, 100],
            'min_samples_leaf': [1, 10, 50],
            'l2_regularization': [0.1, 1.0, 3.0, 10.0],
            'max_bins': [64, 255],
            'early_stopping': [True]
        }

        n_iter = 1000
        model = HistGradientBoostingClassifier(random_state=1)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=5,
            verbose=3,
            n_jobs=-1,
            random_state=1,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        all_results = random_search.cv_results_
        all_model_performance = []

        best_overall_score = -np.inf
        best_model = None
        best_params = None
        best_threshold = None

        for i in range(n_iter):
            params = all_results["params"][i]
            print(f"Evaluating Model {i + 1}/{n_iter}: {params}")

            model = HistGradientBoostingClassifier(**params, random_state=1)
            model.fit(X_train, y_train)

            test_probs = model.predict_proba(X_test)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 20)
            test_f1_scores = [f1_score(y_test, (test_probs >= t).astype(int)) for t in thresholds]
            best_t = thresholds[np.argmax(test_f1_scores)]
            test_preds = (test_probs >= best_t).astype(int)

            test_f1 = f1_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds)
            test_bal_acc = balanced_accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)

            tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

            model_val = HistGradientBoostingClassifier(**params, random_state=1)
            model_val.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
            val_probs = model_val.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= best_t).astype(int)
            val_f1 = f1_score(y_val, val_preds)

            print(f"Test F1: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}, Test Balanced Accuracy: {test_bal_acc}")
            print(f"Test TP: {tp}, Test FP: {fp}, Test TN: {tn}, Test FN: {fn}")
            print(f"Validation F1 (Fixed Threshold {best_t}): {val_f1}")

            if test_f1 > best_overall_score:
                best_overall_score = test_f1
                best_model = model
                best_params = params
                best_threshold = best_t

            all_model_performance.append({
                **params,
                "iteration": i + 1,
                "cv_f1": all_results["mean_test_score"][i],
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_auc": test_auc,
                "test_bal_acc": test_bal_acc,
                "test_tp": tp,
                "test_fp": fp,
                "test_tn": tn,
                "test_fn": fn,
                "val_f1": val_f1,
                "best_threshold": best_t,
                "balanced_score": val_f1  # final generalization metric
            })

        output_data = {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "balanced_score": best_overall_score
        }

        with open(output.best_params, "w") as f:
            json.dump(output_data, f)

        print("Best Model Selected:", best_params)
        print("Best Balanced Score (Test F1):", best_overall_score)

        all_performance_df = pd.DataFrame(all_model_performance)
        all_performance_df.to_csv(output.all_models, index=False)

        top_10_performance_df = all_performance_df.nlargest(10, "val_f1")
        top_10_performance_df.to_csv(output.top_models, index=False)

        print(f"Randomized search and evaluation complete. {n_iter} models evaluated and saved.")

rule random_forest:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
        best_params = 'results/iteration1/random_forest/best_random_params.json',
        #features = 'X_train.csv',
        #outcomes = 'y_train.csv',
    output:
        evaluation_metrics = 'results/iteration1/random_forest/evaluation_metrics.tsv',
        model = 'results/iteration1/random_forest/random_forest_model.pkl',
        importance_scores = 'results/iteration1/random_forest/feature_importance_scores.txt',
    resources:
        time= 1440,
        mem_mb= 60000,
        nodes= 1,
        cpus_per_task=128,
        ntasks=1,
    run:
        import json
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, 
            balanced_accuracy_score, confusion_matrix
        )
        from joblib import dump

        # **Step 1: Load Best Hyperparameters & Threshold from JSON**
        with open(input.best_params, "r") as f:
            best_params_data = json.load(f)

        best_params = {
            'n_estimators': 1500,
            'max_depth': 50,
            'min_samples_split': 50,
            'min_samples_leaf': 1,
            'max_features': 0.5,
            'criterion': 'entropy'
            }

        best_threshold = best_params_data["best_threshold"]  # Extract optimized threshold
        print(f"Loaded Best Hyperparameters: {best_params}")
        print(f"Using Optimized Threshold: {best_threshold}")

        # Open output file for writing
        outfile = open(output.evaluation_metrics, "wt")

        # **Step 2: Read in the data**
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index("sample", inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop("y", axis=1)
        y = data["y"]

        # **Step 3: Build RandomForest Model with Best Parameters**
        BRF = RandomForestClassifier(**best_params, n_jobs=-1, random_state=42)  # Use best params

        # **Step 4: Create Stratified K-fold Cross-Validation**
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
        results = []

        # **Step 5: Perform Cross-Validation and Apply Optimized Threshold**
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            BRF.fit(X_train, y_train)
            
            # **Use Probability Predictions Instead of Default 0.5 Threshold**
            y_probs = BRF.predict_proba(X_test)[:, 1]  # Get probability scores
            y_pred = (y_probs >= best_threshold).astype(int)  # Apply best threshold

            # **Step 6: Calculate Evaluation Metrics**
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_probs)  # Use probabilities for AUC
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            # **Step 7: Calculate Confusion Matrix**
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Append metrics to the results list
            results.append([f1, precision, recall, auc, accuracy, balanced_accuracy, sensitivity, specificity, tn, fp, fn, tp])

        # **Step 8: Save Evaluation Metrics to File**
        results_df = pd.DataFrame(
            results, columns=["F1 Score", "Precision", "Recall", "AUC", "Accuracy", "Balanced Accuracy", "TN", "FP", "FN", "TP"]
        )
        results_df.to_csv(output.evaluation_metrics, sep="\t", index=False)

        # **Step 9: Train Final Model on Full Dataset**
        BRF.fit(X, y)

        # **Use Best Threshold for Final Predictions**
        y_probs_final = BRF.predict_proba(X)[:, 1]  # Get probability scores
        y_pred_final = (y_probs_final >= best_threshold).astype(int)  # Apply best threshold

        # Save the trained model
        dump(BRF, output.model)

        # **Step 10: Compute Feature Importances**
        importances = BRF.feature_importances_
        feature_importances = pd.DataFrame(importances, index=X.columns, columns=["importance"])
        feature_importances.sort_values(by="importance", ascending=False, inplace=True)
        feature_importances = feature_importances.reset_index().rename(columns={"index": "feature"})

        # Save Feature Importances
        feature_importances.to_csv(output.importance_scores, header=False, index=False)

        print("Model training and evaluation complete. Results saved.")

rule gradient_boosting:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
        best_params = 'results/iteration1/gradient_boosting/best_random_params.json',
        #features = 'X_train.csv',
        #outcomes = 'y_train.csv',
    output:
        evaluation_metrics = 'results/iteration1/gradient_boosting/evaluation_metrics.tsv',
        model = 'results/iteration1/gradient_boosting/gradient_boosting_model.pkl',
        importance_scores = 'results/iteration1/gradient_boosting/feature_importance_scores.txt',
    resources:
        time= 1440,
        mem_mb= 120000,
    run:
        from sklearn.ensemble import HistGradientBoostingClassifier

        # --- Step 1: Load Best Threshold from JSON ---
        with open(input.best_params, "r") as f:
            best_params_data = json.load(f)

        best_params = {
            'learning_rate': 0.5,
            'max_iter': 300,
            'max_leaf_nodes': 31,
            'max_depth': 3,
            'min_samples_leaf': 10,
            'l2_regularization': 1.0,
            'max_bins': 255,
            'early_stopping': True
        }
        best_threshold = best_params_data["best_threshold"]
        print(f"Loaded Best Hyperparameters: {best_params}")
        print(f"Using Optimized Threshold: {best_threshold}")

        # --- Step 2: Load Data ---
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index("sample", inplace=True)
        X = data.drop("y", axis=1)
        y = data["y"]

        # --- Step 3: Build Model ---
        model = HistGradientBoostingClassifier(**best_params, random_state=42)

        # --- Step 4: Cross-Validation Setup ---
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
        results = []

        # --- Step 5: Cross-Validation Loop ---
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)

            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs >= best_threshold).astype(int)

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_probs)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            results.append([f1, precision, recall, auc, accuracy, balanced_accuracy, sensitivity, specificity, tn, fp, fn, tp])

        # --- Step 6: Save Evaluation Metrics ---
        results_df = pd.DataFrame(
            results, columns=["F1 Score", "Precision", "Recall", "AUC", "Accuracy",
                              "Balanced Accuracy", "Sensitivity", "Specificity", "TN", "FP", "FN", "TP"]
        )
        results_df.to_csv(output.evaluation_metrics, sep="\t", index=False)

        # --- Step 7: Train Final Model on Full Data ---
        model.fit(X, y)
        y_probs_final = model.predict_proba(X)[:, 1]
        y_pred_final = (y_probs_final >= best_threshold).astype(int)

        # --- Step 8: Save Model ---
        dump(model, output.model)

        # --- Step 9: Feature Importances ---
        importances = model.feature_importances_
        feature_importances = pd.DataFrame(importances, index=X.columns, columns=["importance"])
        feature_importances = feature_importances.sort_values(by="importance", ascending=False).reset_index().rename(columns={"index": "feature"})
        feature_importances.to_csv(output.importance_scores, header=False, index=False)

        print("Model training and evaluation complete. Results saved.")

rule top_80_features:
    input:
        feature_importances = 'results/iteration1/random_forest/feature_importance_scores.txt',
    output:
        top80 = 'results/iteration1/random_forest/top_80_features.txt',
    resources:
        time    = 10,
        mem_mb  = 20000,
    run:
        df = pd.read_csv(input.feature_importances, header=None)

        # Calculate the number of rows to select (80% of total rows)
        num_rows_to_select = int(0.8 * len(df))

        # Select the first 80% of rows
        df_first_80_percent = df.iloc[:num_rows_to_select]

        df_first_80_percent.iloc[1:,0].to_csv(output.top80, index=False, header=False)

rule top_80_variants:
    input:
        modeling_vcf = config['modeling_vcf'],
        top80 = 'results/iteration1/random_forest/top_80_features.txt',
    output:
        first_iteration_vcf = 'results/iteration1/random_forest/BackElim.vcf.gz',
    resources:
        time    = 10,
        mem_mb  = 2000,
    shell:
        '''
            bcftools view -Oz -o {output.first_iteration_vcf} --include ID=@{input.top80} {input.modeling_vcf}

            gatk IndexFeatureFile -I {output.first_iteration_vcf}

            cp results/iteration1/random_forest results/temp -r

            mv results/temp results/previous_iteration
        '''
