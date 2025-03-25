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
        time=2160,  # Extend if needed (20 min instead of 10)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

        # Check class balance in train, test, and validation sets
        print("Training set size:", len(y_train), "Positive cases:", np.sum(y_train))
        print("Test set size:", len(y_test), "Positive cases:", np.sum(y_test))
        print("Validation set size:", len(y_val), "Positive cases:", np.sum(y_val))

        # Define hyperparameter search space for Randomized Search
        param_dist = {
            "n_estimators": [50, 100, 500, 1000, 1500],
            "max_depth": [3, 7, 12, 20, 50, 100],
            "max_features": ["sqrt", "log2", 0.5],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10, 20, 50],
            "min_samples_leaf": [1, 5, 10, 20, 50]
        }

        # Number of iterations for RandomizedSearchCV
        n_iter = 500  # Adjust as needed

        # Step 1: Run Randomized Search on Training Data
        rf = RandomForestClassifier(random_state=1)

        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist,
            n_iter=n_iter,  # Adjustable
            scoring="f1", cv=5, n_jobs=-1, random_state=1, verbose=3,
            return_train_score=True
        )

        random_search.fit(X_train, y_train)

        # Get **all** models' parameters and metrics
        all_results = random_search.cv_results_
        all_model_performance = []

        # Step 2: Evaluate Every Model
        best_overall_score = -np.inf
        best_model = None
        best_params = None
        best_val_f1 = -1
        best_threshold = None
        lambda_factor = 0.7

        # Sort and get the **top 10** models
        sorted_results = sorted(
            zip(all_results["mean_test_score"], all_results["params"]),
            key=lambda x: x[0], reverse=True
        )[:10]

        for i in range(n_iter):  # Ensure we process **every** iteration
            params = all_results["params"][i]
            print(f"Evaluating Model {i + 1}/{n_iter}: {params}")

            model = RandomForestClassifier(**params, random_state=1)
            model.fit(X_train, y_train)

            # Test Predictions
            test_preds = model.predict(X_test)
            test_probs = model.predict_proba(X_test)[:, 1]

            # Compute Test Metrics
            test_f1 = f1_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds)
            test_bal_acc = balanced_accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)

            # Compute Confusion Matrix
            tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

            # Find best threshold for validation set
            thresholds = np.linspace(0.1, 0.9, 20)
            f1_scores = [f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= t).astype(int)) for t in thresholds]
            best_t = thresholds[np.argmax(f1_scores)]
            val_f1 = max(f1_scores)

            # Compute Balanced Score
            balanced_score = min(test_f1, val_f1) - lambda_factor * abs(test_f1 - val_f1)

            print(f"Test F1: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}, Test Balanced Accuracy: {test_bal_acc}")
            print(f"Test TP: {tp}, Test FP: {fp}, Test TN: {tn}, Test FN: {fn}")
            print(f"Validation F1 (Best Threshold {best_t}): {val_f1}")

            if balanced_score > best_overall_score:
                best_overall_score = balanced_score
                best_model = model
                best_params = params
                best_val_f1 = val_f1
                best_threshold = best_t

            # Store results for CSV export
            all_model_performance.append({
                **params,
                "iteration": i + 1,
                "cv_f1": all_results["mean_test_score"][i],  # Original cross-validation F1
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
                "balanced_score": balanced_score
            })

        # Step 3: Save Best Model and Metrics
        output_data = {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "validation_f1": best_val_f1,
            "balanced_score": best_overall_score
        }

        with open(output.best_params, "w") as f:
            json.dump(output_data, f)

        print("Best Model Selected:", best_params)
        print("Best Validation F1:", best_val_f1)
        print("Best Balanced Score:", best_overall_score)

        # Save **ALL models' performance** to a CSV
        all_performance_df = pd.DataFrame(all_model_performance)
        all_performance_df.to_csv(output.all_models, index=False)  # New file storing **all** results

        # Save **only the top 10 models** separately
        top_10_performance_df = all_performance_df.nlargest(10, "balanced_score")
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
        time=2160,  # Extend if needed (20 min instead of 10)
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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

        print("Training set size:", len(y_train), "Positive cases:", np.sum(y_train))
        print("Test set size:", len(y_test), "Positive cases:", np.sum(y_test))
        print("Validation set size:", len(y_val), "Positive cases:", np.sum(y_val))

        # Define hyperparameter search space for HistGradientBoostingClassifier
        param_dist = {
            'learning_rate': np.linspace(0.01, 1.0, 50),
            'max_iter': [100, 200, 300, 500, 1000],
            'max_leaf_nodes': [15, 31, 63, 127, 255],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_leaf': [10, 20, 30, 50, 100],
            'l2_regularization': np.logspace(-4, 2, 20),
            'max_bins': [64, 128, 256, 512],
            'early_stopping': [True, False]
        }

        n_iter = 500

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
        best_val_f1 = -1
        best_threshold = None
        lambda_factor = 0.7

        for i in range(n_iter):
            params = all_results["params"][i]
            print(f"Evaluating Model {i + 1}/{n_iter}: {params}")

            model = HistGradientBoostingClassifier(**params, random_state=1)
            model.fit(X_train, y_train)

            test_probs = model.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)

            test_f1 = f1_score(y_test, test_preds)
            test_precision = precision_score(y_test, test_preds)
            test_recall = recall_score(y_test, test_preds)
            test_bal_acc = balanced_accuracy_score(y_test, test_preds)
            test_auc = roc_auc_score(y_test, test_probs)

            tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

            thresholds = np.linspace(0.1, 0.9, 20)
            f1_scores = [f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= t).astype(int)) for t in thresholds]
            best_t = thresholds[np.argmax(f1_scores)]
            val_f1 = max(f1_scores)

            balanced_score = min(test_f1, val_f1) - lambda_factor * abs(test_f1 - val_f1)

            print(f"Test F1: {test_f1}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test AUC: {test_auc}, Test Balanced Accuracy: {test_bal_acc}")
            print(f"Test TP: {tp}, Test FP: {fp}, Test TN: {tn}, Test FN: {fn}")
            print(f"Validation F1 (Best Threshold {best_t}): {val_f1}")

            if balanced_score > best_overall_score:
                best_overall_score = balanced_score
                best_model = model
                best_params = params
                best_val_f1 = val_f1
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
                "balanced_score": balanced_score
            })

        output_data = {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "validation_f1": best_val_f1,
            "balanced_score": best_overall_score
        }

        with open(output.best_params, "w") as f:
            json.dump(output_data, f)

        print("Best Model Selected:", best_params)
        print("Best Validation F1:", best_val_f1)
        print("Best Balanced Score:", best_overall_score)

        all_performance_df = pd.DataFrame(all_model_performance)
        all_performance_df.to_csv(output.all_models, index=False)

        top_10_performance_df = all_performance_df.nlargest(10, "balanced_score")
        top_10_performance_df.to_csv(output.top_models, index=False)

        print(f"Randomized search and evaluation complete. {n_iter} models evaluated and saved.")

rule random_forest_grid_search:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
    output:
        best_params = 'results/iteration1/random_forest/best_params.json',
    resources:
        time=720,  # Extend if needed (20 min instead of 10)
        mem_mb=64000,  # Reduce to 64GB unless testing needs more
        nodes=1,  # Keep to 1 node
        cpus_per_task=64,  # Reduce to 64 for efficiency
        ntasks=1
    run:
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, ParameterGrid
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            f1_score, precision_score, recall_score, precision_recall_curve, 
            balanced_accuracy_score, roc_auc_score
        )

        # Read in the data
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index('sample', inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop('y', axis=1)
        y = data['y']

        # **NEW: Split into Train (60%) / Validation (20%) / Test (20%)**
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

        # **Check class balance in train, validation, and test sets**
        train_pos = np.sum(y_train)
        val_pos = np.sum(y_val)
        test_pos = np.sum(y_test)

        print("Training set size:", len(y_train), "Positive cases:", train_pos)
        print("Validation set size:", len(y_val), "Positive cases:", val_pos)
        print("Test set size:", len(y_test), "Positive cases:", test_pos)
        print("Class distribution in train:", train_pos / len(y_train), "positive")
        print("Class distribution in validation:", val_pos / len(y_val), "positive")
        print("Class distribution in test:", test_pos / len(y_test), "positive")

        # Define hyperparameter grid
        param_grid = {
            "n_estimators": [500, 1000, 1500],
            "max_depth": [3, 7, 12, 20],  # None = unlimited depth
            "max_features": ["sqrt", "log2", 0.5],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 10, 50],
            "min_samples_leaf": [1, 10, 50]
        }

        # **Step 1: Select the best hyperparameters based on validation set**
        best_params = None
        best_val_f1 = -1  # Track the best validation score

        # Loop through parameter combinations manually
        for params in ParameterGrid(param_grid):
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)  # Train on training set only

            val_preds = model.predict(X_val)  # Predict on validation set
            val_f1 = f1_score(y_val, val_preds)

            print(f"Params: {params}, Validation F1: {val_f1}")

            # Store the best parameters based on validation performance
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_params = params

        print("Best parameters selected based on validation set:", best_params)

        # **Step 2: Train final model using train + validation sets**
        best_rf = RandomForestClassifier(**best_params, random_state=42)
        best_rf.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

        # **Step 3: Evaluate on the test set with multiple thresholds**
        y_probs = best_rf.predict_proba(X_test)[:, 1]  # Get probabilities

        # Define multiple thresholds to test
        thresholds = np.linspace(0.1, 0.9, 20)  # Test 20 thresholds between 0.1 and 0.9

        # Store F1, Precision, and Recall scores for each threshold
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)  # Apply threshold
            f1_scores.append(f1_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))

        # Find the best threshold based on the highest F1 score
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best threshold for F1 score: {best_threshold}")

        # Get final test predictions using the best threshold
        y_pred_best = (y_probs >= best_threshold).astype(int)
        test_f1_adjusted = f1_score(y_test, y_pred_best)

        # **Step 4: Compute Balanced Accuracy & AUC-ROC**
        test_bal_acc = balanced_accuracy_score(y_test, y_pred_best)
        test_auc = roc_auc_score(y_test, y_probs)

        print("Balanced Accuracy:", test_bal_acc)
        print("AUC-ROC Score:", test_auc)

        # **Step 5: Evaluate Train & Validation Performance for Overfitting Check**
        train_preds = best_rf.predict(X_train)
        val_preds = best_rf.predict(X_val)

        train_f1 = f1_score(y_train, train_preds)
        val_f1 = f1_score(y_val, val_preds)

        print("Train F1 Score:", train_f1)
        print("Validation F1 Score:", val_f1)
        print("Test F1 Score (before threshold tuning):", f1_score(y_test, (y_probs >= 0.5).astype(int)))
        print("Test F1 Score (after threshold tuning):", test_f1_adjusted)

        # **Step 6: Plot F1, Precision, and Recall for Different Thresholds**
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1_scores, label="F1 Score", marker='o')
        plt.plot(thresholds, precision_scores, label="Precision", linestyle="--", marker='s')
        plt.plot(thresholds, recall_scores, label="Recall", linestyle="--", marker='^')

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("F1, Precision, and Recall at Different Thresholds")
        plt.legend()
        plt.grid()

        # **Save the plot instead of just displaying it**
        plot_path = "results/iteration1/random_forest/f1_precision_recall_thresholds.png"
        plt.savefig(plot_path)  # Saves the figure to a file
        plt.close()  # Closes the plot to free memory

        print(f"Plot saved to: {plot_path}")

        # **Overfitting Warning**
        if train_f1 - val_f1 > 0.15:
            print("Warning: Model may be overfitting (train-validation F1 gap > 0.15)")

        # Save best parameters and performance to JSON
        output_data = {
            "best_params": best_params,
            "best_threshold": best_threshold,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1_before_threshold": f1_score(y_test, (y_probs >= 0.5).astype(int)),
            "test_f1_after_threshold": test_f1_adjusted,
            "balanced_accuracy": test_bal_acc,
            "auc_roc": test_auc
        }

        with open(output.best_params, "w") as f:
            json.dump(output_data, f)

rule random_forest:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
        best_params = 'results/iteration1/random_forest/best_params.json',
        #features = 'X_train.csv',
        #outcomes = 'y_train.csv',
    output:
        evaluation_metrics = 'results/iteration1/random_forest/evaluation_metrics.tsv',
        model = 'results/iteration1/random_forest/random_forest_model.pkl',
        importance_scores = 'results/iteration1/random_forest/feature_importance_scores.txt',
    resources:
        time= 300,
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

        best_params = best_params_data["best_params"]  # Extract hyperparameters
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

            # Append metrics to the results list
            results.append([f1, precision, recall, auc, accuracy, balanced_accuracy, tn, fp, fn, tp])

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
        modeling_input = 'results/iteration1/random_forest/inputs/modeling_input.txt',
        #features = 'X_train.csv',
        #outcomes = 'y_train.csv',
    output:
        evaluation_metrics = protected('results/iteration1/random_forest/gradient_boosting/evaluation_metrics.tsv'),
        model = protected('results/iteration1/random_forest/gradient_boosting/gradient_boosting_model.pkl'),
        importance_scores = 'results/iteration1/random_forest/gradient_boosting/feature_importance_scores.txt',
    resources:
        time= 5760,
        mem_mb= 120000,
    run:
        # Read in the data
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index('sample', inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop('y', axis=1)
        y = data['y']

        # Define HistGradientBoosting model
        model = HistGradientBoostingClassifier(
                    learning_rate=0.001,        # Shrinkage factor
                    max_iter=2000,              # Limited to fit within 4 days
                    max_depth=8,                # Controls interaction depth
                    min_samples_leaf=1,         # Minimum samples per terminal node
                    early_stopping=True,        # Enables automatic stopping
                    n_iter_no_change=50,        # Stops if no improvement in 50 rounds
                    validation_fraction=0.1,    # 10% of training data used for validation
                    verbose=1,                  # Prints progress
                    random_state=42
        )

        # Create Stratified K-fold cross-validation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)  # Reduced repeats for speed
        results = []

        # Perform cross-validation and collect scores
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            # Append metrics to the results list
            results.append([f1, precision, recall, auc, accuracy, balanced_accuracy, tn, fp, fn, tp])

        # Convert the results list to a DataFrame
        results_df = pd.DataFrame(results, columns=['F1 Score', 'Precision', 'Recall', 'AUC', 'Accuracy', 'Balanced Accuracy', 'TN', 'FP', 'FN', 'TP'])
        # Save the DataFrame to a tab-delimited file
        results_df.to_csv(output.evaluation_metrics, sep='\t', index=False)

        # Train final model on full dataset
        model.fit(X, y)
        dump(model, output.model)

        # Compute permutation-based feature importance
        perm_importances = permutation_importance(model, X, y, n_repeats=10, random_state=42)

        # Convert to DataFrame
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importances.importances_mean
        })

        # Sort and save feature importances
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importances.to_csv(output.importance_scores, header=False, index=False)

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
