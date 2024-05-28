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

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

include: 'src/utils.py'

rule all:
    input:
        'results/iteration1/inputs/modeling.table',
        'results/iteration1/inputs/raw_tables/',
        'results/iteration1/inputs/modeling_input.txt',
        'results/iteration1/evaluation_metrics.tsv',
        'results/iteration1/random_forest_model.pkl',
        'results/iteration1/feature_importance_scores.txt',
        'results/iteration1/top_80_features.txt',
        'results/iteration1/BackElim.vcf.gz',

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

rule random_forest:
    input:
        modeling_input = 'results/iteration1/inputs/modeling_input.txt',
        #features = 'X_train.csv',
        #outcomes = 'y_train.csv',
    output:
        evaluation_metrics = 'results/iteration1/evaluation_metrics.tsv',
        model = 'results/iteration1/random_forest_model.pkl',
        importance_scores = 'results/iteration1/feature_importance_scores.txt',
    resources:
        time= 300,
        mem_mb= 60000,
        nodes= 1,
        cpus_per_task=128,
        ntasks=1,
    run:
        outfile = open(output.evaluation_metrics, 'wt')

        # Read in the data
        data = pd.read_csv(input.modeling_input, delimiter='\t')
        data.set_index('sample', inplace=True)

        # Split data into features (X) and outcomes (y)
        X = data.drop('y', axis=1)
        y = data['y']

        # Build BRF model
        BRF = BalancedRandomForestClassifier(criterion='entropy', n_estimators=1000, max_depth=3, max_features='sqrt', n_jobs=-1, random_state=42)
        # Create Stratified K-fold cross validation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
        results = []
        # Perform cross-validation and collect scores
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            BRF.fit(X_train, y_train)
            y_pred = BRF.predict(X_test)

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

#        scoring = ['f1', 'recall', 'precision', 'sensitivity', 'specificity', 'roc_auc', 'balanced_accuracy', 'accuracy']
#        # Evaluate BRF model
#        scores = cross_validate(BRF, X, y, scoring=scoring, cv=cv)
       # # Get average evaluation metrics
       # print('Mean f1: ' + str(mean(scores['test_f1'])), file=outfile)
       # print('Mean recall: ' + str(mean(scores['test_recall'])), file=outfile)
       # print('Mean precision: ' + str(mean(scores['test_precision'])), file=outfile)

      #  # split the data into 90% training and 10% testing
      #  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
#        #Train BRF
#        BRF.fit(X, y)
#        #BRF prediction result
#        y_pred = BRF.predict(X_test)
#
#        # Plot confusion matrix with numbers
#        fig = confusion_matrix(y_test, y_pred)
#        plt.imshow(fig, interpolation='nearest', cmap=plt.cm.Blues)
#        plt.title('Confusion matrix')
#        plt.colorbar()
#
#        classes = np.unique(y_test)
#        tick_marks = np.arange(len(classes))
#        plt.xticks(tick_marks, classes)
#        plt.yticks(tick_marks, classes)
#
#        plt.ylabel('True label')
#        plt.xlabel('Predicted label')
#        plt.tight_layout()
#
#        # Add numbers in the boxes
#        thresh = fig.max() / 2.
#        for i in range(fig.shape[0]):
#            for j in range(fig.shape[1]):
#                plt.text(j, i, format(fig[i, j], 'd'),
#                         ha="center", va="center",
#                         color="white" if fig[i, j] > thresh else "black")
#
#        # Save the plot as a PNG file
#        plt.savefig(output.confusion_matrix, format='png')
#
        BRF.fit(X,y)

        dump(BRF, output.model)

        importances = BRF.feature_importances_

        # Assuming importances is a list of feature importances and X_train.columns contains the feature names
        feature_importances = pd.DataFrame(importances, index=X.columns, columns=['importance'])
        feature_importances.sort_values(by='importance', ascending=False, inplace=True)

        feature_importances = feature_importances.reset_index().rename(columns={'index': 'feature'})

        # Save to CSV
        feature_importances.to_csv(output.importance_scores, header=False, index=False)


rule top_80_features:
    input:
        feature_importances = 'results/iteration1/feature_importance_scores.txt',
    output:
        top80 = 'results/iteration1/top_80_features.txt',
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
        top80 = 'results/iteration1/top_80_features.txt',
    output:
        first_iteration_vcf = 'results/iteration1/BackElim.vcf.gz',
    resources:
        time    = 10,
        mem_mb  = 2000,
    shell:
        '''
            bcftools view -Oz -o {output.first_iteration_vcf} --include ID=@{input.top80} {input.modeling_vcf}

            gatk IndexFeatureFile -I {output.first_iteration_vcf}

            cp results/iteration1 results/temp -r

            mv results/temp results/previous_iteration
        '''
