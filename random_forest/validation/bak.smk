rule all:
    input:
        expand('results/inputs/variants.txt', itr=[str(i) for i in range(1,47)]),
        expand('results/inputs/validation.vcf.gz', itr=[str(i) for i in range(1,47)]),
        expand('results/inputs/validation.vcf.gz.tbi', itr=[str(i) for i in range(1,47)]),
        expand('results/inputs/validation.table', itr=[str(i) for i in range(1,47)]),
        expand('results/inputs/raw_tables/', itr=[str(i) for i in range(1,47)]),
        expand('results/inputs/validation_input.txt', itr=[str(i) for i in range(1,47)]),
        expand('results/f1_score.txt', itr=[str(i) for i in range(1,47)]),
        expand('results/confusion_matrix.png', itr=[str(i) for i in range(1,47)]),

rule get_variants:
    input:
        table = config['training_table'], 
    output:
        variants = 'results/inputs/variants.txt', 
    resources:
        time    = 60,
        mem_mb  = 24000,
        cpus    = 4,
    run:
        infile = open(input.table, 'rt')
        outfile = open(output.variants, 'wt')

        line = infile.readline()

        for line in infile:
            line = line.rstrip()
            split = line.split('\t')
            print(split[0], file=outfile)

rule get_vcf:
    input:
        full_vcf = config['validation_vcf'],
        variants = 'results/inputs/variants.txt', 
    output:
        itr_vcf = 'results/inputs/validation.vcf.gz',
        itr_tbi = 'results/inputs/validation.vcf.gz.tbi' 
    resources:
        time    = 60,
        mem_mb  = 24000,
        cpus    = 4,
    shell:
        '''
            bcftools view -Oz -o {output.itr_vcf} --include ID=@{input.variants} {input.full_vcf}

            gatk IndexFeatureFile -I {output.itr_vcf}
        '''

rule gatk_table:
    input:
        itr_vcf = 'results/inputs/validation.vcf.gz', 
    output:
        table = 'results/inputs/validation.table', 
    resources:
        time    = 60,
        mem_mb  = 24000,
        cpus    = 4,
    shell:
        '''
            gatk VariantsToTable \
                -V {input.itr_vcf} \
                -F ID -F CHROM -F POS -F REF -F ALT -GF GT \
                --split-multi-allelic true \
                -O {output.table}
        '''

checkpoint split_table:
    input:
        table = 'results/inputs/validation.table',
    output:
        split_tables = directory('results/inputs/raw_tables/'),
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
        output_dir = output.split_tables
        lines_per_file = 1000
        split_file_by_lines(input_file, output_dir, lines_per_file)


rule convert_tables:
    input:
        table = 'results/inputs/raw_tables/part_{num}.txt'
    output:
        converted_table = 'results/inputs/converted_tables/part_{num}.txt',
    params:
        phenofile = config['validation_pheno_file'], 
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
    # Assuming 'itr' is the wildcard for the iteration number
    ck_output = checkpoints.split_table.get(itr=wildcards.itr).output['split_tables']
    parts = str(Path(ck_output) / 'part_{num}.txt')
    PARTS, = glob_wildcards(parts)
    return sorted(expand(
        'results/inputs/converted_tables/part_{num}.txt',
        itr=wildcards.itr,
        num=PARTS
    ))

rule combine_tables:
    input:
        tables = get_file_nums,
    output:
        validation_input = 'results/inputs/validation_input.txt',
    params:
        directory = 'results/inputs/converted_tables/',
    resources:
        time    = 120,
        mem_mb  = 24000,
        cpus    = 4,
    run:
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
        combined_df.to_csv(output.validation_input, index=True, header=True, sep='\t')


rule run_forest:
    input:
        forest = '../results/random_forest_model.pkl',
        table = 'results/inputs/validation_input.txt',
       # features = '../X_val.csv',
       # outcomes = '../y_val.csv',
    output:
        f1_score = 'results/f1_score.txt',
        confusion_matrix = 'results/confusion_matrix.png',
    resources:
        time    = 30,
        mem_mb  = 60000,
        cpus    = 32,
    run:
        # Load the saved model
        SMOTE_SRF = load(input.forest)

        outfile = open(output.f1_score, 'wt')

        df = pd.read_csv(input.table, delimiter='\t', index_col=0)

        X = df.drop('y', axis=1)
        y = df['y']

        # Set the threshold
        threshold = 0.35

        # Predict probabilities
        y_probs = SMOTE_SRF.predict_proba(X)
        y_pred = (y_probs[:, 1] >= threshold).astype(int)

        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        average_precision = average_precision_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        print('f1: ' + str(f1), file=outfile)
        print('Precision: ' + str(precision), file=outfile)
        print('Recall: ' + str(recall), file=outfile)
        print('Average Precision: ' + str(average_precision), file=outfile)
        print('AUC: ' + str(roc_auc), file=outfile)
        print('Balanced Accuracy: ' + str(balanced_acc), file=outfile)

        # Calculate TN, FP, FN, TP
        TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

        # Calculate Specificity, PPV, NPV
        specificity = TN / (TN + FP)
        ppv = TP / (TP + FP)
        npv = TN / (TN + FN)

        # Print the results
        print('Specificity: ' + str(specificity), file=outfile)
        print('PPV: ' + str(ppv), file=outfile)
        print('NPV: ' + str(npv), file=outfile)

        # Plot confusion matrix with numbers
        fig = confusion_matrix(y_pred, y)
        plt.imshow(fig, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()

        classes = np.unique(y)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        # Add numbers in the boxes
        thresh = fig.max() / 2.
        for i in range(fig.shape[0]):
            for j in range(fig.shape[1]):
                plt.text(j, i, format(fig[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if fig[i, j] > thresh else "black")

        # Save the plot as a PNG file
        plt.savefig(output.confusion_matrix, format='png')
