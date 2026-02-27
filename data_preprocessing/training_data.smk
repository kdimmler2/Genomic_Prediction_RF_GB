import os
from pathlib import Path

import numpy as np
import pandas as pd

rule all:
    input:
        'training_data/modeling.table',
        'training_data/raw_tables/',
        'training_data/modeling_input.txt',

rule gatk_table:
    input:
        training_vcf = config['training_vcf'], 
    output:
        table = 'training_data/modeling.table', 
    resources:
        time    = 60,
        mem_mb  = 6000,
    shell:
        '''
            # Convert VCF to a flat table for downstream modeling.
            # Extract core variant fields + genotype (GT) and
            # split multi-allelic sites to ensure one row per allele,
            # simplifying feature matrix construction. 

            gatk VariantsToTable \
                -V {input.training_vcf} \
                -F ID -F CHROM -F POS -F REF -F ALT -GF GT \
                --split-multi-allelic true \
                -O {output.table}
        '''

checkpoint split_table:
    input:
        table = 'training_data/modeling.table',
    output:
        split_tables = directory('training_data/raw_tables/'),
    resources:
        time    = 10,
        mem_mb  = 12000,
    run:

        # Split the large modeling table into fixed-size chunks to enable
        # parallel downstream processing and keep per-job memory/IO manageable.
        # Each chunk includes the header so it can be processed independently.

        def split_file_by_lines(input_file, output_dir, lines_per_file):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(input_file, 'r') as f:
                header = next(f)
                lines_written = 0
                file_num = 1

                # Use a simple numbered naming convention so downstream rules can glob parts.
                output_file = output_dir + '/part_' + str(file_num) + '.txt'
                with open(output_file, 'w') as out:
                    out.write(header)

                    for line in f:
                        out.write(line)
                        lines_written += 1

                        # Rotate to a new chunk after N data lines (header not counted).
                        if lines_written >= lines_per_file:
                            lines_written = 0
                            file_num += 1
                            output_file = output_dir + '/part_' + str(file_num) + '.txt'
                            out.close()
                            out = open(output_file, 'w')
                            out.write(header)

        input_file = input.table
        output_dir = 'training_data/raw_tables'
        lines_per_file = 1000  # tuned for small, fast chunks (adjust based on table size / IO)
        split_file_by_lines(input_file, output_dir, lines_per_file)


rule convert_tables:
    input:
        table = 'training_data/raw_tables/part_{num}.txt'
    output:
        converted_table = 'training_data/converted_tables/part_{num}.txt',
    params:
        phenofile = config['training_pheno_file'], 
    resources:
        time    = 30,
        mem_mb  = 12000,
        cpus    = 4,
    run:
        # Convert a GATK VariantsToTable export (per-variant rows, per-sample GT columns)
        # into a modeling-ready matrix:
        #   - rows = samples
        #   - columns = variants (0/1/2 dosage encoding)
        #   - add outcome (y) and covariate (sex) from an external phenotype file
        #
        # Assumptions:
        #   - The input table includes REF and ALT rows (see indices used below)
        #   - Genotypes are strings like A/A, A|A, A/G, etc. Missing/other formats are handled
        #     with conservative defaults (see genotype mapping section)

        df = pd.read_csv(input.table, delimiter='\t')
        df.set_index('ID', inplace=True)

        # Transpose so that samples become rows and variants become columns
        df2 = df.transpose()

        # Encode genotype strings as allele dosage (0,1,2) relative to REF/ALT.
        # This produces a simple additive encoding suitable for many classifiers.
        for c in range(len(df2.columns)):
            for r in range(4, len(df2)):
                ref = df2.iat[2, c]
                alt = df2.iat[3, c]
                gt = df2.iat[r, c]

                # Handle phased/unphased representations. Treat any ALT-containing call
                # that doesn't match the strict patterns as heterozygous (dosage=1).
                if gt == ref + '/' + ref or gt == ref + '|' + ref:
                    df2.iat[r, c] = 0
                elif gt == ref + '/' + alt or gt == ref + '|' + alt:
                    df2.iat[r, c] = 1
                elif gt == alt + '/' + alt or gt == alt + '|' + alt:
                    df2.iat[r, c] = 2
                elif alt in str(gt):
                    df2.iat[r, c] = 1
                else:
                    # Default for unexpected/missing patterns: treat as REF/REF
                    df2.iat[r, c] = 0

        # Drop non-genotype rows (CHROM, POS, REF, ALT) now that dosage encoding is done
        df3 = df2[4:]

        # Build final design matrix: explicit sample column + variant IDs as headers
        df4 = df3.copy()
        df4.reset_index(inplace=True)
        df4.rename(columns={'index': 'sample'}, inplace=True)

        # Normalize sample IDs (remove .GT suffix) to match the phenotype file keys
        for r in range(len(df4)):
            split = df4.at[r, 'sample'].split('.')
            df4.at[r, 'sample'] = split[0]

        # Add placeholder columns for labels/covariates before populating from file
        df4.insert(1, 'phenos', 5)

        # Load phenotype and sex metadata produced upstream (get_phenos.py)
        infile = open(params.phenofile, 'rt')

        # Map sample_id -> phenotype / sex for fast lookup and consistent joins
        phenos = {}
        sexes = {}
        for line in infile:
            line = line.rstrip()
            split = line.split(' ')
            phenos[split[1]] = split[0]
            sexes[split[1]] = split[2]

        df5 = df4
        df5.insert(2, 'sex', 5)

        # Attach phenotype + sex to each sample row (only replace if sample is present)
        for r in range(len(df5)):
            sid = df5.at[r, 'sample']
            if sid in phenos:
                df5.at[r, 'phenos'] = int(phenos[sid])
            if sid in sexes:
                df5.at[r, 'sex'] = int(sexes[sid])

        # Final cleanup: rename phenotype column to y and set sample as index
        df5.rename(columns={'phenos': 'y'}, inplace=True)
        df5.set_index('sample', inplace=True)

        # Write modeling matrix as tab-delimited text for downstream sklearn steps
        df5.to_csv(output.converted_table, header=True, index=True, sep='\t')

def get_file_nums(wildcards):
    # Resolve the split_table checkpoint to dynamically determine how many
    # chunked files were produced. Since the number of parts depends on
    # input table size, we cannot know the downstream targets ahead of time.

    ck_output = checkpoints.split_table.get(**wildcards).output[0]

    # Discover all generated part_{num}.txt files from the checkpoint directory
    parts = str(Path(ck_output) / 'part_{num}.txt')
    PARTS, = glob_wildcards(parts)

    # Return corresponding converted output targets for each discovered part
    return sorted(expand(
        'training_data/converted_tables/part_{num}.txt',
        num=PARTS
    ))
 
rule combine_tables:
    input:
        tables = get_file_nums,
    output:
        modeling_input = 'training_data/modeling_input.txt',
    params:
        directory = 'training_data/converted_tables/',
    resources:
        time    = 120,
        mem_mb  = 12000,
    run:
        # Combine per-chunk, modeling-ready tables back into a single design matrix.
        # Each chunk file contains:
        #   - metadata/labels columns in the first 5 fields (e.g., sample, y, sex, ...)
        #   - genotype feature columns after that
        #
        # We keep the label columns from the first chunk and merge feature columns
        # across chunks by sample index to reconstruct the full feature set.

        files = input.tables

        combined_df = pd.DataFrame()

        for file in files:
            if file.endswith('.txt'):
                df = pd.read_csv(file, sep='\t')

                if combined_df.empty:
                    # Keep label/metadata columns once (assumed consistent across chunks)
                    # Index number here will be 4 + the number of covariates
                    labels = df.iloc[:, :5]
                    combined_df = df.iloc[:, 5:]
                else:
                    # Merge feature columns by sample (outer join preserves samples
                    # even if some chunks are missing rows)
                    # Again, 4 + the number of covariates
                    combined_df = pd.merge(
                        combined_df,
                        df.iloc[:, 5:],
                        left_index=True,
                        right_index=True,
                        how='outer'
                    )

        # Reattach labels/metadata columns to the left of the final feature matrix
        combined_df = pd.concat([labels, combined_df], axis=1)
        combined_df.set_index('sample', inplace=True)

        # Write final design matrix for downstream sklearn steps
        combined_df.to_csv(output.modeling_input, index=True, header=True, sep='\t')
