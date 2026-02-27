import joblib                   
import pandas as pd             
import numpy as np              
import matplotlib
matplotlib.use("Agg")  # headless backend (writes PNGs, no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix,
)

rule all:
    input:
        expand('results/f1_score.txt', itr=[str(i) for i in range(1,47)]),
        expand('results/confusion_matrix.png', itr=[str(i) for i in range(1,47)]),

rule run_forest:
    input:
        forest = config['random_forest'],
        table = config['validation_table'],
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
        forest = joblib.load(input.forest)

        outfile = open(output.f1_score, 'wt')

        df = pd.read_csv(input.table, delimiter='\t', index_col=0)

        X = df.drop('y', axis=1)
        y = df['y']

        # Set the threshold
        threshold = 0.35

        # Predict probabilities
        y_probs = forest.predict_proba(X)
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
