# %% [markdown]
#
# ### Credit Card  Fraud Detection Lab!
#
# #### supporting functions
#

# %%

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy import interp
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
get_ipython().magic(u'matplotlib inline')


# classifiers
seed = 42

# convert time string to float value
def string_to_timestamp(date_string):
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


# plot roc curve
def plot_roc_curve( classifiers, data, cv, title = 'ROC Curve',
                    filename = 'Plot.png', OnlySMOTEd = True ):

    fig = plt.figure(figsize=(8, 8))

    # for specific classifier & sampler
    for classifier in classifiers:

        # Settings
        x = data[0]
        y = data[1]
        clf = classifier[1]

        ## Plot SMOTE'd data
        name = '{}-SMOTE'.format(classifier[0])
        i = 1 #i-th fold cross validation

        mean_tpr = 0.0  # initial dummy true positive rate
        mean_fpr = np.linspace(0, 1, 100)  # initial dummy false positive rate

        # for n-folds cross validations
        for train, test in cv.split(x, y):
            print('%s - cv #%i' % (name, i))
            x_res, y_res = SMOTE(n_jobs = -1, random_state= seed).fit_resample(x[train], y[train]) # SMOTE'd data
            y_pred = clf.fit(x_res, y_res).predict(x[test]) # train/fit the model & predict y_ using x[test]
            # Compute ROC curve and auc
            fpr, tpr, thresholds = roc_curve(y[test], y_pred)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            i += 1
        # calculate the means for n-folds cv's
        mean_tpr /= cv.get_n_splits(x[train], y[train])
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print('{} (auc = %f)\n'.format(name) % (mean_auc))
        # plot ROC curve
        plt.plot(mean_fpr, mean_tpr, linestyle='--', lw=2,
                 label='{} (area = %0.2f)'.format(name) % mean_auc)
        
        ## Plot non-SMOTE'd data (if necessary)
        if OnlySMOTEd == False:
            name = '{}-Standard'.format(classifier[0])
            i = 1
            mean_tpr = 0.0  # initial dummy true positive rate
            mean_fpr = np.linspace(0, 1, 100)  # initial dummy false positive rate

            # for n-folds cross validations
            for train, test in cv.split(x, y):
                print('%s - cv #%i' % (name, i))
                y_pred = clf.fit(x[train], y[train]).predict(x[test]) # train/fit the model & predict y_ using x[test]
                # Compute ROC curve and auc
                fpr, tpr, thresholds = roc_curve(y[test], y_pred)
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                roc_auc = auc(fpr, tpr)
                i += 1
            # calculate the means for n-folds cv's
            mean_tpr /= cv.get_n_splits(x[train], y[train])
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            print('{} (auc = %f)\n'.format(name) % (mean_auc))
            # plot ROC curve
            plt.plot(mean_fpr, mean_tpr, linestyle='--', lw=2,
                    label='{} (area = %0.2f)'.format(name) % mean_auc)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)


