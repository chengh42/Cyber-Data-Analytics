
# coding: utf-8
#
# # Credit Card  Fraud Detection Lab!
#
# ## Reference
# * Unbalanced data
#     1. Learning from Imbalanced Data, He et al. (**improtant!**)
#     2. Cost-sensitve boosting for classification of imbalanced data, Sun et al.
#     3. SMOTE: Synthetic Minority Over-sampling Technique, Chawla et al.
# * Fraud detection
#     1. Data mining for credit card fraud: A comparative study, Bhattacharyya et al.
#     2. Minority Report in Fraud Detection: Classification of Skewed Data, Phua et al.
#

# %%

import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from scipy import interp
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import Counter

# classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#sns.set(style='whitegrid')

# %%

'''
##data preparation
'''

# load dataset
src = './data/credit_card_fraud_data.csv'
df = pd.read_csv(src, header=0, index_col=0)

# data inspection
df.info()
df.describe()
df.sample(n=5)

# %%


def string_to_timestamp(date_string):  # convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


# data cleaning
# drop 'Refused' in *simple_journals* since it's uncertain about fraud
df = df[-df.simple_journal.str.contains('Refused')]
# drop 'emailNA' in *mail_id* (Email address)
df = df[-df.mail_id.str.contains('NA')]
# overwrite cvcresponse = 3 if it was not 0, 1, 2. *cvcresponsecode*: 0=Unknown, 1=Match, 2=No Match, 3-6=Not checked
df.loc[df.cvcresponsecode >= 3, 'cvcresponsecode'] = 3

# remove prefix for id's
df.mail_id = [float(xx.strip('email')) for xx in df.mail_id]
df.ip_id = [float(xx.strip('ip')) for xx in df.ip_id]
df.card_id = [float(xx.strip('card')) for xx in df.card_id]

# modify categorial features to label integers
# target: *simple_journal*: 0='Chargeback' (fraud), 1='Settled' (benign)
columns_cat = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
               'shopperinteraction', 'simple_journal', 'cardverificationcodesupplied', 'accountcode']

df[columns_cat] = df[columns_cat].astype(str)
df[columns_cat] = df[columns_cat].fillna('na')
df[columns_cat] = df[columns_cat].replace(np.nan, 'nan')

df[columns_cat] = df[columns_cat].apply(
    lambda col: LabelEncoder().fit_transform(col))

# modify datetime/timestamps
df.bookingdate = [string_to_timestamp(i) for i in df.bookingdate]
df.creationdate = [string_to_timestamp(i) for i in df.creationdate]

# check dataset
df.info()
df.sample(5)

## Features & Labels
y = df.simple_journal.values  # labels
x = df.drop(columns=['simple_journal', 'bookingdate']).values  # features

## Resampling using SMOTE
x_res, y_res = SMOTE(random_state=42).fit_resample(x, y)  # get SMOTEd data
Counter(y), Counter(y_res) #check resampled results

# %%

'''
##Task 1
##Visualisation
'''

# Visualisation : Heatmap, number of frauds, per country & amount of transactions

df_fraud = df.loc[df.simple_journal == 0]  # subset only fraud transactions
df_fraud_p = pd.crosstab(df_fraud.amount, df_fraud.issuercountrycode)

# plot heatmap
plt.figure(1, figsize=(8, 8))
sns.heatmap(df_fraud_p, cmap="YlGnBu", yticklabels=20)
plt.title('Number of credit card frauds')
plt.ylabel('Amount of transaction')
plt.xlabel('Country where the card was issued')
plt.savefig('Heatmap_CountryVsAmount.png')

# plot stripplot
plt.figure(2, figsize=(8, 4))
sns.stripplot(data=df, x='issuercountrycode', y='amount', hue='simple_journal', hue_order=[1, 0],
              alpha=.4, palette='Set2', dodge=True, linewidth=.1)
plt.xticks([])
plt.legend(labels=('Benign', 'Fraud'), title='Label')
plt.savefig('Scattermap_CountryVsAmount_all.png')

# plot distplot
# plt.figure(3)
# sns.distplot(df.issuercountrycode)
# plt.savefig('Distplot_issuercountry.png')

# %%

'''
##Task 2
##Imbalanced data: standard vs SMOTE
'''

class DummySampler(object):
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        return self.sample(X, y)


# settings
cv = StratifiedKFold(10)
classifiers = [
    ['NN-MLP', MLPClassifier()],
    ['DecisionTree', DecisionTreeClassifier()],
    ['RandomForest', RandomForestClassifier(n_estimators=10)]
]
samplers = [
    ['Standard', DummySampler()],
    ['SMOTE', SMOTE(random_state=42)]
]
pipelines = [
    ['{}-{}'.format(classifier[0], sampler[0]),
     make_pipeline(sampler[1], classifier[1])]
    for classifier in classifiers
    for sampler in samplers
]

# loop and plot
fig = plt.figure(4, figsize=(8, 8))

# for specific classifier & sampler
for name, pipeline in pipelines:
    # print(name)
    i = 1
    mean_tpr = 0.0  # initial dummy true positive rate
    mean_fpr = np.linspace(0, 1, 100)  # initial dummy false positive rate

    # for n-folds cross validations
    for train, test in cv.split(x, y):
        print('%s - cv #%i' % (name, i))
        clf = pipeline.fit(x[train], y[train])  # train/fit the model
        y_ = clf.predict(x[test])  # predict *y_* using x[test]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], y_)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        i += 1
    # calculate the means for n-folds cv's
    mean_tpr /= cv.get_n_splits(x, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # print output message    
    print('%s (auc = %f)\n' % (name, mean_auc))
    # plot ROC curve
    plt.plot(mean_fpr, mean_tpr, linestyle='--', lw=2,
             label='{} (area = %0.2f)'.format(name) % mean_auc)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('ROC Imbalanced data 4.png')


# %%

'''
##Task 3-1
##Classification : Blackbox
'''

cv = StratifiedKFold(10)
x_res, y_res = SMOTE(random_state=42).fit_resample(x, y)  # get SMOTEd data

Counter(y), Counter(y_res) #check resampled results

models = [
    ['NN-MLP', MLPClassifier()],
    ['RandomForest', RandomForestClassifier(n_estimators=10)]
]

results = []
names = []
for name, model in models:
    cv_results = cross_val_score(
        model, x_res, y_res, cv = cv, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare between classifiers
plt.figure(5, figsize=(6, 6))
plt.boxplot(results, labels = names)
plt.title('Comparison of Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('ROC-AUC Score')
plt.savefig('Comparison Classification -b.png')

# %%

'''
##Task 3-2
##Classification : Whitebox
'''

models = [
    ['DecisionTree', DecisionTreeClassifier()]
]

results = []
names = []
for name, model in models:
    cv_results = cross_val_score(
        model, x_res, y_res, cv = cv, scoring = 'roc_auc')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare between classifiers
fig = plt.figure(5, figsize=(6, 6))
plt.boxplot(results)
plt.title('Comparison of Classification Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('ROC-AUC Score')
plt.set_xticklabels(names)
plt.savefig('Comparison Classification -w.png')
