# %% [markdown]
#
# # Credit Card  Fraud Detection Lab!
# 

# %%

get_ipython().magic(u'matplotlib inline')
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize
from scipy import interp
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import (StratifiedKFold, ShuffleSplit, 
                                     cross_val_score, cross_val_predict)
from collections import Counter #for comparing shape of SMOTE'd vs non-SMOTE'd data
import pydotplus #for making tree diagram
from IPython.display import Image

sns.set(style = 'whitegrid')

# classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier
seed = 42 # assign random state nr

from fraud_detection_functions import * #load functions

from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings

# %% [markdown]
# 
# ## 1. Data preparation
#
# %%

# load dataset
src = './data/credit_card_fraud_data.csv'
df = pd.read_csv(src, header=0, index_col=0)

# data inspection
df.info()
#df.sample(n=5)

# %%

## data cleaning

# drop 'Refused' in *simple_journals* since it's uncertain about fraud
df = df[-df.simple_journal.str.contains('Refused')]
# drop 'emailNA' in *mail_id* (Email address)
df = df[-df.mail_id.str.contains('NA')]
# overwrite cvcresponse = 3 if it was not 0, 1, 2. *cvcresponsecode*: 0=Unknown, 1=Match, 2=No Match, 3-6=Not checked
df.loc[df.cvcresponsecode >= 3, 'cvcresponsecode'] = 3

# remove prefix for id's
df.mail_id = [float(xx.strip('email')) for xx in df.mail_id]
df.ip_id   = [float(xx.strip('ip')) for xx in df.ip_id]
df.card_id = [float(xx.strip('card')) for xx in df.card_id]

# modify datetime/timestamps
df.bookingdate = [string_to_timestamp(i) for i in df.bookingdate]
df.creationdate = [string_to_timestamp(i) for i in df.creationdate]

# convert amount to USD
df.currencycode.unique()  # currency: 'MXN', 'AUD', 'NZD', 'GBP', 'SEK'
conversionrate = [
    ['MXN', 0.052],
    ['AUD', 0.70],
    ['NZD', 0.66],
    ['GBP', 1.30],
    ['SEK', 0.10]
]  # conversion rate to USD

for code, rate in conversionrate:
    df.loc[df.currencycode == code, 'amount'] *= rate  # amount multiplied by rate
    #print('country: %s, rate = %0.2f' % (code, rate))
#df[['currencycode', 'amount']].sample(5)  # check

## modify categorial features to label integers
columns_cat = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
               'shopperinteraction', 'cardverificationcodesupplied', 'accountcode']

# na and missing values are treated as individual labelled category
df[columns_cat] = df[columns_cat].astype(str)
df[columns_cat] = df[columns_cat].fillna('na')
df[columns_cat] = df[columns_cat].replace(np.nan, 'nan')

df[columns_cat] = df[columns_cat].apply(
    lambda col: LabelEncoder().fit_transform(col))

## assign classification label to *simple_journal*
#  0='Chargeback' (fraud), 1='Settled' (benign)
df.simple_journal = label_binarize(['Chargeback'], classes = df.simple_journal).transpose()

df.reset_index(inplace = True)  # Reseting index due to drops

# check dataset
df.info()
#df.sample(5)

## Features & Labels
y = df.simple_journal.values  # labels
x = df.drop(columns = ['txid', 'simple_journal', 'bookingdate']).values  # features
feature_names = df.drop(columns = ['txid', 'simple_journal', 'bookingdate']).columns.values

# Resampling using SMOTE
#x_res, y_res = SMOTE(random_state = seed).fit_resample(x, y)  # get SMOTEd data
#Counter(y), Counter(y_res)  # check resampled results

# %% [markdown]
# 
# ## 1. Visualisation task
# Load the fraud data into your favorite analysis platform (R, Matlab, Python, Weka, KNIME, ...) and make a visualization showing an interesting relationship in the data when comparing the fraudulent from the benign credit card transactions. You may use any visualization method such as a Heat map, a Scatter plot, a Bar chart, a set of Box plots, etc. as long as they show all data points in one figure. What feature(s) and relation to show is entirely up to you. Describe the plot briefly. 
#
# %%

'''
##Task 1
##Visualisation
'''

## Correlation heatmap
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure('heatmap', figsize=(10, 8))
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            mask=mask, square=True)
plt.title('Correlation heatmap')
plt.savefig('Heatmap corr fraud.png')

## Heatmap, number of frauds, per country & amount of transactions

df_fraud = df.loc[df.simple_journal == 1]  # subset only fraud transactions
df_fraud_p = pd.crosstab(df_fraud.amount, df_fraud.issuercountrycode)

# plot heatmap
plt.figure(1, figsize=(8, 8))
sns.heatmap(df_fraud_p, cmap="YlGnBu", yticklabels = 20)
plt.title('Number of credit card frauds')
plt.ylabel('Amount of transaction')
plt.xlabel('Country where the card was issued')
plt.savefig('Heatmap_CountryVsAmount.png')

## plot stripplot
plt.figure(2, figsize=(8, 4))
sns.stripplot(data=df, x='issuercountrycode', y='amount', hue='simple_journal', hue_order=[0, 1],
              alpha=.4, palette='Set2', dodge=True, linewidth=.1)
plt.xticks([])
plt.legend(labels=('Benign', 'Fraud'), title='Label')
plt.savefig('Scattermap_CountryVsAmount_all.png')

# plot distplot
# plt.figure(3)
# sns.distplot(df.issuercountrycode)
# plt.savefig('Distplot_issuercountry.png')

# %% [markdown]
# 
# ## 3. Imbalanced task
# Process the data such that you can apply SMOTE to it. SMOTE is included in most analysis platforms, if not you can write the scripts for it yourself. Analyze the performance of at least three classifiers on the SMOTEd and UNSMOTEd data using ROC analysis. Provide the obtained ROC curves and explain which method performsbest. Is using SMOTE a good idea? Why (not)? 
# 
# %%

'''
##Task 2
##Imbalanced data: standard vs SMOTE
'''

# settings
cv = StratifiedKFold(n_splits = 5, random_state = seed)
classifiers = [
    ['LogisticRegression', LogisticRegression(random_state = seed, solver = 'lbfgs', n_jobs = -1)],
    ['NN-MLP', MLPClassifier(random_state = seed, n_jobs = -1)],
    ['DecisionTree', DecisionTreeClassifier(random_state = seed, n_jobs = -1)],
    ['RandomForest', RandomForestClassifier(n_estimators = 10, random_state = seed, n_jobs = -1)]
]
data = [x, y]

# plot roc curve
plot_roc_curve(classifiers, data, cv, 
               filename = 'ROC Imbalanced data.png', OnlySMOTEd = False)

# %% [markdown]
# 
# ## 4. Classification task
#
# Build two classifiers for the fraud detection data as well as you can manage: 
#
# * A black-box algorithm, ignoring its inner workings: it is the performance that counts. 
# * A white-box algorithm, making sure that we can explain why a transaction is labeled as being fraudulent. 
#
# Explain the applied data pre-processing steps, learning algorithms, and post-processing steps or ensemble methods. Compare the performance of the two algorithms, focusing on performance criteria that are relevant in practice, use 10-fold cross-validation. Write clear code/scripts for this task, for peer-review your fellow students will be asked to run and understand your code! 
# 
# %%

'''
##Task 3-1
##Classification : Blackbox
'''

# settings
rs = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = seed)
classifiers = [
    ['NN-MLP', MLPClassifier(hidden_layer_sizes = 100, random_state = seed)],
    ['RandomForest', RandomForestClassifier(n_estimators = 10, random_state = seed, n_jobs = -1)],
    ['3NN', KNeighborsClassifier(n_neighbors = 5, weights = 'distance', n_jobs = -1)]
]
data = [x, y]

# plot roc curve for blackbox algoritms
plot_roc_curve(classifiers, data, rs, title = 'ROC curve: blackbox classifiers',
               filename = 'ROC curve Blackbox.png')

## comparison for random forests
classifiers = [
    ['RandomForest (n = 5)',  RandomForestClassifier(n_estimators = 5,  random_state = seed, n_jobs = -1)],
    ['RandomForest (n = 10)', RandomForestClassifier(n_estimators = 10, random_state = seed, n_jobs = -1)],
    ['RandomForest (n = 15)', RandomForestClassifier(n_estimators = 15, random_state = seed, n_jobs = -1)],
    ['RandomForest (n = 10, max_d = 2)', RandomForestClassifier(n_estimators = 10, max_depth = 2, random_state = seed, n_jobs = -1)],
    ['RandomForest (n = 10, max_d = 5)', RandomForestClassifier(n_estimators = 10, max_depth = 5, random_state = seed, n_jobs = -1)]
]

# plot roc curve for random forest with few parameters adjusted
plot_roc_curve(classifiers, data, rs, 
               filename = 'ROC curve Blackbox RandomForest.png')


## plot roc curve for random forest n = 15 per shuffle
clf = RandomForestClassifier(n_estimators = 15, random_state = seed, n_jobs = -1)
i = 1
confusionmx = [[0, 0], [0, 0]]
for train, test in rs.split(x, y):
    x_res, y_res = SMOTE(random_state = seed, n_jobs = -1).fit_resample(x[train], y[train])
    y_pred = clf.fit(x_res, y_res).predict(x[test])
    fpr, tpr, thresholds = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle='--', lw=2, label='%i (area = %0.2f)' % (i, roc_auc))
    confusionmx += confusion_matrix(y[test], y_pred)
    print('RandomForest split %i: area = %0.2f' % (i, roc_auc))
    i += 1
print('confusion matrix:\n{}\n'.format(confusionmx))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: RandomForest (n = 15) per 10 shuffles')
plt.legend(loc="lower right")
plt.savefig('ROC curve Blackbox RandomForest per shuffle.png')

# %%

'''
##Task 3-2
##Classification : Whitebox
'''

# settings
rs = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = seed)
classifiers = [
    ['DecisionTree', DecisionTreeClassifier(random_state = seed)],
    ['LogisticRegression', LogisticRegression(random_state = seed, n_jobs = -1, solver = 'lbfgs')],
    ['Lasso', Lasso(random_state = seed)],
    ['SGDClassifier', SGDClassifier(random_state = seed, n_jobs = -1)]
]
data = [x, y]

# plot roc curve for whitebox algoritms
plot_roc_curve(classifiers, data, rs, title = 'ROC curve: whitebox classifiers',
               filename = 'ROC curve Whitebox.png')

## plot roc curve for decision tree per shuffle
clf = DecisionTreeClassifier(random_state = seed)
i = 1
confusionmx = [[0, 0], [0, 0]]
for train, test in rs.split(x, y):
    x_res, y_res = SMOTE(random_state = seed, n_jobs = -1).fit_resample(x[train], y[train])
    y_pred = clf.fit(x_res, y_res).predict(x[test])
    fpr, tpr, thresholds = roc_curve(y[test], y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle='--', lw=2, label='split %i (area = %0.2f)' % (i, roc_auc))
    confusionmx += confusion_matrix(y[test], y_pred)
    print('DecisionTree split %i: area = %0.2f' % (i, roc_auc))
    i += 1
print('confusion matrix:\n{}\n'.format(confusionmx))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: DecisionTree per 10 shuffles')
plt.legend(loc="lower right")
plt.savefig('ROC curve Whitebox DecisionTree per shuffle.png')

## Plot decision tree
dot_data = export_graphviz(clf, rounded = True, filled = True, 
                           feature_names = feature_names, 
                           class_names = ['Benign', 'Fraud'])

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) # display in python
graph.write_png("DecisionTreeViz.png")
