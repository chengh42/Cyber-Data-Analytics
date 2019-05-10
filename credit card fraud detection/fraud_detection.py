# %% [markdown]
#
# # Credit Card  Fraud Detection Lab!
# 

#%% Change working directory from the workspace root to the .py file location
import os
try:
	os.chdir(os.path.dirname(os.path.realpath('__file__')))
	print(os.getcwd())
except:
	pass

# %%

get_ipython().magic(u'matplotlib inline')
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from scipy import interp
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import (StratifiedKFold, ShuffleSplit, 
                                     cross_val_score, cross_val_predict)
from collections import Counter #for comparing shape of SMOTE'd vs non-SMOTE'd data
import pydotplus #for making tree diagram
from IPython.display import Image

# classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier
seed = 42 # assign random state nr

from fraud_detection_functions import * #load functions

from warnings import simplefilter # import warnings filter
simplefilter(action = 'ignore', category = FutureWarning) # ignore all future warnings

# %% [markdown]
# 
# ## 0. Data preparation ##
# 
# + load data
# + data cleaning
# + aggregating features
#
# %%

# load dataset
src = './data/credit_card_fraud_data.csv'
df = pd.read_csv(src, header=0, index_col=0)

# data inspection
#df.info()
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

## modify datetime/timestamps
df.bookingdate  = [string_to_timestamp(i) for i in df.bookingdate]
df.creationdate = [string_to_timestamp(i) for i in df.creationdate]

# convert amount to USD
df.currencycode.unique()  # currency: 'MXN', 'AUD', 'NZD', 'GBP', 'SEK'
conversionrate = [  ['MXN', 0.052],
                    ['AUD', 0.70],
                    ['NZD', 0.66],
                    ['GBP', 1.30],
                    ['SEK', 0.10]  ]  # conversion rate to USD

for code, rate in conversionrate:
    df.loc[df.currencycode == code, 'amount'] *= rate  # amount multiplied by rate
    #print('country: %s, rate = %0.2f' % (code, rate))
#df[['currencycode', 'amount']].sample(5)  # check

# modify categorial features to label integers
columns_cat = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
               'shopperinteraction', 'cardverificationcodesupplied', 'accountcode']

# na and missing values are treated as individual labelled category
df[columns_cat] = df[columns_cat].astype(str)
df[columns_cat] = df[columns_cat].fillna('na')
df[columns_cat] = df[columns_cat].replace(np.nan, 'nan')

df[columns_cat] = df[columns_cat].apply(
    lambda col: LabelEncoder().fit_transform(col))

# assign classification label to *simple_journal*
#  0='Settled' (benign), 1='Chargeback' (fraud)
df.simple_journal = label_binarize(['Chargeback'], classes = df.simple_journal).transpose()

df.reset_index(inplace = True)  # Reseting index due to drops
df.drop(columns = ['txid'], inplace = True) # drop txid

# check dataset
df.info()
#df.sample(5)

## Features & Labels
y = df.simple_journal.values  # labels
x = df.drop(columns = ['simple_journal', 'bookingdate']).values  # features
feature_names = df.drop(columns = ['simple_journal', 'bookingdate']).columns.values

# Resampling using SMOTE
#x_res, y_res = SMOTE(random_state = seed).fit_resample(x, y)  # get SMOTEd data
#Counter(y), Counter(y_res)  # check resampled results

# %% [markdown]
# 
# ## 1. Aggregate features (bonus task) ##
# 
# Create a new dataframe for only aggregated features of **the past transactions frequency/amount per credit card (card_id)** at
# 
# + the same merchant
# + the same commerce type (online, etc.)
# + the same country where transaction took place
# 

# %%
# Bonus assigment - adding context variables
'''
Note: This subpart takes ~20 minutes to run
The result is saved in .csv file './data/aggregated_data.csv'
'''
# If necessary, load exported data
# df = pd.read_csv('./data/aggregated_data.csv')

# Creating the frequency column that counts how many times the same card appears
df['creditcard_frequency'] = df.groupby('card_id')['card_id'].transform('count')

# Defining default/dummy values for new variables
df['amountSpent'] = 0 # Variables related to the expenditure
df['SpentWithSameMerchant'] = 0 # Variables related to the merchant
df['SpentWithECommerce'] = 0 # E-commerce
df['SpentInTheCountry'] = 0 # Country

#AmountSpent in the past
def amountSpent(creationdate,card_id):
    return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id)][['amount']].sum())

#Average spent per operation
def AVGamountSpent(creationdate,card_id):
    counting = float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id)][['amount']].count())
    if counting > 0:
        return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id)][['amount']].sum())/counting
    else:
        return 0
  
#Amount spent in the past with the same merchant
def SpentWithSameMerchant(creationdate,card_id,accountcode):
    return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["accountcode"] == accountcode)][['amount']].sum())

#Average spent in the past per operation with the same merchant
def AVGSpentWithSameMerchant(creationdate,card_id,accountcode):
    counting = float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["accountcode"] == accountcode)][['amount']].count())
    if counting>0:
        return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["accountcode"] == accountcode)][['amount']].sum())/counting
    else:
        return 0

#Amount spent by type of commerce in the past
def SpentWithECommerce(creationdate,card_id,shopperinteraction):
    return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shopperinteraction"] == shopperinteraction)][['amount']].sum())

#Average spent in the past per operation by type of commerce 
def AVGSpentWithECommerce(creationdate,card_id,shopperinteraction):
    counting=float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shopperinteraction"] == shopperinteraction)][['amount']].count())
    if counting>0:
        return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shopperinteraction"] == shopperinteraction)][['amount']].sum())/counting
    else:
        return 0

#Amount spent in the country previously
def SpentInTheCountry(creationdate,card_id,shoppercountrycode):
    return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shoppercountrycode"] == shoppercountrycode)][['amount']].sum())

#Average spent in the country per operation in the past
def AVGSpentInTheCountry(creationdate,card_id,shoppercountrycode):
    counting=float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shoppercountrycode"] == shoppercountrycode)][['amount']].count())
    if counting>0:
        return float(df[(df["creationdate"] < creationdate) & (df["card_id"] == card_id) & (df["shoppercountrycode"] == shoppercountrycode)][['amount']].sum())/counting
    else:
        return 0

df['amountSpent'] = df.apply(lambda df: amountSpent(df["creationdate"], df["card_id"]) if df["creditcard_frequency"] >1 else 0, axis=1)
df['AVGamountSpent'] = df.apply(lambda df: AVGamountSpent(df["creationdate"], df["card_id"]) if df["creditcard_frequency"] >1 else 0, axis=1)
df['SpentWithSameMerchant'] = df.apply(lambda df: SpentWithSameMerchant(df["creationdate"], df["card_id"], df['accountcode'])if df["amountSpent"] >0 else 0, axis=1)
df['AVGSpentWithSameMerchant'] = df.apply(lambda df: AVGSpentWithSameMerchant(df["creationdate"], df["card_id"], df['accountcode'])if df["amountSpent"] >0 else 0, axis=1)
df['SpentWithECommerce'] = df.apply(lambda df: SpentWithECommerce(df["creationdate"], df["card_id"], df['shopperinteraction'])if df["amountSpent"] >0 else 0, axis=1)
df['AVGSpentWithECommerce'] = df.apply(lambda df: AVGSpentWithECommerce(df["creationdate"], df["card_id"], df['shopperinteraction'])if df["amountSpent"] >0 else 0, axis=1)
df['SpentInTheCountry'] = df.apply(lambda df: SpentInTheCountry(df["creationdate"], df["card_id"], df['shoppercountrycode'])if df["amountSpent"] >0 else 0, axis=1)
df['AVGSpentInTheCountry'] = df.apply(lambda df: AVGSpentInTheCountry(df["creationdate"], df["card_id"], df['shoppercountrycode'])if df["amountSpent"] >0 else 0, axis=1)
  
#Ratio variables - dividing amount of the transaction to previous transactions
df['RatioAmountToAmountSpent']=df.apply(lambda df: df["amountSpent"]/df["amount"] if df["amountSpent"]>0  else 0, axis=1)
df['RatioAmountToAmountSpentMerchant']=df.apply(lambda df: df["SpentWithSameMerchant"]/df["amount"] if df["SpentWithSameMerchant"]>0  else 0, axis=1)
df['RatioAmountToAmountSpentECommerce']=df.apply(lambda df: df["SpentWithECommerce"]/df["amount"] if df["SpentWithECommerce"]>0  else 0, axis=1)
df['RatioAmountToAmountSpentCountry']=df.apply(lambda df: df["SpentInTheCountry"]/df["amount"] if df["SpentInTheCountry"]>0  else 0, axis=1)

df.info() #check dataframe

df.to_csv(r'./data/aggregated_data.csv') #export to csv file

## Features & Labels
y_agg = df.simple_journal.values  # labels
x_agg = df.drop(columns = ['simple_journal', 'bookingdate', 'creditcard_frequency']).values  # features
feature_names_agg = df.drop(columns = ['simple_journal', 'bookingdate', 'creditcard_frequency']).columns.values


# %% [markdown]
# 
# ## 2. Visualisation task ##
# Make a visualization showing an interesting relationship in the data when comparing the fraudulent from the benign credit card transactions. You may use any visualization method such as a Heat map, a Scatter plot, a Bar chart, a set of Box plots, etc. as long as they show all data points in one figure. What feature(s) and relation to show is entirely up to you. Describe the plot briefly. 
#
# %%

'''
## Task 1
## Visualisation
'''

sns.set(style = 'whitegrid')

# get a subset of df which contains 1/4 fraud and 3/4 benign
df_a = df.loc[df.simple_journal == 1]
df_a = pd.concat([df_a, df.loc[df.simple_journal == 0].sample(n = len(df_a)*3)])
df_a.drop(columns = 'creditcard_frequency', inplace = True) #drop creditcard frequency
df_a.sort_values('simple_journal', inplace = True)
df_a.info()

# row colors
lut = dict(zip(df_a.simple_journal.unique(), "gy"))
row_colors = df_a.simple_journal.map(lut)

# plot clustermap
plt.figure(figsize=(15, 15))
sns.clustermap(df_a.drop(columns = ['simple_journal']), row_colors = row_colors,
               standard_scale = 1, row_cluster=False, col_cluster=False, yticklabels = False) 
plt.title('Label (standardised)')
plt.savefig('df_a.png')

# %%

## subset data for only few columns 
##   which seem 'correlated'

## Features & Labels
y_sub = df.simple_journal.values  # labels
x_sub = df[['issuercountrycode', 'amount', 'cvcresponsecode', 'accountcode', 'ip_id']].values  # features
feature_names_sub = ['issuercountrycode', 'amount', 'cvcresponsecode', 'accountcode', 'ip_id']

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

sns.set(style = 'whitegrid')

# settings
cv = StratifiedKFold(n_splits = 5, random_state = seed)
classifiers = [
    ['LogisticRegression', LogisticRegression(random_state = seed, solver = 'lbfgs', n_jobs = -1)],
    ['Lasso', Lasso(random_state = seed)],
    ['DecisionTree', DecisionTreeClassifier(random_state = seed)],
    ['RandomForest', RandomForestClassifier(n_estimators = 10, random_state = seed, n_jobs = -1)]
]
data = [x, y]

# plot roc curve
plot_roc_curve(classifiers, data, cv, title = "ROC curve: SMOTE'd vs non-SMOTE'd",
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

sns.set(style = 'whitegrid')

# settings
rs = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = seed)
list_clf = [
    ('LogisticRegression', LogisticRegression(random_state = seed, solver = 'lbfgs', n_jobs = -1)), 
    ('DecisionTree', DecisionTreeClassifier(random_state = seed))
]
classifiers = [
    ['RandomForest', RandomForestClassifier(n_estimators = 10, random_state = seed, n_jobs = -1)],
    ['3NN', KNeighborsClassifier(n_neighbors = 3, weights = 'distance', n_jobs = -1)],
    ['ExtraTrees', ExtraTreesClassifier(random_state = seed, n_jobs = -1)],
    ['Voting (LR & tree)', VotingClassifier(list_clf, n_jobs = -1)]
]
data = [x, y]
data_sub = [x_sub, y_sub]

# plot roc curve for blackbox algoritms
plot_roc_curve(classifiers, data, rs, title = 'ROC curve: blackbox classifiers',
               filename = 'ROC curve Blackbox.png')

# plot roc curve for df w/ subseted features
plot_roc_curve(classifiers, data_sub, rs, title = 'ROC curve: blackbox classifiers - subseted features',
               filename = 'ROC curve Blackbox sub.png')

# %%

## Run KNN classifiers with different nr of neighbors

classifiers = [
    ['3NN', KNeighborsClassifier(n_neighbors = 3, weights = 'distance', n_jobs = -1)],
    ['5NN', KNeighborsClassifier(n_neighbors = 5, weights = 'distance', n_jobs = -1)],
    ['10NN', KNeighborsClassifier(n_neighbors = 10, weights = 'distance', n_jobs = -1)],
]

# plot roc curve for df w/ subseted features
plot_roc_curve(classifiers, data_sub, rs, title = 'ROC curve: KNN classifiers - subseted features',
               filename = 'ROC curve Blackbox KNN sub.png')

# %%

'''
##Task 3-2
##Classification : Whitebox
'''

sns.set(style = 'whitegrid')

# settings
rs = ShuffleSplit(n_splits = 10, test_size = 0.1, random_state = seed)
classifiers = [
    ['DecisionTree', DecisionTreeClassifier(random_state = seed)],
    ['LogisticRegression', LogisticRegression(random_state = seed, n_jobs = -1, solver = 'lbfgs')],
    ['Lasso', Lasso(random_state = seed)],
    ['SGDClassifier', SGDClassifier(random_state = seed, n_jobs = -1)]
]
data = [x, y]
data_agg = [x_agg, y_agg]

# plot roc curve for whitebox algoritms
plot_roc_curve(classifiers, data, rs, title = 'ROC curve: whitebox classifiers',
               filename = 'ROC curve Whitebox.png')

# plot roc curve for df w/ aggregated features
plot_roc_curve(classifiers, data_agg, rs, title = 'ROC curve: whitekbox classifiers w/ aggregated features',
               filename = 'ROC curve Whitekbox agg.png')

# %%

# Lasso with different alpha's
classifiers = [
    ['Lasso (a = 1)', Lasso(alpha = 1, random_state = seed)], #default
    ['Lasso (a = .5)', Lasso(alpha = .5, random_state = seed)],
    ['Lasso (a = .1)', Lasso(alpha = .1, random_state = seed)],
    ['Lasso (a = .01)', Lasso(alpha = .01, random_state = seed)]
]

# plot roc curve for Lasso algoritms
plot_roc_curve(classifiers, data, rs, title = 'ROC curve: Lasso classifiers',
               filename = 'ROC curve Whitebox Lasso.png')