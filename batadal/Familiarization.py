# %% [markdown]

# # BATADAL 
# 
# The BATtle of the Attack Detection ALgorithms (BATADAL) will objectively compare the performance of algorithms for the detection of cyber attacks in water distribution systems. Participants will contribute an attack detection algorithm for a given water network following a set of rules that determine the exact goal of the algorithms.
# 
# This script is for the **familiarization** task.
#

# %%

import pandas as pd
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool

# %% [markdown]

# ## Training Dataset 1: 
# 
# This dataset was released on November 20 2016, and it was generated from a one-year long simulation. 
# The dataset does not contain any attacks, i.e. all the data pertains to C-Town normal operations. 
# 

# %%

#If necessary, change your working directory
#import os
#os. chdir("C:/Users/pvbia/Documents/GitHub/Cyber-Data-Analytics/batadal")

# Load dataset
dftrain1 = pd.read_csv('./data/BATADAL_dataset03.csv')

# Check dataset
dftrain1.sample(n = 10)
dftrain1.info()

# Modify string date to timestamp
dftrain1.DATETIME = dftrain1.DATETIME.apply(lambda s: pd.to_datetime(s, format = '%d/%m/%y %H'))

# %% [markdown]
#
# ### Visualisation of training dataset #1 per day(s)
# 
# - subset the dataset for one day
# - plot per category of variables:
#   - L_T#:    *Water level of tank # (meter)*
#   - S/F_PU#: *Setting (dmnl) / flowrate (L/s) of pump #*
#   - S/F_V# : *Setting (dmnl) / flowrate (L/s) of valve #*
#   - P_J#:    *Pressure reading of node/junction # (meter)*
# 

# %% plot variables in training dataset 1

plt.style.use('classic')

# assign time scales to plot
time_scale = [24, 72, 168]

## tanks
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'L'
    dftrain1[0:hr].filter(regex = '^L').plot()
    plt.title('Water level of tank # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Tank water level [meter]')

# %%

## pressure nodes
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'P'
    dftrain1[0:hr].filter(regex = '^P').plot()
    plt.title('Pressure readings at junction # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Pressure [meter]')

# %%

## flowrate of pumps and valves
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'F'
    dftrain1[0:hr].filter(regex = '^F').plot()
    plt.title('Flowrate at pump/valve # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Flowrate [L/s]')

# %%

## setting of pumps and valves
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'S'
    dftrain1[0:hr].filter(regex = '^S').plot()
    plt.title('Setting at pump/valve # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Setting (dmnl)')

# %% [markdown]
# 
# ### Correlated behaviour
# 

# %% correlation matrix

corr = np.corrcoef(dftrain1.drop(columns = ['DATETIME', 'ATT_FLAG']))

np.correlate(dftrain1.L_T3.values, dftrain1.L_T1.values)
plt.xcorr(dftrain1.L_T1.values, dftrain1.L_T3.values)

corr = dftrain1.drop(columns = ['DATETIME', 'ATT_FLAG']).corr(method = 'spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths=.5, cbar_kws={"shrink": .5})


# %%

fig = plt.figure()
fx = fig.add_subplot(311, ylabel = 'water level (m)', 
                     title = 'L_T1 vs L_T3')
fx.plot(dftrain1.loc[100:172, 'L_T1'])
sx = fig.add_subplot(312, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'water level (m)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain1.loc[100:172, 'L_T3'])
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels


# %%

## plot for valve 2: setting vs flowrate

fig = plt.figure()
fx = fig.add_subplot(211, ylabel = 'Flowrate (L/s)', 
                     title = 'Valve 2: correlation between setting (S_V2) and flowrate (F_V2)')
fx.plot(dftrain1.loc[0:72, 'F_V2'])
sx = fig.add_subplot(614, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'Setting (dmnl)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain1.loc[0:72, 'S_V2'])
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels
fig.savefig('.\image\comparison\compare-v2.png',
            bbox_inches = 'tight', facecolor = 'w')

# %%

## plot for pump 7: setting vs flowrate

fig = plt.figure()
fx = fig.add_subplot(211, ylabel = 'Flowrate (L/s)', 
                     title = 'Pump 7: correlation between setting (S_PU7) and flowrate (F_PU7)')
fx.plot(dftrain1.loc[0:72, 'F_PU7'])
sx = fig.add_subplot(614, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'Setting (dmnl)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain1.loc[0:72, 'S_PU7'])
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels
fig.savefig('.\image\comparison\compare-pu7.png',
            bbox_inches = 'tight', facecolor = 'w')

# %%

## plot for tank 3: vs pump 4 (setting & flowrate)

fig = plt.figure()
tx = fig.add_subplot(311, ylabel = 'Water level \n(meter)', 
                     title = 'Tank 3: correlation with Pump 4 (S_PU4 & F_PU4)')
tx.plot(dftrain1.loc[0:72, 'L_T3'])
fx = fig.add_subplot(312, ylabel = 'Flowrate \n(L/s)')
fx.plot(dftrain1.loc[0:72, 'F_PU4'])
sx = fig.add_subplot(615, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'Setting \n(dmnl)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain1.loc[0:72, 'S_PU4'])
plt.setp(tx.get_xticklabels(), visible = False) #hide tx's xlabel
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels
fig.savefig('.\image\comparison\compare-lt3.png',
            bbox_inches = 'tight', facecolor = 'w')
# %%

## plot for tank 4: vs pump 7 (setting & flowrate)

fig = plt.figure()
tx = fig.add_subplot(311, ylabel = 'Water level \n(meter)', 
                     title = 'Tank 4: correlation with Pump 7')
tx.plot(dftrain1.loc[0:72, 'L_T4'])
fx = fig.add_subplot(312, ylabel = 'Flowrate \n(L/s)')
fx.plot(dftrain1.loc[0:72, 'F_PU7'])
sx = fig.add_subplot(615, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'Setting \n(dmnl)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain1.loc[0:72, 'S_PU7'])
plt.setp(tx.get_xticklabels(), visible = False) #hide tx's xlabel
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels
fig.savefig('.\image\comparison\compare-lt4.png',
            bbox_inches = 'tight', facecolor = 'w')


# %% [markdown]
#
# ### Visualisation of autocorrelations
# 
# - subset the dataset for one day
# - plot per category of variables:
#   - L_T#:    *Water level of tank # (meter)*
#   - S/F_PU#: *Setting (dmnl) / flowrate (L/s) of pump #*
#   - S/F_V# : *Setting (dmnl) / flowrate (L/s) of valve #*
#   - P_J#:    *Pressure reading of node/junction # (meter)*
# 

# %%

# plot autocoreelation of time series
autocorrelation_plot(dftrain1.loc[0:72, 'L_T4'])
plt.title('Autocorrelation of L_T4 (in a period of 3 days)')
plt.savefig('.\image\comparison\Autocorrelation-lt4.png',
            bbox_inches = 'tight', facecolor = 'w') #lag of 5 seems reasonable

# %%

# ARIMA
model = ARIMA(dftrain1.loc[0:72, 'L_T4'], order = (5, 1, 0))

model_fit = model.fit(disp = 0) #turn off debug information
print(model_fit.summary())

# residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot() #ARMA fit residual error line

# %% [markdown]

# ## Rolling Forecast ARIMA Model

'''
ARIMA, order = (p, d, q)
p: AR, lag value for autoregression
d: difference,  order of 1 to make the time series stationary
q: MA
'''

# %%

X = dftrain1['L_T4'].values
size = int(len(X) * 0.66) # take 2/3 of the dataset for training
train, test = X[0:size], X[size:len(X)]

history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order = (5, 1, 0))
    model_fit = model.fit(disp = 0) #discp=0: turn off debug information
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('predicted = %0.2f, expected = %0.2f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('test MSE: %0.3f' % error)

# plot the results
plt.plot(test[0:72])
plt.plot(predictions[0:72], color = 'red')
plt.title('Water level of tank 4: real vs ARIMA rolling forecasted')
plt.legend(('real', 'ARIMA, order = (5, 1, 0)'))
plt.savefig('.\image\comparison\ARIMA-lt4.png',
            bbox_inches = 'tight', facecolor = 'w')

# %% [markdown]

# ## Training Dataset 2: 
# 
# This dataset with partially labeled data was released on November 28 2016.
# The dataset is around 6 months long and contains several attacks, some of which are approximately labeled. 
# 

# %%

# Load dataset
dftrain2 = pd.read_csv('./data/BATADAL_dataset04.csv')

# Check dataset
dftrain2.sample(n = 10)
dftrain2.info()

# Remove the spaces in column names
dftrain2.columns = dftrain2.columns.str.strip()

# Modify string date to timestamp
dftrain2.DATETIME = dftrain2.DATETIME.apply(lambda s: string_to_timestamp(s))

# %% [markdown]
# 
# ### Visualisation of correlation between variables
# 
# %%

fig = plt.figure()
tx = fig.add_subplot(311, ylabel = 'Water level \n(meter)', 
                     title = 'Tank 4: correlation with Pump 7 (training dataset 2)')
tx.plot(dftrain2.loc[0:72, 'L_T4'])
fx = fig.add_subplot(312, ylabel = 'Flowrate \n(L/s)')
fx.plot(dftrain2.loc[0:72, 'F_PU7'])
sx = fig.add_subplot(615, sharex = fx, xlabel = 'Hour', 
                     ylabel = 'Setting \n(dmnl)', ymargin=.5, yticks = [0, 1])
sx.plot(dftrain2.loc[0:72, 'S_PU7'])
plt.setp(tx.get_xticklabels(), visible = False) #hide tx's xlabel
plt.setp(fx.get_xticklabels(), visible = False) #hide fx's xlabel
fig.align_ylabels() #align y labels
fig.savefig('.\image\comparison\compare-lt4-df2.png',
            bbox_inches = 'tight', facecolor = 'w')

# %% [markdown]
#
# ### Visualisation of training dataset #2 per day(s)
# 
# - subset the dataset for one day
# - plot per category of variables:
#   - L_T#:    *Water level of tank # (meter)*
#   - S/F_PU#: *Setting (dmnl) / flowrate (L/s) of pump #*
#   - S/F_V# : *Setting (dmnl) / flowrate (L/s) of valve #*
#   - P_J#:    *Pressure reading of node/junction # (meter)*
# 

# %% plot variables in training dataset 2

plt.style.use('classic')

# assign time scales to plot
time_scale = [24, 72, 168]

## tanks
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'L'
    dftrain2[0:hr].filter(regex = '^L').plot()
    plt.title('Water level of tank # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.25, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Tank water level [meter]')


# %%

## pressure nodes
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'P'
    dftrain2[0:hr].filter(regex = '^P').plot()
    plt.title('Pressure readings at junction # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Pressure [meter]')


# %%

## flowrate of pumps and valves
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'F'
    dftrain2[0:hr].filter(regex = '^F').plot()
    plt.title('Flowrate at pump/valve # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Flowrate [L/s]')


# %%

## setting of pumps and valves
for hr in time_scale:
    # Subset the columns for given hrs & name starting with 'S'
    dftrain2[0:hr].filter(regex = '^S').plot()
    plt.title('Setting at pump/valve # in %i hrs (%i days)' % (hr, hr/24))
    plt.legend(loc = 'right', bbox_to_anchor=(1.3, 0.5))
    plt.xlabel('Hour')
    plt.ylabel('Setting (dmnl)')

# %% [markdown]

# ## Test Dataset: 
# 
# This 3-months long dataset contains several attacks but no labels.
# The dataset was released on February 20 2017, and it is used to compare the performance of the algorithms (see rules document for details). 
# 

# %%

# Load datset
dftest = pd.read_csv('./data/BATADAL_test_dataset.csv')

# Check dataset
dftest.sample(n = 15)
dftest.info()

dftest.S_PU1.unique()

# %% [markdown]

# ## Training Datasets: Normal vs Under Attack
# 
# Compare the behaviour of water level of a tank when
# - Under attack: flagged data taken from training dataset 2
# - Normal : data (of the same date and time, different year) taken from training dataset 1
# 

# %%

## Subset the data under attack from training datset 2
dftrain2_att = dftrain2.loc[dftrain2.ATT_FLAG == 1]
dftrain1_nor = dftrain1

# Check dataset
dftrain2_att.info()
print(len(dftrain2_att)) #data reduced from 2089 rows to 219 rows

# %%

# assign sequential, integer (from 1) labels to (un)flagged data sequence
partitions = ((dftrain2.ATT_FLAG == 1) != (dftrain2.ATT_FLAG == 1).shift()).cumsum()

# get a copy of training dataset 1, modify the datetime to %m-%d %H format for comparison 
cpdftrain1 = dftrain1.copy()
cpdftrain1.loc[:, 'DATETIME'] = dftrain1.DATETIME.apply(lambda t: timestamp_to_date(t))

# assign the variable in interest to plot
L_var = 'L_T1'

# %%

# per flagged data sequence, make plots
for groupnr, groupdf in dftrain2[dftrain2.ATT_FLAG == 1].groupby(partitions):
    # modify datetime to %m-%d %H format
    modifyDATETIME = groupdf.DATETIME.apply(lambda t: timestamp_to_date(t))
    groupdf.loc[:, 'DATETIME'] = modifyDATETIME.to_numpy()
    start_time = groupdf.loc[partitions == groupnr, 'DATETIME'].values[0]
    end_time = groupdf.loc[partitions == groupnr, 'DATETIME'].values[-1]
    # merge the data of same datetime (in different year though) from training datset #1
    comparedf = pd.merge(groupdf[['DATETIME', L_var]],
                         cpdftrain1[['DATETIME', L_var]],
                         how = 'left', on = 'DATETIME', suffixes=('_att', '_normal'))
    # plot
    plt.plot(comparedf.drop(columns = 'DATETIME'))
    plt.title('Water level %s : normal vs under-attack' % L_var)
    plt.legend(('under attack', 'normal'), loc = 'lower right')
    plt.xlabel('Hour: from %shr to %shr' % (start_time, end_time))
    plt.ylabel('Water level of tank #%s [meter]' % L_var.partition('_')[2])
    plt.savefig('.\image\comparison\%s-%s.png' % (L_var, start_time), 
                bbox_inches = 'tight', facecolor = 'w')
    plt.show()
    plt.close()
