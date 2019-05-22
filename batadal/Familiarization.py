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

# %% function for tranforming string date to timestamp and vice versa

def string_to_timestamp(s):
    return datetime.datetime.strptime(s, "%d/%m/%y %H").timestamp()

def timestamp_to_date(t):
    return datetime.datetime.fromtimestamp(t).strftime('%m-%d %H') #to covert timestamp to datetime

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
dftrain1.DATETIME = dftrain1.DATETIME.apply(lambda s: string_to_timestamp(s))

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
