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
