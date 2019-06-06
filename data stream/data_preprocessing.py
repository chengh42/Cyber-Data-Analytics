# -*- coding: utf-8 -*-
"""
CTU-13 Dataset Data Preprocessing

@author: imchengh
"""
import numpy as np
import pandas as pd
from datetime import datetime

def data_preprocessing(filepath):
    # load dataset
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    
    df = {}
    if len(lines[1].split('\t')) == 12: # if format as expected
        # start cleaning
        data = lines[1:] # drop the first row (header)
        
        for i in range(len(data)):
            s = data[i].split('\t')
            if '' in s:
                s = [x for x in s if x] # remove empty elements
            # modify data type (example of data in comments)
            df.update({'StartTime' : datetime.strptime(s[0], '%Y-%m-%d %H:%I:%S.%f').timestamp(), # '2011-08-15 17:13:40.449'
                       'Duration' : float(s[1]),  # '4.204'
                       'Protocol' : s[2],  # 'TCP'
                       'ScrAddr' : s[3],   # '147.32.84.144:22'
                       'DstAddr' : s[5],  # '90.177.154.197:20127'
                       'Flags' : s[6].rstrip(),     # 'PA_'
                       'Tos' : int(s[7]), # '0'
                       'Packets': int(s[8]), # '329'
                       'Bytes' : int(s[9]), # '478578'
                       'Flows' : int(s[10]), # '1'
                       'Label' : s[11].rstrip('\n')}) # 'Background\n'
    else:
        print('Data format not as expected!')
    return df