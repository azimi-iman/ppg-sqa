#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import ppgSQAee as pse
from joblib import load
import os

def classification(sig,fs):
    #Inputs: signal, sampling frequency
    #Returns: prediction value
    ppg_filtered = pse.butter_filtering(sig,fs,[0.6,3.0],5,'bandpass')
    feature_list = pse.feature_extraction(ppg_filtered,fs)
    clf = load('ppg-sqa-svm.joblib')
    y_pred = clf.predict(np.array(feature_list).reshape(1, -1))
    return (y_pred)
              
if __name__ == '__main__':
    fs = 20.0 #Hz
    load_dir = "data/" #PPG samples
    filenames = os.listdir(load_dir)
    if not filenames:
        sys.exit('No file in the directory')
    for f in filenames:
        df = pd.read_csv(load_dir + f)
        y_pred = classification(df['ppg'].values,fs)
        print(y_pred)
    