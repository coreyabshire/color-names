

import os
import sys
import csv

import numpy as np
import scipy
import pandas as pd

from sklearn.ensemble.forest import RandomForestClassifier

class ColorChip(object):
    def __init__(self, cid, lab_coords, obs_cats):
        pass

class ColorChipDS(object):
    def __init__(self):
        pass

def read_color_chip_data(filename):
    ds = ColorChipDS()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print row

def just_pred(x, y):
    xlen = len(x)
    i = range(xlen)
    np.random.shuffle(i)
    trainpct = 0.7
    trainlen = int(trainpct * xlen)
    testlen = xlen - trainlen
    xtrain = x.ix[:trainlen,:]
    ytrain = y.ix[:trainlen]
    xtest = x.ix[trainlen:,:]
    ytest = y.ix[trainlen:]
    rf = RandomForestClassifier()
    rf.fit(xtrain, ytrain)
    ypred = rf.predict(xtest)
    return ytest, ypred

def crossval(x, y, k=5):
    for i in range(k):
        i = range(len(X))
        np.random.shuffle(i)
        xlen = len(x)
        trainpct = 0.7
        trainlen = int(trainpct * xlen)
        testlen = xlen - trainlen
        xtrain = x.ix[:trainlen,:]
        ytrain = y.ix[:trainlen]
        xtest = x.ix[trainlen:,:]
        ytest = y.ix[trainlen:]
        rf = RandomForestClassifier()
        rf.fit(xtrain, ytrain)
        ypred = rf.predict(xtest)
        print ypred

if __name__ == '__main__':
    filename = '../../../../data/old_data.csv'
    X = pd.read_csv(filename)
    X = X.ix[0:len(X)-5,:]
    x = X.ix[:,2:5]
    y = X.ix[:,5]
    ytest, ypred = just_pred(x, y)
    #ds = read_color_chip_data(filename)
    #print ds
    
