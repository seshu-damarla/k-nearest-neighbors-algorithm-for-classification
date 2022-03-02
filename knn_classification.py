# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:37:18 2022

@author: Seshu Kumar Damarla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
#from collections import counter
from scipy.stats import mode

np.random.seed(0)

xdata = pd.read_csv('xdata.csv', header=None)
ydata = pd.read_csv('ydata.csv', header=None)

xdata = np.array(xdata)
ydata = np.array(ydata)

m = xdata.shape[0]
n_inputs = xdata.shape[1]

# randomly selecting training and testing subsets
permutt = list(np.random.permutation(m))
#print(permutt)
shuffled_x = xdata[permutt,:]
shuffled_y = ydata[permutt,:]

trainx = shuffled_x[0:120,:]
trainy = shuffled_y[0:120,:]

m2 = trainx.shape[0]

testx = shuffled_x[120:150,:]
testy = shuffled_y[120:150,:]

m1 = testx.shape[0]
k = 5  # no. of nearest neighbors
ypred=[]
for i in range(m1):
    query_sample = testx[i,:]
    distance_index = []
    for j in range(m2):
        train_sample = trainx[j,:]
        dist = np.linalg.norm(query_sample-train_sample)
        distance_index.append(dist)
        
    didx = sorted(range(len(distance_index)), key=lambda k: distance_index[k])
    didx = np.array(didx)
    didx = didx.reshape(len(didx),1)
#    print(didx.shape)
    ylabels = trainy[didx[1:k]]
    ypred.append(mode(ylabels)[0][0])

ypred = np.array(ypred)
ypred = ypred.reshape(ypred.shape[0],1)

#print(ypred)
#print(np.array([testy, ypred]))

plt.plot(ypred)
plt.plot(testy)


        