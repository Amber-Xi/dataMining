#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 21:56:10 2020

@author: heqiongxi
"""

import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

def fetch(path):
    
    #fetching the data for x and y
    y = data[:,0]
    x = data[:,1:]
    return x,y

def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
def fit(x,y,eta,eps,maxiter):
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x #augment the data
    weights = np.zeros(d+1) #initial weight vector
    
    i = 0
    # For randomly going over the data
    perm = np.random.permutation(y.size)
    x_aug = x_aug[perm, :]
    y = y[perm]
    while i < maxiter:
        i += 1
        weight_norm_old = np.linalg.norm(weights) #make a copy
        for j in range(n):
            x_tmp, y_tmp = x_aug[j], y[j]
            gradient = (y_tmp-sigmoid(np.dot(weights.T, x_tmp))) * x_tmp #compute the gradient
            weights += eta * gradient #update
        if abs(np.linalg.norm(weights)- weight_norm_old) < eps: #terminal condition
            print("successfully convergent")
            break
    return weights

def predict(x,y,w):
    # return the predit 0 or 1
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x #augment the data
    return sigmoid(np.matmul(x_aug, w)) >= 0.5
#-------------------------------------------------------------------------

file = sys.argv[1]
with open(file) as f:
    ncols = len(f.readline().split(','))
data = pd.read_csv(file, delimiter=',', skiprows=1,usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

for row in data:
    if(row[0] <= 50):
        row[0] = 1
    else:
        row[0] = 0
    
i = 0
data_traning = []
count = []
while i <= len(data)*0.7:
    #random generate the 70% traning data
    j = random.randint(0,len(data)-1)
    if(j not in count):
        count.append(j)
        data_traning.append(data[j])
        i += 1
        
data_test = []      
for i in range(0,len(data)):
    #random generate the 30% test data
    if(i not in count):
        data_test.append(data[i])
        

data_traning = np.array(data_traning)
data_test = np.array(data_test)

#get the x,y for each data
data_traning_x,data_traning_y = fetch(data_traning)
data_test_x,data_test_y = fetch(data_test)
#--------------------------------------------------------------------------
eta = float(sys.argv[2])
eps = float(sys.argv[3])
maxiter = int(sys.argv[4])
w = fit(data_traning_x,data_traning_y,eta,eps,maxiter)
print("The eta :", eta)
print("The eps :", eps)
print("The maxiter :", maxiter)
print("The weight is : ",w)
print("The norm of weight is", LA.norm(w))
p_train= list(predict(data_traning_x, data_traning_y,w) - data_traning_y)
accurancy_train = p_train.count(0)/len(p_train)
p_test= list(predict(data_test_x, data_test_y,w) - data_test_y)
accurancy_test = p_test.count(0)/len(p_test)
print("The acccurancy for the training data is: ", accurancy_train)
print("The acccurancy for the test data is: ", accurancy_test)
