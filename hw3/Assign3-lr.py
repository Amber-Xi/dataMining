#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:40:10 2020

@author: heqiongxi
"""
import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random

def fetch(data):
    '''
    fetching the data for x and y
    '''
    y = data[:,0]
    x = data[:,1:]
    return x,y

def back_solve(a, b):
    '''
    back solving for w
    '''
    x = np.ones(a.shape[1])
    for i in range(len(x) - 1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, :], x) + a[i, i]) / a[i, i]
    return x


def solve(x,y):
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x #augment the data
    q,r = np.linalg.qr(x_aug) #Q, R 
    delta = np.matmul(q.T,q)
    rhs = np.multiply(1 / np.diag(delta),np.matmul(q.T, y))
    w = back_solve(r, rhs)
    '''
    Test by call inverse
    tmp = np.matmul(np.linalg.inv(r),np.linalg.inv(delta))
    tmp = np.matmul(tmp,np.transpose(q))
    w = np.matmul(tmp,y)
    '''
    norm_w = np.linalg.norm(w)
    return w,norm_w

def predict(x,y,w):
    '''
    solve for sse, tsss, mse
    '''
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x
    predict_y = np.matmul(x_aug,w)
    sse = np.linalg.norm(predict_y - y) ** 2
    tss = np.linalg.norm(y - np.mean(y)) ** 2
    mse = tss/n
    square_r  = (tss - sse) / tss
    return sse,mse,square_r

#---------------------------------------------------------------
file = sys.argv[1]
with open(file) as f:
    ncols = len(f.readline().split(','))
data = pd.read_csv(file, delimiter=',', skiprows=1,usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

i = 0
data_traning = []
count = []
while i <= len(data)*0.7:
    '''
    random generate the 70% traning data
    '''
    j = random.randint(0,len(data)-1)
    if(j not in count):
        count.append(j)
        data_traning.append(data[j])
        i += 1
        
data_test = []      
for i in range(0,len(data)):
    '''
    random generate the 70% test data
    '''
    if(i not in count):
        data_test.append(data[i])
#---------------------------------------------------------------
        
# get the two different purpose data
data_traning = np.array(data_traning)
data_test = np.array(data_test)

#get the x,y for each data
data_traning_x,data_traning_y = fetch(data_traning)
data_test_x,data_test_y = fetch(data_test)


weight, norm_weight = solve(data_traning_x,data_traning_y)
print("The weight vector for the traning data :",weight)
print("The L2 norm weight for the traning data :" ,norm_weight)

sse,mse,square_r = predict(data_traning_x,data_traning_y,weight)

with np.printoptions(precision = 5, suppress = True):
    print("+ Part II \n The SSE is {:}, MSE is {:}, R Square is {:} ".format(sse,mse,square_r))

sse,mse,square_r = predict(data_test_x,data_test_y,weight)

with np.printoptions(precision = 5, suppress = True):
    
    print("+ Part II \n The SSE is {:}, MSE is {:}, R Square is {:} ".format(sse,mse,square_r))
