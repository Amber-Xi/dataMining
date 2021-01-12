#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
import random 

def aug(x):
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x 
    return x_aug
    
def fetch(data):
    
    #fetching the data for x and y
    y = data[:,0]
    x = data[:,1:]
    return x,y
    
def gaussian(data,spread,loss,C):
    '''
    Gaussian method generating for kernel PCA
    '''
    K = np.zeros(shape = (data.shape[0], data.shape[0]))
    delta = -1
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            K[i,j] = np.exp((-(np.linalg.norm(data[i]-data[j])**2))/(2*spread))
            if loss == 'quadratic':
                if i == j:
                    delta = 1
                else:
                    delta = 0
                K[i,j] = K[i,j] + (1/2*C)*delta
    return K, 'gaussian'

def linear(data,spread,loss,C):
    '''
    Linear method generating for kernel PCA
    '''
    K = np.zeros(shape = (data.shape[0], data.shape[0]))
    delta = -1
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            K[i,j] = np.dot(data[i],data[j])
            if loss == 'quadratic':
                if i == j:
                    delta = 1
                else:
                    delta = 0
                K[i,j] = K[i,j] + (1/2*C)*delta
    return K,'linear'

def SVM_DUAL(x,y, C, EPS,MAXITER,loss, algo,spread):
    print("Regularization Constant is ", C)
    print("EPS is ", EPS)
    print("MAXITER is ", MAXITER)
    print("LOSS is ", loss)
    print("KERNEL is ", algo)
    print("KERNEL_PARAM (spread) is ",spread)
    x = aug(x)
    K,name = algo(x,spread, loss,C)
    r,c = np.shape(K)
    step_size = np.diag(K)
    step_size = 1/step_size #calculate the step size
    t = 0
    alpha = np.zeros((r,1))
    alpha = np.concatenate(([], alpha), axis=None)
    while t < MAXITER:
        t = t+1
        perm = np.random.permutation(r) #in random order doing
        x = x[perm]
        y = y[perm]
        step_size = step_size[perm]
        alpha_old = []
        for item in alpha:
            alpha_old.append(item)
        for k in range(0, r):
            tmp = 0
            for i in range(0,r):
                tmp  = tmp + (alpha[i]*y[i]*K[i,k])
            alpha[k] = alpha[k] + step_size[k]* (1-tmp*y[k]) #calculate the step size
            if alpha[k]< 0:
                alpha[k] = 0
            elif alpha[k] > C and loss == 'hinge':
                alpha[k] = C
        if (np.linalg.norm(alpha- alpha_old))< EPS:
            print("Successfully congerve in given iteration")
            break;
 
    predict = [] 
    for i in range(0,r):
        sign = 0
        for j in range(0,r):
            if alpha[j] >0:
                sign = sign + alpha[j]*y[j]*K[j,i]
        if(sign < 0):
            sign = -1
        else:
            sign = 1
        predict.append(sign)
       
    print("The final accuracy : ", list(predict-y).count(0)/len(y))
    
    if name == 'linear': # if linear print the wieght and bias vector
        w = np.zeros((26,1))
        w = np.concatenate(([], w), axis=None)
        for i in range(0,r):
            tmp = x[i]
            tmp1= alpha[i]* y[i]
            w = w + np.multiply(tmp1,tmp[1:]) #calculte the weight given y and alpha
        b = []
        for i in range(0,r):
            tmp = x[i]
            b.append(y[i]-np.dot(w,tmp[1:])) #find the bias
        b_avg =np.mean(b)
        print("The weight vector is: ",w)
        print('The bias vector is: ', b) #print bart of bias
        print("The bias avergae is: ", b_avg)
    
    # count the number within suppotor machine
    if loss == 'hinge':
        count = 0
        for item in alpha:
            if 0 < item < C:
                count += 1
        print("The number of support vectors for HINGE is ", count)
        
    if loss == 'quadratic':
        count = 0
        for item in alpha:
            if 0< item:
                count += 1
        print("The number of support vectors for QUADRATIC is ", count)
                
    
    
#-------------------------------------------------------------------------
file = sys.argv[1]
loss = sys.argv[2]
c = float(sys.argv[3])
eps = float(sys.argv[4])
maxiter = int(sys.argv[5])
Kernel = sys.argv[6]
KERNEL_PARAM = int(sys.argv[7])

with open(file) as f:
    ncols = len(f.readline().split(','))
data = pd.read_csv(file, delimiter=',', skiprows=1,nrows = 5000,usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

for row in data:
    if(row[0] <= 50):
        row[0] = 1
    else:
        row[0] = -1
    
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
if Kernel== 'linear':
    Kernel = linear
else:
    Kernel = gaussian
print("The Traning: ")
SVM_DUAL(data_traning_x,data_traning_y,c, eps, maxiter,loss, Kernel,KERNEL_PARAM)
print()
print("The Testing: ")
SVM_DUAL(data_test_x,data_test_y,c, eps, maxiter,loss, Kernel,KERNEL_PARAM)




