
import sys
import numpy as np
import pandas as pd
import random

def fetch(path):
    
    #fetching the data for x and y
    y = data[:,0]
    x = data[:,1:]
    return x,y
def aug(x):
    n, d = x.shape
    x_aug = np.ones((n, d + 1))
    x_aug[:, 1:] = x 

    return x_aug;          
def sigmoid(z):
        return 1 / (1 + np.exp(np.clip(-z, -np.Inf, 5e2)))   
def RELU(l):
    # RELU activative function
    r = []
    for x in l:
        if x >0:
            r.append(x)
        else:
            r.append(0)
    return r
def RELU_dri(l):
    # RELU partial deritive function
    r = []
    for x in l:
        if x >0:
            r.append(1)
        else:
            r.append(0)
    return r
def MLP(x,y, m, step, maxiter):
    r,c = data.shape
    # initialize with random parameter
    b_h = []
    for i in range(0,m):
        b_h.append(random.random())
    b_o = []
    for i in range(0,r):
        b_o.append(random.random())
    w_h = np.random.randn(c-1, m)
    w_o = np.random.randn(m, r)
    for i in range(maxiter):
        # do in random order
        perm = np.random.permutation(r)
        x = x[perm]
        y = y[perm]
        for j in range(r):
            # find the net hidden layer and apply the active function
            y_sgd = y[j]
            netz = b_h + np.matmul(w_h.T,x[j])
            z_sgd = RELU(netz)
            # find the net output layer and apply the active function
            neto = b_o + np.matmul(w_o.T,z_sgd)
            o_sgd = sigmoid(neto)
            tmp1 = []
            o_sgd = np.concatenate((tmp1, o_sgd), axis=None)
            for i in range(0,len(o_sgd)):
                # if it is too small, trate as zero
                if(o_sgd[i] < 0.000001):
                    o_sgd[i] = 0
                else: 
                     o_sgd[i] = 1
            # find the delta for the hidden layer and output layer
            y_sgd = np.full((1, r), y_sgd)
            delta_o = o_sgd- y_sgd
            tmp = np.matmul(w_o,delta_o.T)
            tmp1 = []
            tmp = np.concatenate((tmp1, tmp), axis=None)
            delta_h = np.multiply(RELU_dri(netz), tmp)
            
            #apply the gradient
            grad_b_o = delta_o
            b_o = b_o - step * grad_b_o
            grad_b_h = delta_h
            b_h = b_h - step * grad_b_h
            tmp = []
            tmp.append(z_sgd)
            grad_w_o = np.matmul(np.transpose(tmp), delta_o)
            # update the weight maxtrix for the output layer
            w_o = w_o - step* grad_w_o
            grad_w_h = np.outer(x[j], delta_h)
             # update the weight maxtrix for the hidden layer
            w_h = w_h - step* grad_w_h
            
    return w_h, w_o, b_h, b_o
#-------------------------------------------------------------------------
input_file = sys.argv[1]
file = input_file
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

ETA = float(sys.argv[2])
MAXITER = int(sys.argv[3])
HIDDENSIZE = int(sys.argv[4])
#--------------------------------------------------------------------------  
w_h, w_o, b_h, b_o = MLP(data_traning_x,data_traning_y, HIDDENSIZE, ETA, MAXITER)
print("The hidden size is ", HIDDENSIZE)
print("The step size is ", ETA)
print("The maxiter is ", MAXITER)
print("The weight matrix is for hidden layer is", w_h)
print("The bias matrix is for hidden layer is", b_h)
print("The weight matrix is for ouput layer is", w_o)
print("The bias matrix is for output layer is", b_o)
#use the test data doing the forward with  approximation
z = np.transpose(np.matmul(w_h.T,data_test_x.T))
z = sum(z) + b_h
z = RELU(z) 
o = np.transpose(np.multiply(z,w_o.T))
o = sum(o) + b_o
o = sigmoid(o)
tmp1 = []
o = np.concatenate((tmp1, o), axis=None)
predict = []
a = 0;
for i in o:
    if(i >= 0.5):
        a = 1
    else:
        a = 0
    predict.append(a)
p = list(predict - data_test_y).count(0)/len(data_test_y)
print("The final accurancy is ",p)
