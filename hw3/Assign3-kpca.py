import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time

def project(K_center,reduced_vector,name):
    '''
    Projection function
    '''
    u1 = []
    for row in K_center:
        u1.append(reduced_vector[0]  @ row)
    u2 = []
    for row in K_center:
        u2.append(reduced_vector[1] @ row)
        
    plt.figure(figsize=(10,8))
    plt.scatter(u1, u2)
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.title(name)
    plt.show()
    
def Gaussian(data,spread):
    '''
    Gaussian method generating for kernel PCA
    '''
    K = np.zeros(shape = (data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            K[i,j] = np.exp((-(np.linalg.norm(data[i]-data[j])**2))/(2*spread))
    return K

def linear(data,spread):
    '''
    Linear method generating for kernel PCA
    '''
    K = np.zeros(shape = (data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            K[i,j] = np.dot(data[i],data[j])
    return K

def pca(D,aph,name):
    '''
    Stardard PCA implementation
    '''
    mean = np.mean(D,0)
    Z_matrix = D - mean
    covariance_matrix = np.cov(np.transpose(Z_matrix))
    total_variance = np.trace(covariance_matrix)
    value,vector = LA.eig(covariance_matrix) # calculating the eigenvector and eigenvalue

    # Calculate the reduced value and vector
    l = len(value)
    i = 0
    total_r = 0
    while(i<l+1):
        if(total_r/total_variance>=aph):
            break
        total_r = total_r + value[i]
        i += 1
        
    reduced_vector = vector[0:i]
    reduced_value = value[0:i]
    
    # projection according to the first two rwo of eigenvector
    print_vector_row = []
    print_vector_row.append(vector[0])
    print_vector_row.append(vector[1])
    project(Z_matrix,print_vector_row,name)
    
    # projectoin according to the first two column of eigenvector
    print_vector_column = vector[:,:2]
    project(Z_matrix,np.transpose(print_vector_column),name)

    return i, reduced_value
    
def kernalPCA(data, aph, spread, fun,name):
    '''
    General kernel PCA implementation
    '''
    K = fun(data,spread) #the k matirx
    n = K.shape[0] #number of row
    K_center = (np.identity(n) - 1.0/n*np.ones(shape = (n, n))) @ K @ (np.identity(n) - 1.0/n*np.ones(shape = (n, n)))
    
    value,vector = LA.eigh(K_center)
    vector = np.fliplr(vector)
    value = value[::-1]

    #Just select the eigen value greater than zero with corresponding eigenvector
    selections = np.array(list(map(lambda eig: True if eig > 0 else False, value)))
    vector = vector[:, selections]
    value = list(filter(lambda x: x > 0, value))
    
    # Get Variance based on the eigenvalue
    variance = []
    for v in value:
        variance.append(v/len(value))
    # Norm the vector(i) with multiplity 1/sqrt(value(i))
    norm_vector = []
    for i in range(len(value)):
        norm_vector.append(np.multiply(math.sqrt(1/value[i]), vector.T[i]))


    total_r = 0
    i =0
    total_variance = sum(variance)
    
    # Capture the variance with smallest dimension
    while(i<len(variance)):
        if(total_r/total_variance >= aph):
            break
        total_r = total_r + variance[i]
        i += 1
        if(i >= len(variance)):
            print("cannot reduce")
            break
    
    
    reduced_vector = norm_vector[0:i]
    reduced_value = value[0:i]
    print_vector = []
    
    #print the projection
    print_vector.append(norm_vector[0])
    print_vector.append(norm_vector[1])
    project(K_center,print_vector,name)
    return i, reduced_value 

file = sys.argv[1]
aph = sys.argv[2]
aph = float(aph)
spread = sys.argv[3]
spread = int(spread)

# file input
with open(file) as f:
    ncols = len(f.readline().split(',')) 

data = pd.read_csv(file, delimiter=',', skiprows=1,nrows = 5000,usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

linear = kernalPCA(data,aph,spread,linear,"linear kernel projection")
print("The reduced dimension for linear Kernel PCA is ",linear[0])
print("The reduced eigenvalue for linear Kernel PCA is", linear[1])

pca = pca(data,aph,"Standard PCA projection")
print("The reduced dimension for Standard PCA is ",pca[0])
print("The reduced eigenvalue for Standard PCA is", pca[1])

kernel = kernalPCA(data,aph,spread,Gaussian, "Gaussian kernel projection")
print("The reduced dimension is ",kernel[0])
print("he reduced eigenvalue for Gaussian kernel is ",kernel[1])
