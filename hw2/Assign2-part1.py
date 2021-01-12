# Load package
import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA
import sympy as sp
import matplotlib.pyplot as plt

# Complete the pca function
def pca(D,aph):
    mean = np.mean(D,0)
    Z_matrix = D - mean
    covariance_matrix = np.cov(np.transpose(Z_matrix))
    total_variance = np.trace(covariance_matrix)
    value,vector = LA.eig(covariance_matrix)
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
    
    mse = total_variance - sum(reduced_value[0:3])
    print("r is:",i)
    print("Reduced_value is ",reduced_value)
    print("mse is", mse)
    # project the first two pc
    u1 = []
    for row in Z_matrix:
        u1.append(reduced_vector[0]  @ row)
    u2 = []
    for row in Z_matrix:
        u2.append(reduced_vector[1] @ row)
        
    plt.figure(figsize=(10,8))
    plt.scatter(u1, u2)
    plt.xlabel("u1")
    plt.ylabel("u2")
    plt.show()
 
file = sys.argv[1]
aph = float(sys.argv[2])
with open(file) as f:
    ncols = len(f.readline().split(','))

data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

pca(data,aph)
