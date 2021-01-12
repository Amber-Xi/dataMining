# Load package
import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

file = sys.argv[1]
eps = float(sys.argv[2])
with open(file) as f:
    ncols = len(f.readline().split(','))

data = pd.read_csv(file, delimiter=',', skiprows=1, usecols=range(1,ncols-1),header = None)

#computing the mean
column_mean = np.sum(data, axis=0)/data.count(axis=0)
column_mean = column_mean.to_numpy()
data = data.to_numpy()

#computing the inner and outer product
one = np.ones(shape = (data.shape[0], 1))
mean_matrix =  np.reshape(column_mean, (1, data.shape[1]))
Z_matrix = data - np.dot(one,mean_matrix)

#inner covariance
cov_inner = np.matmul(Z_matrix.T, Z_matrix) / data.shape[0]
#total variance
total_vari = np.trace(cov_inner)
#outter covariance
total = 0
for row in Z_matrix:
    total += np.outer(row, row)
cov_outer = total/data.shape[0]

#corrleation
matrix_norm = LA.norm(Z_matrix, axis=0)
correlation  = np.matmul((Z_matrix / matrix_norm).T, Z_matrix / matrix_norm)

with np.printoptions(precision = 5, suppress = True):
    print("+ Part I a)\n- Mean(D):\n{:}\n\n- TotalsVar(D):\n{:.3f}\n".format(column_mean, total_vari))
    print("+ Part I b)\n- Cov(D) Inner:\n{:}\n\n- Cov(D) Outer:\n{:}\n".format(cov_inner, cov_outer))
    print("+ Part I c)\n- Corr(D):\n{:}\n".format(correlation))
    


# heatmap of correlation
fig = plt.figure()
fig.set_size_inches(5,5)
plt.imshow(correlation)
plt.title("heatmap of correlation")
plt.show() 

#The least related data 25 and 1  --- Visibility and Appliances --- -0.0002
plt.figure(figsize=(5,5))
plt.scatter(Z_matrix[:, 24], Z_matrix[:, 0], s = 50, c ='#1f77b4', alpha=0.5)
plt.xlabel('data column #25')  
plt.ylabel('data column #1')  
plt.title("Least Correlated Attributes")
plt.show()  

#The most related data 21 and 13  --- T1 and RH_out --- 0.9748
plt.figure(figsize=(5,5))
plt.scatter(Z_matrix[:, 20], Z_matrix[:, 12], s = 50,c = '#ff7f0e', alpha=0.5)
plt.xlabel('data column #21')  
plt.ylabel('data column #13')  
plt.title("Most Correlated Attributes")
plt.show()


#The most anti-related data 15 and 14  ---  RH_6 and T7 --- -0.754
plt.figure(figsize=(5,5))
plt.scatter(Z_matrix[:, 14], Z_matrix[:, 13], s = 50,c = '#2ca02c', alpha=0.5)
plt.xlabel('data column #15')  
plt.ylabel('data column #14')  
plt.title("Most An-ti Correlated Attributes")
plt.show()            
   
print("+ Part II\n- Dominant eigen vector and eigen value")
print()
def powerIteration(matrix, limit):
    k = 0
    p_pre = np.ones(shape = matrix.shape[1])
    err = 1
    while (err > limit):
        k = k + 1
        p_next= np.matmul(matrix.T, p_pre)
        i = np.argmax(p_next)
        lamda = p_next[i]/p_pre[i]

        p_next = p_next / p_next[i]
        err = np.linalg.norm(p_next - p_pre)
        p_pre = p_next
        with np.printoptions(precision = 5, suppress = True):
            print("Itr: {:} - Eigen Val: {:.3f}; \n Eigen Vec {:}; \nError: {:.2f}".format(k, lamda, p_next, err))
            print()
    return p_next


ev = powerIteration(cov_inner, eps)
projections = []
for row in data:
    projections.append((row @ ev))

plt.figure(figsize=(5,5))
plt.scatter(range(len(projections)), projections)
plt.title("Distribution Along New Axis")
plt.show()