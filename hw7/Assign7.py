import sys
import math
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import random 


def initialize_mean(k,data):
    # initialize means with first few vector
    means = []
    for i in range(k):
        means.append(data[i])
    return means

def dense_fun(x,mean,cov_matrix, d, ridge):
    cov_matrix = cov_matrix + ridge  * np.identity(d) #add the ridge
    center = x-mean
    cov_inverse = np.linalg.inv(cov_matrix) 
    cov_det = np.linalg.det(cov_matrix)
    g = -0.5 * np.dot(np.dot(center.T, cov_inverse),center)
    dense = np.power(2*np.pi, -d/2) * np.power(cov_det, -0.5) * np.exp(g)
    dense = math.log(dense)
    return dense


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3     
    

def EM(data, k, eps, maxiter,ridge):
    n,d = data.shape
    # initialize 
    means = initialize_mean(k,data)
    cov = [np.identity(d) for i in range(k)]
    prior_prob = [1/k for i in range(k)]
    t = 0
    w = np.zeros((k,n))
    while t < maxiter:
        t += 1
        # Expectation Step
        for i in range(k):
            old_means = np.copy(means)
            for j in range(n):
                des_i =  dense_fun(data[j],means[i],cov[i], d,ridge) + np.log( prior_prob[i])
                des_i = np.exp(des_i) #posterior probability
                norm_sum = 0
                for a in range(k):
                    norm_sum += np.exp(dense_fun(data[j],means[a],cov[a], d,ridge) + np.log( prior_prob[a]))
                w[i,j] = des_i/logsumexp(norm_sum) # do the logsumexp
               
        for i in range(k):
            sum_weight = sum([w[i,j] for j in range(n)])
            try:
                means[i] = sum([w[i,j]*data[j] for j in range(n)])/sum_weight # re-estimation mean
            except ZeroDivisionError:
                print("ZeroDivisionError")
            try:
                cov[i] = sum([w[i,j] * np.outer((data[j]-means[i]),data[j]-means[i].T) for j in range(n)])/sum_weight # re-estimation covariance matirx
            except ZeroDivisionError:
                print("ZeroDivisionError")
            prior_prob[i] = sum_weight/n #re-estimate priors
        error =  0
        for i in range(k): # find when to convegence
            error += np.power(np.linalg.norm(means[i] - old_means[i]),2)
        if error < eps: 
            print("Success with total iterations: ", t)
            break;
    
    print("The final mean for each cluster")
    print(means)  
    np.set_printoptions(precision=2)
    print("The final covariance matrix for each cluster")
    print(cov)

    cluster = {}
    cluster[0] = []
    cluster[1] = []
    cluster[2] = []
    cluster[3] = []
    cluster[4] = []
    cluster[5] = []
    '''
    find the  Size of each cluster
    '''
    for i in range(n):
        max_val = w[0,i]
        max_in = 0
        for j in range(k):
            if w[j,i] > max_val:
                max_val = w[j,i]
                max_in = j
        cluster[max_in].append(list(data[i]))
    for i in range(k):
        print("cluster: ", i)
        print(" Length is: " ,len(cluster[i]))
    return cluster
                   
file = sys.argv[1]
k = int(sys.argv[2])
eps = float(sys.argv[3])
ridge = int(sys.argv[4])
maxiter = int(sys.argv[5])
print("EPS: ", eps)
print("Ridge: ", ridge) 

with open(file) as f:
    ncols = len(f.readline().split(','))
data = pd.read_csv(file, delimiter=',',skiprows=1, nrows = 5000, usecols=range(1,ncols-1),header = None)
data = data.to_numpy()

t_cluster = {}
t_cluster[0] = []
t_cluster[1] = []
t_cluster[2] = []
t_cluster[3] = []
t_cluster[4] = []
t_cluster[5] = []

for row in data:
    if(10<=row[0] <= 40):
        t_cluster[0].append(list(row[1:]))
    if row[0] ==50:
        t_cluster[1].append(list(row[1:]))
    if row[0] == 60:
        t_cluster[2].append(list(row[1:]))
    if 70<=row[0] <=90:
        t_cluster[3].append(list(row[1:]))
    if 100<=row[0] <=160:
        t_cluster[4].append(list(row[1:]))
    if 170<=row[0] <=1080:
        t_cluster[5].append(list(row[1:]))
        
random.shuffle(data)
cluster = EM(data[:,1:], 6, 0.001,100, 7)
n = data.shape[0]
'''
find the the 'purity score'
'''
score = 0
for i in range(k):
    score_lst = []
    for j in range(6):
        tmp = len(intersection(cluster[i], t_cluster[j]))
        score_lst.append(tmp)
    score += max(score_lst)
purity_score = score/n
print("The purity_score is: ", purity_score)