import sys
import math
import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from scipy.special import logsumexp
import random 


def initialize_mean(k,data):
    means = []
    for i in range(k):
        means.append(data[i])
    return means

def initialize_cluster(k):
    cluster = {}
    for i in range(k):
        cluster[i] = []
    return cluster

def density_cal(x,mean,cov_matrix, d, ridge): 
    dense = 1/(np.power(2*np.pi, d/2) * np.power(np.linalg.det(cov_matrix + ridge  * np.identity(d)), 0.5)) * np.exp(-0.5 * np.dot(np.dot((x-mean).T, (np.linalg.inv(cov_matrix + ridge  * np.identity(d)))),(x-mean)))
    return dense 
    
def EstimationMax(data, k, eps, maxiter,ridge):
    n,d = data.shape
    means = initialize_mean(k,data)
    cov = [np.identity(d) for i in range(k)]
    prior_prob = [1/k for i in range(k)]
    w = np.zeros((k,n))
    for t in range(maxiter):
        for i in range(k):
            old_means = np.copy(means)
            for j in range(n):
                des_i =  np.exp(np.log(density_cal(data[j],means[i],cov[i], d,ridge)) + np.log( prior_prob[i]))
                norm_sum = 0
                for a in range(k):
                    norm_sum += np.exp(np.log(density_cal(data[j],means[a],cov[a], d,ridge)) + np.log( prior_prob[a]))
                w[i,j] = des_i/logsumexp(norm_sum)
               
        for i in range(k):
            means[i] = sum([w[i,j]*data[j] for j in range(n)])/sum([w[i,j] for j in range(n)])
            cov[i] = sum([w[i,j] * np.outer((data[j]-means[i]),data[j]-means[i].T) for j in range(n)])/sum([w[i,j] for j in range(n)])
            prior_prob[i] = sum([w[i,j] for j in range(n)])/n

        error = sum([np.power(np.linalg.norm(means[i] - old_means[i]),2) for i in range(k)])
        if error < eps: 
            print("At iteration {}, it is convergent".format(t))
            break; 
    print('Means: -------------------------------------------------------') 
    print(means) 
    print('Covariance Matrix: -------------------------------------------------------') 
    print(cov)
            
    return means, cov, w

def shatter_data(data):
    t_cluster = initialize_cluster(6)
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
    return t_cluster

def find_score(w,k,t_cluster,data):
    n = len(w[0])
    cluster = initialize_cluster(k)
    for i in range(n):
        max_val = w[0,i]
        max_in = 0
        for j in range(k):
            if w[j,i] > max_val:
                max_val = w[j,i]
                max_in = j
        cluster[max_in].append(list(data[i]))
        
    for i in range(k):
        print("cluster: {}, Length is {}".format(i, len(cluster[i])))
    score = 0
    for i in range(6):
        score_lst = []
        for j in range(k):
            tmp = [value for value in t_cluster[j] if value in cluster[i]] 
            score_lst.append(len(tmp))
        max_score = max(score_lst)
        score += max_score
    print("The purity_score is: ", score/n)
    
if __name__ == "__main__":
	file = sys.argv[1]
	k = int(sys.argv[2])
	eps = float(sys.argv[3])
	ridge = int(sys.argv[4])
	maxiter = int(sys.argv[5])
	print("The input k {}, eps {}, ridge {}, maxiter {}:".format(k, eps, ridge, maxiter))
	with open(file) as f:
	    ncols = len(f.readline().split(','))
	data = pd.read_csv(file, delimiter=',',skiprows=1,usecols=range(1,ncols-1),header = None)
	data = data.to_numpy()
	data_without_first = data[:,1:]
	t_cluster = shatter_data(data)
	random.shuffle(data_without_first)
	means, cov, weight= EstimationMax(data_without_first, k, eps,maxiter,ridge)
	find_score(weight,k,t_cluster,data_without_first)





