#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:06:36 2020

@author: heqiongxi
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import statistics 

def solver(d):
    # Random generate 100000 times, d pairs
    half_D_Pair = []
    mylist = [1, -1]
    for i in range(100000):
        half_D_Pair.append(([random.choice(mylist) for j in range(d)],[random.choice(mylist) for j in range(d)]))
    
    #Calculate the angle
    angles = []
    for half in half_D_Pair:
        norm1 = np.linalg.norm(half[0])
        norm2 = np.linalg.norm(half[1])
        radian = np.dot(half[0],half[1])/(norm1 * norm2)
        degree = np.rad2deg(np.arccos(radian))
        angles.append(degree)
    
    #Calculate the probality
    dictionary = {}
    for a in angles:
        if a not in dictionary.keys():
            dictionary[a] = 1
        else:
            dictionary[a] += 1
    
    for k in dictionary.keys():
        dictionary[k] = dictionary[k]/len(angles)
    return angles,dictionary


d_list = [10,100,1000]
for i in d_list:
    angles, dictionary = solver(i)
    angles_min = min(angles)
    angles_max = max(angles)
    angles_mean = statistics.mean(angles)
    print("The max angle is: ", angles_max)
    print("The minimum angle is: ", angles_min)
    print("The Range is: ", abs(angles_max - angles_min))
    print("The Mean is: ", angles_mean)
    print("Variance is: ", statistics.variance(angles))
        
    plt.figure()
    plt.bar(dictionary.keys(), dictionary.values(),8, color='k')
    plt.title("Probability Mass Function Of d = {:}".format(i))
    plt.xlabel("Angle")
    plt.ylabel("Probality")
    plt.show()
