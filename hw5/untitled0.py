#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:11:35 2020

@author: heqiongxi
"""

i = 1
j = 3

def outer1():
    print(i)
    
def outer2(P):
    i = 2
    def middle(k):
        def inner():
            i = 4
            P()
            
        inner()
        print(i,j,k)
    middle(j)
    
outer2(outer1)
print(i,j)

