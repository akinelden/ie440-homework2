# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set() # Plot styling

#%%

coordinates = pd.read_csv('coordinates_2_.csv')
demands = pd.read_csv('demand_2_.csv')
costs = pd.read_csv('costs_2_.csv')

A = np.asarray(coordinates)
H = np.asarray(demands)
C = np.asarray(costs)

#%%

def squaredDistSolforSingle(H=[], A=[], C=[], n=41): 
    facility42 = np.delete(C[n], 0)
    
    x_v1_star = (np.sum(np.multiply(np.multiply(H[:,1],facility42),A[:,1]))/np.dot(H[:,1], facility42))
    x_v2_star = (np.sum(np.multiply(np.multiply(H[:,1],facility42),A[:,2]))/np.dot(H[:,1], facility42))
 
    return x_v1_star, x_v2_star

#%%

plt.scatter(A[:, 1], A[:, 2], s=np.size(A,axis=0))

[x1, x2] = squaredDistSolforSingle(H,A,C,41)
