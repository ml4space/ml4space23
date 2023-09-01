#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ludwig Krippahl
"""
Auxiliary function for exercise 3
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_svm(data, sv, C):
    plt.figure(figsize=(5,5))
    plt.title(f'Regularization factor (C) of {C}')
    pxs = np.linspace(-2.5,2.5,200)
    pys = np.linspace(-2.5,2.5,200)        
    pX,pY = np.meshgrid(pxs,pys)
    pZ = np.zeros((len(pxs),len(pys)))
    xts = np.zeros((len(pxs),2))
    xts[:,1] = pys
    for col in range(len(pxs)):
        xts[:,0] = pxs[col]
        pZ[:,col] = sv.decision_function(xts)    
    y = data[:,-1]
    plt.plot(data[y<0,0],data[y<0,1],'o',mec='k')    
    plt.plot(data[y>0,0],data[y>0,1],'o',mec='r')
    plt.contourf(pX, pY, pZ, [-1e9, 0, 1e9],
                 colors = ('b','r'), alpha=0.2)
    plt.contour(pX, pY, pZ, [-1, 0, 1], linewidths =(2,3,2), colors = 'k',
                linestyles='solid')