#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:13:29 2020

@author: zhijianli
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error as mse

def read_files(cell_type='plain',cl=0,dynamic=False):
    filename=''
    filename+=cell_type
    filename+='class'
    filename+=str(cl)
    if dynamic:
        filename+='dynamic'
    filename+='.csv'
    Y_pred=np.loadtxt(filename,delimiter=',')
    Y=np.loadtxt(str(cl)+'exact.csv',delimiter=',')
    return Y_pred, Y

def bsort(u,v):
    n=len(u)
    for i in range(n-1):
        for j in range(n-i-1):
            if u[j]>u[j+1]:
                u[j], u[j+1]=u[j+1], u[j]
                v[j], v[j+1]=v[j+1], v[j]
    return u, v

def convert_dict(X,k=0,m=200):
    n=len(X.keys())
    Y=np.empty((n,m),dtype=np.float32)
    for node in X.keys():
        Y[node,:]=X[node][k,:]
    return Y

def ave_error(Y_pred,Y):
    return np.sqrt(mse(Y_pred,Y))


def ave_error2(Y_pred,Y,n=1):
    error=0
    for k in range(n):
        y_hat,y=convert_dict(Y_pred,k=k),convert_dict(Y,k=k)
        error+=ave_error(y_hat,y)
    error/=n
    return error

def save_data(cell_type,num_cls,clustering,dynamic,Y_pred,Y,dire='result'):
    foldername=cell_type+clustering+str(num_cls)+'class'
    if dynamic:
        foldername+='dynamic'
    foldername=dire+'/'+foldername
    directory=os.path.join(foldername)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for node in Y_pred.keys():
        filename=foldername+'/Node'+str(node)+'Pred.csv'
        np.savetxt(filename,Y_pred[node],delimiter=',')
        filename=foldername+'/Node'+str(node)+'Exact.csv'
        np.savetxt(filename,Y[node],delimiter=',')
        
def load_results(cell_type,clustering,num_cls,dynamic,k=-1,dire='result'):
    
    foldername=cell_type+clustering+str(num_cls)+'class'
    if dynamic:
        foldername+='dynamic'
    foldername=dire+'/'+foldername
    directory=os.path.join(foldername)
    if k==-1:
        Y_pred={}
        Y={}
        for node in range(68):
            filename=foldername+'/Node'+str(node)+'Pred.csv'
            pred=np.loadtxt(filename,delimiter=',')
            filename=foldername+'/Node'+str(node)+'Exact.csv'
            exact=np.loadtxt(filename,delimiter=',')
            Y_pred[node]=pred
            Y[node]=exact
    else:
        Y_pred=np.empty((68,200),np.float32)
        Y=np.empty((68,200),np.float32)
        for node in range(68):
            filename=foldername+'/Node'+str(node)+'Pred.csv'
            pred=np.loadtxt(filename,delimiter=',')[k,:]
            filename=foldername+'/Node'+str(node)+'Exact.csv'
            exact=np.loadtxt(filename,delimiter=',')[k,:]
            Y_pred[node,:]=pred
            Y[node,:]=exact
    return Y_pred, Y
def plot(Y_pred,Y):
    fig=plt.figure(figsize=(8,8))
    for i in range(9):
        fig.add_subplot(3,3,i+1)
        plt.plot(Y_pred[i+10,:],label='prediction')
        plt.plot(Y[i+10,:],label='exact')
        plt.legend()
        error=np.sqrt(mse(Y_pred[i,:],Y[i,:]))
        plt.title('RMSE: '+str(error))
    plt.show()
def plot2(Y_pred,Y,k=0,m=200):
    pred,true=convert_dict(Y_pred,k=k,m=m),convert_dict(Y,k=k,m=m)
    plot(pred,true)
    return