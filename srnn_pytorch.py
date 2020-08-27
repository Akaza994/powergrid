#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:30:36 2020

@author: zhijianli
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from momentumnet import MomentumLSTMCell
from momentumnet import AdamLSTMCell


class RNN(nn.Module):
    def __init__(self,input_dim,num_units,output_dim,cell_type='plain',dropout=False):
        super(RNN,self).__init__()
        self.num_units=num_units
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.dropout1=nn.Dropout(p=0.1)
        self.dropout2=nn.Dropout(p=0.1)
        self.lstm1=nn.LSTMCell(input_dim,num_units[0])
        self.dropout=dropout
        self.cell_type=cell_type
        if cell_type=='plain':
            self.lstm2=nn.LSTMCell(num_units[0],num_units[1])
            self.lstm3=nn.LSTMCell(num_units[1],num_units[2])
        if cell_type=='momentum':
            self.lstm2=MomentumLSTMCell(self.num_units[0],self.num_units[1],mu=0.5,epsilon=0.5)
            self.lstm3=MomentumLSTMCell(num_units[1],num_units[2],mu=0.5,epsilon=0.5)
        if cell_type=='Adam':
            self.lstm2=AdamLSTMCell(self.num_units[0],self.num_units[1],mu=0.9,epsilon=0.1,mus=0.9)
            #self.lstm2=MomentumLSTMCell(self.num_units[0],self.num_units[1],mu=0.8,epsilon=0.2)
            self.lstm3=AdamLSTMCell(num_units[1],num_units[2],mu=0.9,epsilon=0.1,mus=0.9)
        self.linear=nn.Linear(num_units[-1],self.output_dim)
        
    def forward(self,input):
        #print(input.size[1])
        h_t = torch.zeros(input.size(0), self.num_units[0], dtype=torch.float)
        c_t = torch.zeros(input.size(0), self.num_units[0], dtype=torch.float)
        v1 = torch.zeros(input.size(0),  4*self.num_units[0],dtype=torch.float)
        #print(v1.size())
        h_t2 = torch.zeros(input.size(0), self.num_units[1], dtype=torch.float)
        c_t2 = torch.zeros(input.size(0), self.num_units[1], dtype=torch.float)
        v2 = torch.zeros(input.size(0), 4* self.num_units[1],dtype=torch.float)
        s2 = torch.zeros((input.size(0), 4 * self.num_units[1]), dtype=torch.float)
        h_t3 = torch.zeros(input.size(0), self.num_units[2], dtype=torch.float)
        c_t3 = torch.zeros(input.size(0), self.num_units[2], dtype=torch.float)
        v3 = torch.zeros(input.size(0), 4 * self.num_units[2], dtype=torch.float)
        s3 = torch.zeros((input.size(0), 4 * self.num_units[2]), dtype=torch.float)
        h_t, c_t = self.lstm1(input, (h_t, c_t))
        if self.dropout:
            h_t=self.dropout1(h_t)
        if self.cell_type=='plain':
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        elif self.cell_type=='momentum':
            h_t2, _, _ = self.lstm2(h_t, (h_t2, c_t2), v2)
        elif self.cell_type=='Adam':
            ht_2,_,_,_=self.lstm2(h_t,(h_t2, c_t2), v2, s2) 
            #h_t2, _, _ = self.lstm2(h_t, (h_t2, c_t2), v2)
        if self.dropout:
            h_t2=self.dropout2(h_t2)
        if self.cell_type=='plain':
            h_t3, c_t3=self.lstm3(h_t2, (h_t3, c_t3))
        elif self.cell_type=='momentum':
            h_t3, _, _ = self.lstm3(h_t2, (h_t3, c_t3),v3)
        elif self.cell_type=='Adam':
            ht_3,_,_,_=self.lstm3(h_t2, (h_t3, c_t3), v3, s3) 
        return h_t3
    
    
class NodeRNN(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim,num_layers):
        super(NodeRNN,self).__init__()
        self.lstm=nn.LSTM(input_dim,hidden_size,num_layers)
        self.linear=nn.Linear(hidden_size,output_dim)
        self.dropout=nn.Dropout(p=0.1)
    def forward(self,input):
        h_t,_=self.lstm(input)
        h_t=h_t[:,-1,:]
        output=self.linear(h_t)
        return output


class SRNN(nn.Module):
    def __init__(self,node_to_type, type_to_edge_connections):
        self.node_to_type=node_to_type
        self.type_to_edge_connections=type_to_edge_connections
        self.clss=np.array([self.node_to_type[t] for t in self.node_to_type])
        self.cls_list=np.unique(self.clss)
        self.node_list={}
        for cl in self.cls_list:
            self.node_list[cl]=[v for v in self.node_to_type if self.node_to_type[v]==cl]
        for cl in self.cls_list:
            print("Num Nodes in Class", cl," : ", np.sum(self.clss==cl)) 
        self.EdgeRNNs={}
        self.NodeRNNs={}
    def build_model(self, samples,predictions,num_units,cell_type,hidden_size,num_layers):
        for cl in self.cls_list:
            self.EdgeRNNs[cl]={}
            for edge in self.type_to_edge_connections[cl]:
                self.EdgeRNNs[cl][edge]=RNN(input_dim=samples,num_units=num_units,output_dim=predictions,cell_type=cell_type)
            self.NodeRNNs[cl]=NodeRNN(num_units[-1],hidden_size,predictions,num_layers)
    def joint_train(self,X_train,Y_train,Epoch,cl,milestones=None):
        '''Function to join train all nodes in a class'''
        '''
        Args:
            cl: class number
            X_train: a dictionary 
            X_train[cl]: a list of data points of nodes in class cl, 
                each element in the list should have shape m x n, where m is the number of nodes
                in class cl, n is the number of obervation. 
        '''
        params=[list(net.parameters()) for net in self.EdgeRNNs[cl].values()][0]
        params+=list(self.NodeRNNs[cl].parameters())
        optimizer=torch.optim.Adam(params,lr=0.01)
        if not milestones is None:
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.1)
        criterion=torch.nn.MSELoss()
        for epoch in range(Epoch):
            total_loss=0.0
            num_data=len(Y_train[cl])
            for k in range(num_data):
                Y=torch.from_numpy(Y_train[cl][k])
                Y=Variable(Y)
                edge_outputs=[]
                for edge in self.type_to_edge_connections[cl]:
                    X=torch.from_numpy(X_train[cl][edge][k])
                    X=Variable(X)
                    output=self.EdgeRNNs[cl][edge](X)
                    edge_outputs+=[output.unsqueeze(1)]
                X=torch.cat(edge_outputs,1)
                y_pred=self.NodeRNNs[cl](X)
                loss=criterion(y_pred,Y)
                total_loss+=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not milestones is None:
                    scheduler.step()
            if epoch%50==0:
                print('loss: ',total_loss)
    def joint_eval(self,X_test,Y_test,cl):
        self.NodeRNNs[cl].eval()
        criterion=torch.nn.MSELoss()
        for net in self.EdgeRNNs[cl].values():
            net.eval()
        edge_outputs=[]
        Y=Variable(torch.from_numpy(Y_test))
        for edge in self.type_to_edge_connections[cl]:
            X=torch.from_numpy(X_test[edge])
            X=Variable(X)
            output=self.EdgeRNNs[cl][edge](X)
        edge_outputs+=[output.unsqueeze(1)]
        X=torch.cat(edge_outputs,1)
        y_pred=self.NodeRNNs[cl](X)
        loss=criterion(y_pred,Y)
        y_pred=y_pred.data.numpy()
        print('test loss: ', loss)
        return y_pred
    def per_node_train(self,X_train,Y_train,Epoch,milestones=None):
        for epoch in range(Epoch):
            total_loss=0.0
            for node in self.node_to_type:
                params=[list(net.parameters()) for net in self.EdgeRNNs[node].values()][0]
                params+=list(self.NodeRNNs[node].parameters())
                optimizer=torch.optim.Adam(params,lr=0.01)
                if not milestones is None:
                    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.1)
                criterion=torch.nn.MSELoss()
                Y=torch.from_numpy(Y_train[node])
                edge_outputs=[]
                for edge in self.type_to_edge_connections[node]:
                    X=torch.from_numpy(X_train[node][edge])
                    X=Variable(X)
                    output=self.EdgeRNNs[node][edge](X)
                    edge_outputs+=[output.unsqueeze(1)]
                X=torch.cat(edge_outputs,1)
                y_pred=self.NodeRNNs[node](X)
                loss=criterion(y_pred,Y)
                total_loss+=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not milestones is None:
                    scheduler.step()
            if epoch%50==0:
                print('loss: ',total_loss/len(X_train.keys()))
    def per_node_joint_train(self,X_train,Y_train,Epoch,milestones=None):
        for epoch in range(Epoch):
            total_loss=0.0
            for cl in self.cls_list:
                params=[list(net.parameters()) for net in self.EdgeRNNs[cl].values()][0]
                params+=list(self.NodeRNNs[cl].parameters())
                optimizer=torch.optim.Adam(params,lr=0.01)
                if not milestones is None:
                    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=0.1)
                criterion=torch.nn.MSELoss()
                for node in self.node_list[cl]:
                    Y=torch.from_numpy(Y_train[node])
                    edge_outputs=[]
                    for edge in self.type_to_edge_connections[cl]:
                        #print(cl)
                        #print(edge  )
                        X=torch.from_numpy(X_train[node][edge])
                        X=Variable(X)
                        output=self.EdgeRNNs[cl][edge](X)
                        edge_outputs+=[output.unsqueeze(1)]
                    X=torch.cat(edge_outputs,1)
                    y_pred=self.NodeRNNs[cl](X)
                    loss=criterion(y_pred,Y)
                    total_loss+=loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if not milestones is None:
                        scheduler.step()
            if epoch%50==0:
                print('loss: ',total_loss/len(self.node_to_type.keys()))
    def per_node_eval(self,X_test,Y_test):
        criterion=torch.nn.MSELoss()
        Y_pred={}
        errors={}
        total_loss=0.0
        for node in self.node_to_type:
            cl=self.node_to_type[node]
            self.NodeRNNs[cl].eval()
            for net in self.EdgeRNNs[cl].values():
                net.eval()   
            Y=torch.from_numpy(Y_test[node])
            edge_outputs=[]
            for edge in self.type_to_edge_connections[cl]:
                X=torch.from_numpy(X_test[node][edge])
                X=Variable(X)
                output=self.EdgeRNNs[cl][edge](X)
                edge_outputs+=[output.unsqueeze(1)]
            X=torch.cat(edge_outputs,1)
            y_pred=self.NodeRNNs[cl](X)
            loss=criterion(y_pred,Y)
            errors[node]=loss.data
            total_loss+=loss
            Y_pred[node]=y_pred.data.numpy()
        total_loss/=len(self.node_to_type.keys())
        print('average test loss: ', total_loss)
        return Y_pred, errors