#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:50:16 2020

@author: zhijianli
"""

import numpy as np
import scipy 
import scipy.io as sio
import os
from sklearn.cluster import KMeans
import tensorflow as tf
from torch.autograd import Variable
import torch



def partition_to_three_parts(fpath):
    df = pd.read_csv(fpath, header = None)
    States = range(df.shape[1])
    eventCounts = np.zeros(shape=[len(States)])
    for state_id in States:
        eventCounts[state_id] = np.sum(df[state_id])
    m = np.min(eventCounts)
    M = np.max(eventCounts)
    node_to_type = {}
    for node, ec in enumerate(eventCounts):
        group_num = int(np.floor(float((ec - m) / (M - m + 1e-6)) * 2.))
        if group_num > 1:
            group_num = 1
        node_to_type[node] = group_num
        print (node,':',group_num)
    return node_to_type

def load_graph_and_normalize(fpath, threshold=0.025, symmetrize=True, exclude_self=True,dynamic=False): 
    weighted_graph = np.loadtxt(fpath, delimiter=',')
    #weighted_graph=np.eye(70)
    if symmetrize:
        weighted_graph = 0.5 * (weighted_graph + np.transpose(weighted_graph))
    if exclude_self:
        N = len(weighted_graph[0])
        for ind in range(N):
            weighted_graph[ind, ind] = 0
    weighted_graph[weighted_graph < threshold] = 0.0
    # normalize by max degree
    degree = np.sum(weighted_graph, axis=1)
    M = np.max(degree)
    weighted_graph = weighted_graph / M
    graph = {}
    N = weighted_graph.shape[0]
    for n1 in range(N):
        graph[n1] = {}
        for n2 in range(N):
            if(weighted_graph[n1][n2] > 1e-8):
                if dynamic:
                    graph[n1][n2]=Variable(torch.tensor(weighted_graph[n1][n2]),requires_grad=True)
                else:
                    graph[n1][n2] = weighted_graph[n1][n2]
    return graph


# Xiyang: Modify to one node RNN per US State.
def partition_to_one_node_per_state(fpath):
    df = pd.read_csv(fpath, header = None)
    States = range(df.shape[1])
    num_states = len(States)

    node_to_type = {}
    for node in xrange(num_states):
        node_to_type[node] = node
    return node_to_type    





def generate_data(file_num,suffix,placeholder=[]):
    for num in file_num:
        file_name='event'+str(num)+suffix
        path = ''#os.path.join(DATA_DIR, 'USStates')
        temp=np.absolute(sio.loadmat(os.path.join(path,file_name))['bus_v'])[:68,:]
        #temp=sio.loadmat(os.path.join(path,file_name))['bus_v'][:68,:]
        placeholder.append(temp)
    return placeholder


def generate_files(num_list,suffix_list,path=''):
    placeholder=[]
    for num,suffix in zip(num_list,suffix_list):
        file_name='event'+num+suffix
        #path=''
        temp=np.absolute(sio.loadmat(os.path.join(path,file_name))['bus_v'])[:68,:]
        placeholder.append(temp)
    return placeholder


class DataReaderPowergrid():
    def __init__(self,samples,predictions):
        self.num_nodes=68
        self.samples=samples
        self.predictions=predictions
        self.path = ''#os.path.join(DATA_DIR, 'USStates')
        self.node_to_type={}
        self.node_graph={}
        self.type_to_edge_connections={}
    def GenerateNodeFeatures(self,files=None):
        trX_Raw={}
        trY_Raw={}
        if files==None:
            files=generate_data(file_num1,suffix1,[])
        for node_id in range(self.num_nodes):
            tempX=np.empty((len(files),self.samples),dtype=np.float32)
            tempY=np.empty((len(files),self.predictions),dtype=np.float32)
            for i in range(len(files)):
                tempX[i,:]=files[i][node_id,:self.samples]
                tempY[i,:]=files[i][node_id,self.samples:1000]
            #tempX=tempX.reshape(list(tempX.shape)+[1])
            trX_Raw[node_id]=tempX
            trY_Raw[node_id]=tempY
        return trX_Raw, trY_Raw
    


    def GenerateNodeClassification(self):
        node_to_type = {}
        for node in range(self.num_nodes):
            node_to_type[node] = node
        self.node_to_type=node_to_type
        return node_to_type
    def GenerateNodeClassification1(self,num_group):
        node_to_type = {}
        X=np.empty((self.num_nodes,1000),dtype=np.float32)
        X[:,:]=generate_data(file_num1,suffix1,[])[0][:,:1000]
        kmeans = KMeans(n_clusters=num_group, random_state=0).fit(X)
        labels=kmeans.labels_
        #print(labels.shape)
        for node in range(self.num_nodes):
            node_to_type[node]=labels[node]
        self.node_to_type=node_to_type
        return node_to_type
    def GenerateGraph(self, threshold=0.01, symmetrize=True, exclude_self=True,dynamic=False):
        return load_graph_and_normalize(os.path.join(self.path, 'PowerGraph.csv'), 
            threshold=threshold, symmetrize=symmetrize, exclude_self=exclude_self,dynamic=dynamic)

    def add_edge_features(self, trX_raw, trY_raw, external_dims=5):
        graph=self.node_graph
        type_to_edge_connections = self.type_to_edge_connections
        node_to_type = self.node_to_type
        self.types=list(set(self.node_to_type.values()))
        for t in self.types:
            self.type_to_edge_connections[t] = []
        type_edges = set()
        for node in graph: 
            t = node_to_type[node]
            ms = graph[node]
            for m in ms: 
                t2 = node_to_type[m]
                ind1 = min(t, t2)
                ind2 = max(t, t2)
                tok = str(ind1) + '_' + str(ind2)
                if(tok not in type_edges):
                    type_edges.add(tok)
                    if tok not in self.type_to_edge_connections[t]: 
                        self.type_to_edge_connections[t] += [tok]
                    if tok not in self.type_to_edge_connections[t2]: 
                        self.type_to_edge_connections[t2] += [tok]
        for t in self.types:
            tok = str(t) + '_' + 'input'
            type_edges.add(tok)
            self.type_to_edge_connections[t] += [tok]




        trX = {}  # {node: {edge_name: input_value}}
        trY = trY_raw  # {node: label_value}
       #graph = self.node_graph 
        
        for node in trX_raw: 
            trX[node] = {}
        for node in trX_raw: 
            t = node_to_type[node]
            N, T= trX_raw[node].shape
            edges = type_to_edge_connections[t]
            edge_input_values = {}
            edge_counts = {}
            edge_input_values[str(t) + '_' + 'input'] = trX_raw[node]
            # for now, just add the features from neighbors of the same
            # type to form edge_features. One can also add in correlation
            # ... later. 
            for edge in edges: 
                if edge.split('_')[-1] != 'input': 
                    edge_input_values[edge] = np.zeros(shape=[N, T],dtype=np.float32)
                    edge_input_values[edge][:, :external_dims] = trX_raw[node][:, :external_dims]
            nbs = graph[node]
            for nb in nbs:
                if type(nb) is dict: 
                    weights = nbs[nb]
                else:
                    weights = 1.0
                t2 = node_to_type[nb]
                ind1 = min(t, t2)
                ind2 = max(t, t2)
                tok = str(ind1) + '_' + str(ind2)
                edge_input_values[tok][:, external_dims:] += (trX_raw[nb][:, external_dims:] * weights)
            trX[node] = edge_input_values 
        self.trX=trX
        self.trY=trY
        return trX, trY
    def load_graph(self, graph):
        self.node_graph=graph
        return 
    def GenerateNodeIndex(self): 
        return {ind: ind for ind in xrange(51)}
    
if __name__=='__main__':
    dataReader=DataReaderPowergrid(samples=800,predictions=200)
    trX_raw, trY_raw=dataReader.GenerateNodeFeatures()
    node_to_type=dataReader.GenerateNodeClassification1()
    graph=dataReader.GenerateGraph()
    #print(graph)
               
        
