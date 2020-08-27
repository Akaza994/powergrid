# srnn_pytorch
a pytorch version of Structured Graph RNN that is equivalent to the tensorflow one used in https://arxiv.org/abs/1902.05113.
I think one can understand the architecture of SRNN easily (It is probably more readable than the tensorflow one).


Why pytorch:
 1. It is easy to apply custom rnn cells and custom optimizers
 2. It is easy to apply regularization and quantization methods, epscially ones that thresholding or/and projection invloved.


SRNN partitions nodes into m, a parameter you can set, classes. Nodes in the same class will share the same Edge RNNs and Node RNN.
Three types of clustering method are implemented here, and a dict that stores the class label of each node will be returned:
 1. Each node is a class itself: DataReader.GenerateNodeClassification1()
 2. Kmeans clustering based on historical obsevations: DataReader.GenerateNodeClassification1(num_groups)
 3. Based the average value of historical obsevations: DataReader.GenerateNodeClassification2(num_groups)
 
 
Function that read the adjacency matrix and return a dict that contains the edge weights:
 DataReader.GenerateGraph(dynamic)
 edge weights are set to be trainable if dynamic=True

Intialize a SRNN model 

model=SRNN()

Build model by passing the graph information. Edge RNNs and Node RNNs are generated in this step

model.build_model()

Suppose we have N data points each node for training. This model offers two training startegies: 
 1. Joint train all nodes in the same class each time. Loop through all classes and data:
    # model.joint_train(self,X_train,Y_train,Epoch,cl,milestones)
       inputs: 
       
         cl: class number
         
         X_train: a list that contains N data points, each data point is a dictionary of node features and each features
                  X_train[k][node][edge] has shape [n,samples] where n is the number of nodes in class cl.
                  
         Y_train: a list contains N data points, each data point is a dictionary of ground truth of predictions.
                  Y_train[node] has shape [n,predictions]
                  
         milestone: a scheduler will be defined for adaptive learning rate if a list is inputed
 2. Train each node one by one while the same node in each class share the same Edge RNNs and Node RNN:
    # model.per_node_joint_train(X_train,Y_train,Epoch,milestones)
      inputs:
      
         X_train: a dict that contains features of each node.
                  X_train[node][edge] has shape [N,samples]
                  
         Y_train: a dict that contains ground truth of each node.
                  Y_train[node] has shape [N. predictions]
                  
         milestone: a scheduler will be defined for adaptive learning rate if a list is inputed
         
 To see an example of running SRNN, please see the .ipynb file
