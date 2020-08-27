# srnn_pytorch
a pytorch version of Structured Graph RNN that is equivalent to the tensorflow one used in https://arxiv.org/abs/1902.05113.
I think one can understand the architecture of SRNN easily (It is probably more readable than the tensorflow one).


Why pytorch:
 1. It is easy to apply custom rnn cells  
 2. It is easy to apply regularization and quantization methods, epscially ones that thresholding or/and projection invloved.
