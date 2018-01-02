"""Initialize your neural network with layers_list,
	for e.g. [5,50,..,1] where 5 -> Input layer, 1-> Output layer and rest layers 
	are Hidden layers."""
import numpy as np
import random


def __init__(layers_list):

	size = len(layers_list)

	# biases and weights are initialized randomly using Gaussian-dist. with mean=0 and var=1.
	weights = [np.random.randn(y,x) for x,y in zip(layers_list[:-1], layers_list[1:])]
	biases = [np.random.randn(x,1) for x in layers_list[1:]]
