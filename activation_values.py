"""return the activation values in output of an input layer"""

import numpy as np

def activation_values(input_layer, biases, weights):

	for b,w in zip(biases, weights):
		act = sigmoid(np.dot(weights, input_layer)+biases)
	return act