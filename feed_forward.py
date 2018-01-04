import numpy as np

def feed_forward(biases, weights, a):
	"""return the output of the network"""

	for b,w in zip(biases, weights):
		a = sigmoid(np.dot(w,a)+b)
	return a