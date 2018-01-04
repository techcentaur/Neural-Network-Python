import numpy as np 
import random

def stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta, test_data=None):
	"""Here, we train NN using mini-batch SGD."""

	if test_data:
		n_test = len(test_data)

	n_train = len(training_data)

	for i in xrange(epochs):
		random.shuffle(training_data)
		mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n, mini_batch_size)]

		for mini_batch in mini_batches:
			update_mini_batch(mini_batch,eta)