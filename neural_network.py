import random
import numpy as np

class Network(object):
	""" A python module to implement the stochastic gradient descent learning algorithm
		for a feedforward neural network using backpropagation."""
	def __init__(self, layers_list):
		size = len(layers_list)
		self.layers_list = layers_list

		self.weights = [np.random.randn(y,x) for x,y in zip(layers_list[:-1], layers_list[1:])]
		self.biases = [np.random.randn(x,1) for x in layers_list[1:]]

	

	def activation_values(a):
	"""return the output of the network"""

		for b,w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return a



	def stochastic_gradient_descent(training_data, epochs, mini_batch_size, eta):
		n_train = len(training_data)

		for i in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n, mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch,eta)



	def update_mini_batch(mini_batch, eta):
		back_b = [np.zeros(b.shape) for b in self.biases]
		back_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			delta_back_b, delta_back_w = self.backpropagation(x, y)

			back_b = [nb+delnb for nb,delnb in zip(back_b, delta_back_b)]
			back_w = [nw+delnw for nw, delnw in zip(back_w, delta_back_w)]

		weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights, back_w)]
		biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases, back_b)]


	def backpropagation(x,y):
		back_b = [np.zeros(b.shape) for b in self.biases]
		back_w = [np.zeros(w.shape) for w in self.weights]

		act = x
		acts = [x] 
		#storing all activations layer-by-layer

		zs=[] #store all z vectors

		for b,w in zip(self.biases,self.weights):
			z = np.dot(w, act)+b
			zs.append(z)

			act = sigmoid(z)
			acts.append(act)

		delta = self.cost_derivative(acts[-1],y) * sigmoid_prime(zs[-1])

		back_b[-1] = delta
		back_w[-1] = np.dot(delta, activations[-2].transpose())

		for t in xrange(2, self.size):
			
			z= zs[-t]
			sp = sigmoid_prime(z)

			delta = np.dot(weights[-t-1].transpose(), delta) * sp

			back_b[-1] = delta
			back_w[-1] = np.dot(delta, activations[-t-1].transpose())

		return (back_b, back_w)


	def cost_derivative(output_acts, y)
		return output_acts-y



def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

