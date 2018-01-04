"""return a tuple as the gradient for the cost func."""
import numpy as np

def backpropagation(biases, weights,x,y,num_layers):

	# back_b and back_w are layer-by-layer numpy arrays similar to biases and weights in a neural network
	back_b = [np.zeros(b.shape) for b in biases]
	back_w = [np.zeros(w.shape) for w in weights]

	act = x
	acts = [x] 
	#storing all activations layer-by-layer

	zs=[] #store all z vectors

	for b,w in zip(biases,weights):
		z = np.dot(w, act)+b
		zs.append(z)

		act = sigmoid(z)
		acts.append(act)

	delta = cost_derivative(acts[-1],y) * sigmoid_prime(zs[-1])

	back_b[-1] = delta
	back_w[-1] = np.dot(delta, activations[-2].transpose())

	for t in xrange(2, num_layers):
		
		z= zs[-t]
		sp = sigmoid_prime(z)

		delta = np.dot(weights[-t-1].transpose(), delta) * sp

		back_b[-1] = delta
		back_w[-1] = np.dot(delta, activations[-t-1].transpose())

	return (back_b, back_w)