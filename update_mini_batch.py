import numpy as np 

def update_mini_batch(mini_batch, eta, biases, weights):
	"""update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch"""

	back_b = [np.zeros(b.shape) for b in biases]
	back_w = [np.zeros(w.shape) for w in weights]

	for x,y in mini_batch:
		delta_back_b, delta_back_w = backpropagation(x, y)

		back_b = [nb+delnb for nb,delnb in zip(back_b, delta_back_b)]
		back_w = [nw+delnw for nw, delnw in zip(back_w, delta_back_w)]

	weights = [w-(eta/len(mini_batch))*nw for w,nw in zip(weights, back_w)]
	biases = [b-(eta/len(mini_batch))*nb for b,nb in zip(biases, back_b)]

