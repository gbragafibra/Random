import numpy as np
import random
import time # to implement timers

class Network:

	def __init__(self, sizes):
		self.num_layers = len(sizes)#Here I will get
		# n layers, with respective amounts of neurons
		# given the example Network([1,2,5]),
		# 3 layers with 1, 2 and 5 neurons each.
		# first layer is input layer
		self.sizes = sizes

		#Initialization
		self.biases = [np.random.randn(y,1) for y 
		in sizes[1:]] #Starting with the second layer
		#No bias in the first layer

		self.weights = [np.random.randn(y,x) for
		x,y in zip(sizes[:-1], sizes[1:])] #Will give initialization of weights
		# between layers
		# Initializing weights between 0th and 1st layers, and so on..

	# compute activation of neurons
	def feed_forward(self, a):
		for b,w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w,a) + b)

		return a

	# Now unto the learning method (Stochastic Gradient Descent)

	def SGD(self, training_data, epochs, mini_batch_size,
		eta, test_data = None): 
		"""
		Eta is step-size of gradient, or learning rate
		Training_data is tuples containing training inputs, and labels
		or desired outputs
		Epochs, how much I train for
		Mini_batch_size, size of batches when sampling
		test_data optional, when provided will give
		evaluation over this test data
		"""

		if test_data:
			n_test = len(test_data)

		n = len(training_data)		
		start = time.time() #start timer
		for j in range(epochs):
			random.shuffle(training_data) #Shuffle data
			mini_batches = [
			training_data[k:k + mini_batch_size] 
			for k in range(0,n, mini_batch_size)]

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta)
			if test_data:
				print("Epoch {0}: {1} / {2}, elapsed time: {3:.2f}s".format(
					j, self.evaluate(test_data),n_test,time.time()-start))
			else:
				print("Epoch {0} complete, elapsed time: {1:.2f}s".format(j,time.time()-start))


	def update_mini_batch(self, mini_batch, eta):

		nabla_b = [np.zeros(b.shape) for b in 
		self.biases]
		nabla_w = [np.zeros(w.shape) for w in 
		self.weights]

		for x,y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb+dnb for nb,dnb in 
			zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw,dnw in 
			zip(nabla_w, delta_nabla_w)]

		self.weights = [w - (eta/len(mini_batch))*nw 
		for w, nw in zip(self.weights, nabla_w)]
		
		self.biases = [b - (eta/len(mini_batch))*nb 
		for b, nb in zip(self.biases, nabla_b)]

	def backprop(self, x, y):
		"""
		Compute gradient
		"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		activation = x
		activations = [x]

		zs = []

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(-zs[-1])
		nabla_b[-1] = delta 
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp 
			nabla_b[-l] = delta 
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)


	def cost_derivative(self, output_activations, y):
		return (output_activations - y)


	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feed_forward(x)), y) 
		for (x, y) in test_data]

		return sum(int(x == y) for x,y in test_results)

def sigmoid(z): #If z is vec, returns vectorized form aswell
	return 1/(1 + np.e**(-z))



def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))