import numpy as np
import random


# General layer
class Layer:

	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):

		pass


	def backward(self, output_grad, learning_rate):

		pass


# Dense Layer (or Fully Connected Layer)

class Dense(Layer):

	def __init__(self, input_size, output_size):
		"""
		Initialize weights and biases
		We are sampling from standard normal dist
		Might possibly want to normalize 
		by sqrt(n) so as to not have
		slow learning of params
		if using activation functions
		with low derivatives for
		"a" close to 1 or 0
		"""
		self.weights = np.random.randn(output_size, input_size)
		self.bias = np.random.randn(output_size, 1)
		

	def forward(self, input):
		self.input = input 

		return np.dot(self.weights, self.input) + self.bias	

	def backward(self, output_grad, learning_rate):
		weights_grad = np.dot(output_grad, self.input.T)
		input_grad = np.dot(self.weights.T, output_grad)
		# Now update weights and biases
		self.weights -= learning_rate * weights_grad
		self.bias -= learning_rate * output_grad

		return np.dot(self.weights.T, output_grad)


# Lets define an Activation Layer

class Activation(Layer):

	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward(self, input):
		self.input = input 

		return self.activation(self.input)

	def backward(self, output_grad, learning_rate):

		return np.multiply(output_grad, self.activation_prime(self.input))


class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			return 1/(1 + np.exp(-x))

		def sigmoid_prime(x):
			sig = sigmoid(x)
			return sig * (1 - sig)

		super().__init__(sigmoid, sigmoid_prime)



def binary_cross_entropy(y_true, y_pred):
	return -np.mean(np.nan_to_num(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)))

def binary_cross_entropy_prime(y_true, y_pred):
	return ((1 - y_true)/(1 - y_pred) - y_true/y_pred) / np.size(y_true)


def predict(network, input):
	output = input 
	for layer in network:
		output = layer.forward(output)

	return output

def train(network, loss, loss_prime, x_train, y_train,
	epochs = 10, learning_rate = 0.1, verbose = True):

	for e in range(epochs):
		err = 0
		for x,y in zip(x_train,y_train):
			# Do forward
			output = predict(network, x)
        	
        	# error (Useless, just to have an idea
        	# of it on screen)
			err += loss(y, output)

        	# Do backward
			grad = loss_prime(y, output)
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)
        #Average error
		err /= len(x_train)

		if verbose:
			print(f"{e + 1}/{epochs}, error = {err}")
