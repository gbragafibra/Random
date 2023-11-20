import numpy as np 
from scipy import signal
from More_tests import *

class Convolution(Layer):

	def __init__(self, input_shape, kernel_size, depth):
		# depth -> how many kernels do we want
		input_depth, input_height, input_width = input_shape
		self.depth = depth
		self.input_shape = input_shape
		self.input_depth = input_depth
		self.output_shape = (depth, input_height - kernel_size + 1,
			input_width - kernel_size + 1)
		self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)			
		# Initialize kernels params
		self.kernels = np.random.randn(*self.kernels_shape)
		self.biases = np.random.randn(*self.output_shape)
		# * before self, unpacks the tuple
		# that we are expecting

	def forward(self, input):
		self.input = input 
		# here we can have the outputs
		# with the bias values already
		self.output = np.copy(self.biases)
		for i in range(self.depth):
			for j in range(self.input_depth):
				# here we are using cross correlation of 2D inputs
				self.output[i] += signal.correlate2d(self.input[j],
					self.kernels[i,j], "valid")

		return self.output 

	def backward(self, output_grad, learning_rate):

		kernels_grad = np.zeros(self.kernels_shape)
		input_grad = np.zeros(self.input_shape)

		# Bias grad is just output grad
		for i in range(self.depth):
			for j in range(self.input_depth):
				kernels_grad[i,j] = signal.correlate2d(self.input[j],
					output_grad[i], "valid")
				input_grad[j] += signal.convolve2d(output_grad[i],
					self.kernels[i,j], "full")

		self.kernels -= learning_rate * kernels_grad
		self.biases -= learning_rate * output_grad

		return input_grad


# Reshape Layer
class Reshape(Layer):

	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape

	def forward(self, input):
		return np.reshape(input, self.output_shape)

	def backward(self, output_grad, learning_rate):
		return np.reshape(output_grad, self.input_shape)