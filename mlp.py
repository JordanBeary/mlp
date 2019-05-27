import numpy as np
import matplotlib.pyplot
from scipy import special


#neural network framing looks like:
# - initialize: set the number of input, hidden, and output nodes
# - train: fine-tune the weights using feedforward and backpropagation methods
# - extract: pull out information from the trained NN 

class NN:

	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		self.input_hidden_weights = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
		self.hidden_output_weights = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

		self.lr = learningrate

		self.activation_function = lambda x: scipy.special.expit(x)

		pass

	def train(self, inputs_list, targets_list):

		
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		hidden_inputs = np.dot(self.input_hidden_weights, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)

		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs

		hidden_errors = np.dot(self.hidden_output_weights.T, output_errors)

		#backpropagating errors with matrix multiplication
		self.hidden_output_weights += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
			np.transpose(hidden_outputs))

		self.input_hidden_weights += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
			np.transpose(inputs))

		pass

	def extract(self, inputs_list):
		inputs = np.array(inputs_list, ndmin = 2).T

		hidden_inputs = np.dot(self.input_hidden_weights, inputs)

		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)

		final_outputs = self.activation_function(final_inputs)

		return final_outputs

input_nodes = 784
hidden_nodes = 20
output_nodes = 10
learning_rate = 0.1

n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_data_file = open("mnist_train.csv", 'r')
train_data = train_data_file.readlines()
train_data_file.close()

for record in train_data:
	all_values = record.split(',')
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	targets = np.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99
	n.train(inputs, targets)
	pass

#test_data = np.loadtxt('mnist_test.csv', dtype = str, delimiter = ',')

