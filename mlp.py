import numpy as np
import matplotlib.pyplot
from scipy import special

#neural network skeleton looks like:
##  initialize: set the number of input, hidden, and output nodes
##  train: fine-tune the weights using feedforward and backpropagation methods
##  extract: pull out information from the NN object for analysis

class NN:
"""
Neural network object that takes in user-defined variables (number of input nodes, hidden nodes, output nodes, and learning rate), 
initializes all of the weights, and trains the network using feedforward and backprop using gradient descent. The third 
function queries the NN instance for useful output information.
"""
    # initialize all the elements of the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
    # weights are randomly choosen using a normal distribution of 0 with a standard deviation raising the number of input nodes to the power of -0.5
        self.input_hidden_weights = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.hidden_output_weights = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
    # set learning rate
        self.lr = learningrate
    # grab the sigmoid function from the scipy package
        self.activation_function = lambda x: special.expit(x)

        pass
    
    # define function that trains the network
    ## first forward propagation using sigmoid function and dot product matrix multiplication
    ## second backpropagation using gradient descent
    def train(self, inputs_list, targets_list):

        # manipulate inputs into a 2d array 
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # find the input to hidden dot product
        hidden_inputs = np.dot(self.input_hidden_weights, inputs)
        # calc hidden activations
        hidden_outputs = self.activation_function(hidden_inputs)
        # find the hidden to output dot product
        final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
        # calc output activations
        final_outputs = self.activation_function(final_inputs)
        # calc error terms
        output_errors = targets - final_outputs
        # hidden to output deltas
        hidden_errors = np.dot(self.hidden_output_weights.T, output_errors)

        # backpropagating to update weights from the output error
        ## output to hidden
        self.hidden_output_weights += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
            np.transpose(hidden_outputs))
        ## hidden to input
        self.input_hidden_weights += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs))

        pass
    # extract function gets information out of the NN instance to be used on the test set. 
    def extract(self, inputs_list):
        # convert inputs to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        # find the input to hidden dot product
        hidden_inputs = np.dot(self.input_hidden_weights, inputs)
        # calc hidden activations
        hidden_outputs = self.activation_function(hidden_inputs)
        # find the hidden to output dot product
        final_inputs = np.dot(self.hidden_output_weights, hidden_outputs)
        # calc output activations
        final_outputs = self.activation_function(final_inputs)
        # return output
        return final_outputs

# MNIST netword parameters
input_nodes = 784
hidden_nodes = 20
output_nodes = 10
learning_rate = 0.1

# create instance
n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

# read training data in
train_data_file = open("mnist_train.csv", 'r')
train_data = train_data_file.readlines()
train_data_file.close()

# train for x number of epochs
epochs = 50
for e in range(epochs):
    for record in train_data:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass
# read in test data
test_data_file = open("mnist_test.csv", 'r')
test_data = test_data_file.readlines()
test_data_file.close()

# create a performance scorecard array
scorecard = []

# use extract function from NN class to get results and then print them into the scorecard
for record in test_data:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.extract(inputs)
    label = np.argmax(outputs)
    
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
   
