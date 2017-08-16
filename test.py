import numpy as np
from nn import Feed_Forward

'''
	This test program exists only for informative purposes.
	It's a step-by-step explanation on how to make the most
	of dantML's features.
'''

'''
	These lines here are reserved for formatting input and
	output data. Input and output data should have shape
	(num_features, num_examples).
'''
xor_inputs = []
xor_inputs.append(np.array([1, 0]))
xor_inputs.append(np.array([1, 1]))
xor_inputs.append(np.array([0, 0]))
xor_inputs.append(np.array([0, 1]))
xor_inputs = np.array(xor_inputs).T
m = len(xor_inputs[0])
xor_outputs = []
xor_outputs.append(np.array([0, 1]))
xor_outputs.append(np.array([1, 0]))
xor_outputs.append(np.array([1, 0]))
xor_outputs.append(np.array([0, 1]))
xor_outputs = np.array(xor_outputs).T

# Standard parameters for the neural net
input_size = 2
output_size = 2
hidden_size = 5

# This initializes a neural network. Very simple.
# Note that the default hidden size is 100 neurons.
net = Feed_Forward(input_size, output_size, hidden_size=hidden_size)
net.Adam(xor_inputs, xor_outputs)
print(net.predict(xor_inputs).T)
