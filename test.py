import numpy as np
from nn import Feed_Forward

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

input_size = 2
output_size = 2
hidden_size = 5
net = Feed_Forward(input_size, output_size, hidden_size=hidden_size)
net.train_fullBatch(xor_inputs, xor_outputs)

print(net.predict(xor_inputs).T)
