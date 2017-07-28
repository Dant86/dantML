import numpy as np

def sigmoid(x):
	return 1 / 1 + np.exp(-x)

def tanh(x):
	return np.tanh(x)

def relu(x):
	return np.max(0,x)

def softmax(x):
	denom = sum([np.exp(logit) for logit in x])
	return np.exp(x) / denom