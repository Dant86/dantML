import numpy as np
import math

def crossentropy(yhat, y):
	m = len(y[0])
	logprobs_left = np.multiply(np.log(yhat), y)
	logprobs_right = np.multiply(np.log(1 - yhat), (1 - y))
	logprobs = logprobs_left + logprobs_right
	cost = -np.sum(logprobs) / m
	return cost

def sq_err(yhat, y):
	m = len(y[0])
	return (yhat - y) ** 2