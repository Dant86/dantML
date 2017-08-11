import numpy as np

def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def tanh(x, deriv=False):
    if deriv:
        return (1 / np.cosh(x))**2
    return np.tanh(x)

def relu(x, deriv=False):
    if deriv:
        res = np.zeros(np.shape(x))
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] > 0:
                    res[i][j] = 1
        return res
    return np.maximum(0,x)

def softmax(x, deriv=False):
    denom = sum([np.exp(logit) for logit in x])
    return np.exp(x) / denom