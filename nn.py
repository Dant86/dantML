import numpy as np
from activations import *
from losses import *

class Feed_Forward:


    def __init__(self, in_size, out_size, hid_size, amt_hid,
                 activation=sigmoid, loss=crossentropy):
        self.in_neurons = in_size
        self.out_neurons = out_size
        self.hid_neurons = hid_size
        self.amt_hid_layers = amt_hid
        self.activation = activation
        self.loss = crossentropy
        self.parameters = {}
        self.parameters["w1"] = np.random.randn(self.hid_neurons,
                                                self.in_neurons)
        self.parameters["b1"] = np.zeros((self.hid_neurons, 1))
        for i in range(2, self.amt_hid_layers+1):
            self.parameters["w"+str(i)] = np.random.randn(self.hid_neurons,
                                                          self.hid_neurons)
            self.parameters["b"+str(i)] = np.zeros((self.hid_neurons, 1))
        self.parameters["w"+str(self.amt_hid_layers+1)] = np.random.randn(self.out_neurons,
                                                                          self.hid_neurons)
        self.parameters["b"+str(self.amt_hid_layers+1)] = np.zeros((self.out_neurons, 1))


    def get_cache(self, x_val):
        cache = {}
        cache["x"] = x_val
        cache["z1"] = self.parameters["w1"].dot(x_val) + self.parameters["b1"]
        cache["a1"] = self.activation(cache["z1"])
        for i in range(2, self.amt_hid_layers+2):
            cache["z"+str(i)] = self.parameters["w"+str(i)].dot(cache["a"+str(i-1)]) + self.parameters["b"+str(i)]
            cache["a"+str(i)] = self.activation(cache["z"+str(i)])
        return cache


    def make_prediction(self, x_val):
        cache = {}
        cache["x"] = x_val
        cache["z1"] = self.parameters["w1"].dot(x_val) + self.parameters["b1"]
        cache["a1"] = self.activation(cache["z1"])
        for i in range(2, self.amt_hid_layers+2):
            cache["z"+str(i)] = self.parameters["w"+str(i)].dot(cache["a"+str(i-1)]) + self.parameters["b"+str(i)]
            cache["a"+str(i)] = self.activation(cache["z"+str(i)])
        return cache["a"+str(self.amt_hid_layers+1)]


    def batch(self, x_vals, y_vals, batch_size):
        x_batches = []
        y_batches = []
        for i in range(0, len(x_vals[0]), batch_size):
            x_batches.append(x_vals[:,i:i+batch_size])
            y_batches.append(y_vals[:,i:i+batch_size])
        return x_batches, y_batches


    def calculate_derivatives(self, cache, y):
        m = len(y[0])
        derivs = {}
        derivs["dz"+str(self.amt_hid_layers+1)] = cache["a"+str(self.amt_hid_layers+1)] - y
        derivs["dw"+str(self.amt_hid_layers+1)] = (1 / m) * derivs["dz"+str(self.amt_hid_layers+1)].dot(cache["a"+str(self.amt_hid_layers)].T)
        derivs["db"+str(self.amt_hid_layers+1)] = (1 / m) * np.sum(derivs["dz"+str(self.amt_hid_layers+1)], axis=1, keepdims=True)
        for i in range(self.amt_hid_layers, 1, -1):
            derivs["dz"+str(i)] = self.parameters["w"+str(i+1)].T.dot(derivs["dz"+str(i+1)]) * self.activation(cache["a"+str(i)], deriv=True)
            derivs["dw"+str(i)] = (1 / m) * derivs["dz"+str(i)].dot(cache["a"+str(i-1)].T)
            derivs["db"+str(i)] = (1 / m) *  np.sum(derivs["dz"+str(i)], axis=1, keepdims=True)
        derivs["dz1"] = self.parameters["w2"].T.dot(derivs["dz2"]) * self.activation(cache["a1"], deriv=True)
        derivs["dw1"] = (1 / m) * derivs["dz1"].dot(cache["x"].T)
        derivs["db1"] = (1 / m) * np.sum(derivs["dz1"], axis=1, keepdims=True)
        return derivs


    def update_parameters(self, derivs, learn_rate):
        for i in range(1, self.amt_hid_layers+1):
            self.parameters["w"+str(i)] -= learn_rate * derivs["dw"+str(i)]
            self.parameters["b"+str(i)] -= learn_rate * derivs["db"+str(i)]


    def SGD(self, X_vals, Y_vals, epochs=90000, learn_rate=0.08, batch_size=0):
        if(batch_size == 0):
            batch_size = len(X_vals[0])
        X_batches, Y_batches = self.batch(X_vals, Y_vals, batch_size)
        for i in range(epochs):
            for X, Y in zip(X_batches, Y_batches):
                cache = self.get_cache(X)
                print(self.loss(cache["a"+str(self.amt_hid_layers+1)], Y))
                derivs = self.calculate_derivatives(cache, Y)
                self.update_parameters(derivs, learn_rate)


    def Adam(self, X_vals, Y_vals, epochs=90000, learn_rate=0.005, batch_size=0,
             beta1 = 0.9, beta2 = 0.9):
        v = {}
        s = {}
        for param in self.parameters.keys():
            v["d"+param] = np.zeros(self.parameters[param].shape)
            s["d"+param] = np.zeros(self.parameters[param].shape)
        if(batch_size == 0):
            batch_size = len(X_vals[0])
        X_batches, Y_batches = self.batch(X_vals, Y_vals, batch_size)
        for i in range(epochs):
            for X, Y in zip(X_batches, Y_batches):
                cache = self.get_cache(X)
                print(self.loss(cache["a"+str(self.amt_hid_layers+1)], Y))
                derivs = self.calculate_derivatives(cache, Y)
                for deriv in derivs.keys():
                    if "z" not in deriv:
                        v[deriv] = (beta1 * v[deriv]) + ((1 - beta1) * derivs[deriv])
                        s[deriv] = (beta2 * s[deriv]) + ((1 - beta2) * (derivs[deriv]) ** 2)
                updates = {}
                for deriv in derivs.keys():
                    if "z" not in deriv:
                        updates[deriv] = (v[deriv] / (s[deriv] ** (1 / 2)))
                self.update_parameters(updates, learn_rate)
