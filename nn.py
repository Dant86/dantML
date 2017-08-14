from losses import *
from activations import *
import numpy as np


class Feed_Forward:
    '''
        Simple 2-layer neural network
        inputs and outputs must be of shape (num_features, num_examples)
    '''

    def __init__(self, input_size, output_size, hidden_size=100,
                 loss=crossentropy, activation=sigmoid):
        '''
            Initializes a simple 2-layer feed forward
            neural network, given the input and output sizes
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))
        self.loss = loss
        self.activation = activation


    def predict(self, feature_vec, train=False):
        '''
            Given a feature vector, this function computes the
            output of the feed-forward network
            feature_vec must be of size (num_features, num_examples)
        '''
        z1 = self.w1.dot(feature_vec) + self.b1
        a1 = self.activation(z1)
        z2 = self.w2.dot(a1) + self.b2
        yhat = self.activation(z2)
        return yhat


    def calculate_loss(self, logits, labels):
        '''
            Given a set of logits produced by the feed-forward
            network and, depending on the loss type, either an
            index or a vector of gold labels
        '''
        return self.loss(logits, labels)

    def batch(self, x_vals, y_vals, batch_size):
        '''
            Split list of feature vectors and labels into batches
            of relatively equal size. Last batch not guaranteed to
            be of length batch_size (probably less).
        '''
        x_batches = []
        y_batches = []
        for i in range(0, len(x_vals[0]), batch_size):
            x_batches.append(x_vals[:,i:i+batch_size])
            y_batches.append(y_vals[:,i:i+batch_size])
        return x_batches, y_batches

    def Gradient_Descent(self, X, Y, epochs=90000, learn_rate=0.005, batch_size=0):
        '''
            full batch optimization
            X must be of size (num_features, num_examples).
            Works with full batch, mini batch, and stochastic.
        '''
        if(batch_size == 0):
            batch_size = len(X)
        X_batches, Y_batches = self.batch(X, Y, batch_size)
        m = len(X[0])
        for i in range(epochs):
            for X, Y in zip(X_batches, Y_batches):
                z1 = self.w1.dot(X) + self.b1
                a1 = self.activation(z1)
                z2 = self.w2.dot(a1) + self.b2
                yhat = self.activation(z2)
                print(self.loss(yhat, Y))
                dz2 = yhat - Y
                dw2 = (1 / m) * dz2.dot(a1.T)
                db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
                dz1 = (self.w2.T).dot(dz2) * sigmoid(a1, deriv=True)
                dw1 = (1 / m) * dz1.dot(X.T)
                db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
                self.w1 -= (learn_rate * dw1)
                self.w2 -= (learn_rate * dw2)
                self.b1 -= (learn_rate * db1)
                self.b2 -= (learn_rate * db2)


    # TODO: Momentum, RMSProp, Adam optimizers
    def Gradient_Descent_with_momentum(self, X, Y, epochs=90000, learn_rate=0.005, batch_size=0):
        pass

    def RMSProp(self, X, Y, epochs=90000, learn_rate=0.005, batch_size=0):
        pass

    def Adam(self, X, Y, epochs=90000, learn_rate=0.005, batch_size=0):
        pass







