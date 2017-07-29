from losses import *
from activations import *
import numpy as np

class Feed_Forward:


    def __init__(self, input_size, output_size, hidden_size=100,
                 loss=crossentropy, activation=relu):
        '''
            Initializes a simple 2-layer feed forward
            neural network, given the input and output sizes
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights_1 = np.zeros((hidden_size, input_size))
        self.biases_1 = np.zeros((hidden_size, 1))
        self.weights_2 = np.zeros((hidden_size, hidden_size))
        self.biases_2 = np.zeros((hidden_size, 1))
        self.weights_3 = np.zeros((output_size, hidden_size))
        self.biases_3 = np.zeros((output_size, 1))
        self.loss = loss
        self.activation = activation


    def make_prediction(self, feature_vec):
        '''
            Given a feature vector, this function computes the
            output of the feed-forward network
        '''
        dot_1 = np.dot(self.weights_1, feature_vec)
        layer_1 = self.activation(dot_1 + self.biases_1)
        dot_2 = np.dot(self.weights_2, layer_1)
        layer_2 = self.activation(dot_2 + self.biases_2)
        dot_3 = np.dot(self.weights_3, layer_2)
        out = self.activation(dot_3 + self.biases_3)
        return softmax(out)


    def calculate_loss(self, logits, labels):
        '''
            Given a set of logits produced by the feed-forward
            network and, depending on the loss type, either an
            index or a vector of gold labels
        '''
        return self.loss(logits, labels)
