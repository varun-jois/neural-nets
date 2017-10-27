'''Base class for a neural network.'''
import numpy as np


# Some helper functions
def sigmoid(array):
    '''
    The sigmoid activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return 1 / (1 + np.exp(-array))


def sigmoid_prime(array):
    '''
    The derivative for the sigmoid activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return sigmoid(array) * (1 - sigmoid(array))


def tanh(array):
    '''
    The tanh activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return (np.exp(array) - np.exp(-array))/(np.exp(array) + np.exp(-array))


def tanh_prime(array):
    '''
    The derivative for the tanh activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return 1 - tanh(array) ** 2


def relu(array):
    '''
    The rectified linear unit activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return np.where(array > 0, array, 0)


def relu_prime(array):
    '''
    The derivative for the rectified linear unit activation function.
    :param array: A numpy array.
    :return: A numpy array.
    '''
    return np.where(array > 0, 1, 0)


def leaky_relu(array, leak=0.01):
    '''
    The leaky rectified linear unit activation function.
    :param array: A numpy array.
    :param leak: The leak constant for negative numbers.
    :return: A numpy array.
    '''
    return np.where(array > 0, array, leak*array)


def leaky_relu_prime(array, leak=0.01):
    '''
    The leaky rectified linear unit activation function.
    :param array: A numpy array.
    :param leak: The leak constant for negative numbers.
    :return: A numpy array.
    '''
    return np.where(array > 0, 1, leak)


class NeuralNet(object):
    '''
    A NeuralNet stores the weights of the model as well as the attributes used to train them.
    '''

    def __init__(self):
        self.parameters = {}
        self.hyper_parameters = {}
        self.epochs_completed = 0


    #def initialize_weights(self, layer_sizes, activation='relu', algorithm='random'):
