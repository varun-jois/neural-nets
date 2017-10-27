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


    def initialize_weights(self, input_layer_size, layer_sizes, algorithm='relu_specific'):
        '''
        This function initializes all weights of the network with respect to the
        algorithm passed.
        :param input_layer_size: An int specifying the number of input units.
        :param layer_sizes: A list of layer sizes (excluding the input layer).
        :param algorithm: A string describing the method to use to initialize the weights. Possible options are:
                1) zeros - All the weights are zero.
                2) random - The weights are random numbers taken from the standard normal distribution and
                            multiplied by 0.01.
                3) relu_specific - The weights are random numbers taken from the standard normal distribution and
                            multiplied by sqrt(2 / n) where n is the number of units in the previous layer. As
                            discussed in He et al.
                4) xavier - The weights are random numbers taken from the standard normal distribution and
                            multiplied by sqrt(1 / n) where n is the number of units in the previous layer. Good for
                            the tanh activation function. As discussed in "Understanding the difficulty of training
                            deep feedforward neural networks".
        :return:
        '''
        self.parameters['input_layer_size'] = input_layer_size
        self.parameters['layer_sizes'] = layer_sizes
        self.parameters['algorithm'] = algorithm

        if algorithm == 'zeros':
            self.parameters['W1'] = np.zeros((layer_sizes[0], input_layer_size))
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l + 2)] = np.zeros((layer_sizes[l+1], layer_sizes[l]))

        elif algorithm == 'random':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * 0.01
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l+2)] = np.random.randn(layer_sizes[l+1], layer_sizes[l]) * 0.01

        elif algorithm == 'relu_specific':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * np.sqrt(2 / input_layer_size)
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l+2)] = np.random.randn(layer_sizes[l+1], layer_sizes[l]) * \
                                                  np.sqrt(2 / layer_sizes[l])

        elif algorithm == 'xavier':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * np.sqrt(1 / input_layer_size)
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l+2)] = np.random.randn(layer_sizes[l+1], layer_sizes[l]) * \
                                                  np.sqrt(1 / layer_sizes[l])

        self.parameters['B1'] = np.zeros((layer_sizes[0], 1))
        for l in range(len(layer_sizes) - 1):
            self.parameters['B' + str(l + 2)] = np.zeros((layer_sizes[l + 1], 1))


