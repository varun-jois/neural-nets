import copy
import numpy as np
from scipy.stats import truncnorm


# The NeuralNet Class
class NeuralNet(object):
    def __init__(self):
        self.parameters = {}
        self.hyper_parameters = {}
        self.details = {}
        self.input_layer_size = None
        self.layers = None
        self.train_size = None
        self.layer_sizes = None
        self.initializer = None
        self.activation = None
        self.epochs = 0

    def _sigmoid(self, array):
        """
        The sigmoid activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return 1 / (1 + np.exp(-array))

    def _sigmoid_prime(self, array):
        """
        The derivative for the sigmoid activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return self._sigmoid(array) * (1 - self._sigmoid(array))

    def _tanh(self, array):
        """
        The tanh activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return np.tanh(array)

    def _tanh_prime(self, array):
        """
        The derivative for the tanh activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return 1 - np.tanh(array) ** 2

    def _relu(self, array):
        """
        The rectified linear unit activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return np.where(array > 0, array, 0)

    def _relu_prime(self, array):
        """
        The derivative for the rectified linear unit activation function.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return np.where(array > 0, 1, 0)

    def _leaky_relu(self, array, leak=0.01):
        """
        The leaky rectified linear unit activation function.
        :param array: A numpy array.
        :param leak: The leak constant for negative numbers.
        :return: A numpy array.
        """
        return np.where(array > 0, array, leak*array)

    def _leaky_relu_prime(self, array, leak=0.01):
        """
        The leaky rectified linear unit activation function.
        :param array: A numpy array.
        :param leak: The leak constant for negative numbers.
        :return: A numpy array.
        """
        return np.where(array > 0, 1, leak)

    def _softmax(self, array):
        """
        The sofmax function in case of multi-class classification.
        :param array: A numpy array.
        :return: A numpy array.
        """
        return np.exp(array)/np.sum(np.exp(array))

    def _function(self, activation, prime=False):
        functions_dict = {
            'sigmoid': self._sigmoid,
            'sigmoid_prime': self._sigmoid_prime,
            'tanh': self._tanh,
            'tanh_prime': self._tanh_prime,
            'relu': self._relu,
            'relu_prime': self._relu_prime,
            'leaky_relu': self._leaky_relu,
            'leaky_relu_prime': self._leaky_relu_prime,
            'softmax': self._softmax
        }
        if prime is False:
            return functions_dict[activation]
        else:
            return functions_dict[activation + '_prime']

    def initializer_zero(self, input_units, layer_sizes):
        self.details['input_units'] = input_units
        self.details['layer_sizes'] = layer_sizes
        self.details['layers'] = len(layer_sizes)
        self.details['initializer'] = 'zeros'
        all_layers = [input_units] + layer_sizes
        for l in range(1, len(layer_sizes) + 1):
            self.parameters['W' + str(l)] = np.zeros((all_layers[l], all_layers[l - 1]))
            self.parameters['B' + str(l)] = np.zeros((all_layers[l], 1))
            
    def initializer_ones(self, input_units, layer_sizes):
        self.details['input_units'] = input_units
        self.details['layer_sizes'] = layer_sizes
        self.details['layers'] = len(layer_sizes)
        self.details['initializer'] = 'ones'
        all_layers = [input_units] + layer_sizes
        for l in range(1, len(layer_sizes) + 1):
            self.parameters['W' + str(l)] = np.ones((all_layers[l], all_layers[l - 1]))
            self.parameters['B' + str(l)] = np.zeros((all_layers[l], 1))

    def initializer_random(self, input_units, layer_sizes):
        self.details['input_units'] = input_units
        self.details['layer_sizes'] = layer_sizes
        self.details['layers'] = len(layer_sizes)
        self.details['initializer'] = 'random'
        all_layers = [input_units] + layer_sizes
        for l in range(1, len(layer_sizes) + 1):
            self.parameters['W' + str(l)] = np.random.randn(all_layers[l], all_layers[l - 1])
            self.parameters['B' + str(l)] = np.zeros((all_layers[l], 1))

    def initializer_he(self, input_units, layer_sizes):
        self.details['input_units'] = input_units
        self.details['layer_sizes'] = layer_sizes
        self.details['layers'] = len(layer_sizes)
        self.details['initializer'] = 'he'
        all_layers = [input_units] + layer_sizes
        for l in range(1, len(layer_sizes) + 1):
            self.parameters['W' + str(l)] = truncnorm(a=-2, b=2, loc=0,
                scale=np.sqrt(2 / all_layers[l - 1])).rvs(size=(all_layers[l], all_layers[l - 1]))
            self.parameters['B' + str(l)] = np.zeros((all_layers[l], 1))
    
    def initializer_xavier(self, input_units, layer_sizes):
        self.details['input_units'] = input_units
        self.details['layer_sizes'] = layer_sizes
        self.details['layers'] = len(layer_sizes)
        self.details['initializer'] = 'xavier'
        all_layers = [input_units] + layer_sizes
        for l in range(1, len(layer_sizes) + 1):
            self.parameters['W' + str(l)] = truncnorm(a=-2, b=2, loc=0,
                scale=np.sqrt(2 / (all_layers[l - 1] + all_layers[l]))).rvs(size=(all_layers[l], all_layers[l - 1]))
            self.parameters['B' + str(l)] = np.zeros((all_layers[l], 1))

    def _forward_prop(self, x, activation='tanh', dropout_keep_prob=1):
        parameters = self.parameters
        f = self._function
        layers = self.details['layers']
        layer_sizes = self.details['layer_sizes']
        parameters['Z1'] = parameters['W1'].dot(x) + parameters['B1']
        if layers > 1:
            parameters['A1'] = f(activation)(parameters['Z1'])
            parameters['M1'] = (np.random.rand(*parameters['A1'].shape) < dropout_keep_prob) / dropout_keep_prob
            parameters['A1'] *= parameters['M1']
            for l in range(layers - 2):
                pl = str(l + 1)
                cl = str(l + 2)
                parameters['Z' + cl] = parameters['W' + cl].dot(parameters['A' + pl]) + \
                    parameters['B' + cl]
                parameters['A' + cl] = f(activation)(parameters['Z' + cl])

                parameters['M' + cl] = (np.random.rand(
                    *parameters['A' + cl].shape) < dropout_keep_prob) / dropout_keep_prob
                parameters['A' + cl] *= parameters['M' + cl]

            parameters['Z' + str(layers)] = \
                parameters['W' + str(layers)].dot(parameters['A' + str(layers - 1)]) + \
                parameters['B' + str(layers)]
        if layer_sizes[-1] == 1:
            parameters['A' + str(layers)] = f('sigmoid')(parameters['Z' + str(layers)])
        else:
            parameters['A' + str(layers)] = f('softmax')(parameters['Z' + str(layers)])
