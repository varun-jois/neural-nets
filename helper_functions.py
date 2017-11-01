import numpy as np


# Some helper functions
def sigmoid(array):
    """
    The sigmoid activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return 1 / (1 + np.exp(-array))


def sigmoid_prime(array):
    """
    The derivative for the sigmoid activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return sigmoid(array) * (1 - sigmoid(array))


def tanh(array):
    """
    The tanh activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return np.tanh(array)


def tanh_prime(array):
    """
    The derivative for the tanh activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return 1 - tanh(array) ** 2


def relu(array):
    """
    The rectified linear unit activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return np.where(array > 0, array, 0)


def relu_prime(array):
    """
    The derivative for the rectified linear unit activation function.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return np.where(array > 0, 1, 0)


def leaky_relu(array, leak=0.01):
    """
    The leaky rectified linear unit activation function.
    :param array: A numpy array.
    :param leak: The leak constant for negative numbers.
    :return: A numpy array.
    """
    return np.where(array > 0, array, leak*array)


def leaky_relu_prime(array, leak=0.01):
    """
    The leaky rectified linear unit activation function.
    :param array: A numpy array.
    :param leak: The leak constant for negative numbers.
    :return: A numpy array.
    """
    return np.where(array > 0, 1, leak)


def softmax(array):
    """
    The sofmax function in case of multi-class classification.
    :param array: A numpy array.
    :return: A numpy array.
    """
    return np.exp(array)/np.sum(np.exp(array))


def function(activation, prime=False):
    functions_dict = {
        'sigmoid': sigmoid,
        'sigmoid_prime': sigmoid_prime,
        'tanh': tanh,
        'tanh_prime': tanh_prime,
        'relu': relu,
        'relu_prime': relu_prime,
        'leaky_relu': leaky_relu,
        'leaky_relu_prime': leaky_relu_prime,
        'softmax': softmax
    }
    if prime is False:
        return functions_dict[activation]
    else:
        return functions_dict[activation + '_prime']


def get_normalisation_constants(array):
    """
    :param array: A numpy array.
    :return: The constants mean and variance for normalising the train and test data.
    """
    m = array.shape[1]
    mean = np.sum(array) / m
    variance = np.sum(array ** 2) / m
    return mean, variance
