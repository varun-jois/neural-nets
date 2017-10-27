"""Base class for a neural network."""
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
    return (np.exp(array) - np.exp(-array))/(np.exp(array) + np.exp(-array))


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


def get_normalisation_constants(array):
    """
    :param array: A numpy array.
    :return: The constants mean and variance for normalising the train and test data.
    """
    m = array.shape[1]
    mean = np.sum(array) / m
    variance = np.sum(array ** 2) / m
    return mean, variance


def forward_prop(input, parameters, activation, functions_dict):
    """
    Performs the forward propagation step.
    :param input: A numpy array of shape (n,m) where n is the number of input units and m is the number of training
                    examples. The data to be trained on.
    :param parameters: The dict containing the weights.
    :param activation: A string describing the activation function to be used for the hidden units. Options are:
                        1) sigmoid - The sigmoid activation function.
                        2) tanh - The tanh activation function.
                        3) relu - The rectified linear unit activation function.
                        4) leaky_relu - The leaky rectified linear unit activation function.
    :param functions_dict: A dict with mapping of all the activation functions.
    :return: A numpy array of shape (nl,m) where nl is the number of output units and m is the number of
                        training examples. The predicted probabilities. Also the Z and A parameters in a dict.
    """
    output = {}
    layer_sizes = parameters['layer_sizes']
    output['Z1'] = parameters['W1'].dot(input) + parameters['B1']
    output['A1'] = functions_dict[activation](output['Z1'])
    for l in range(len(layer_sizes) - 2):
        output['Z' + str(l + 2)] = parameters['W' + str(l + 2)].dot(output['A' + str(l + 1)]) + parameters['W' + str(l + 2)]
        output['A' + str(l + 2)] = functions_dict[activation](output['Z' + str(l + 2)])
    output['Z' + str(len(layer_sizes))] = parameters['W' + str(len(layer_sizes))].dot(output['A' + str(len(layer_sizes) - 1)]) + \
                                          parameters['W' + str(len(layer_sizes))]
    if layer_sizes[-1] == 1:
        output['A' + str(len(layer_sizes))] = functions_dict['sigmoid'](output['Z' + str(len(layer_sizes))])
    else:
        output['A' + str(len(layer_sizes))] = functions_dict['softmax'](output['Z' + str(len(layer_sizes))])
    return output['A' + str(len(layer_sizes))], output


def compute_cost(labels, Yhat, parameters, lambd=0):
    """
    Computes the cost after forward prop.
    :param labels: A numpy array of shape (nl,m) where nl is the number of output units and m is the number of
                        training examples. The actual labels of the training data.
    :param Yhat: A numpy array of shape (nl,m) where nl is the number of output units and m is the number of
                        training examples. The predicted probabilities.
    :param lambd: A non-negative real number. The regularisation parameter.
    :return: The cost
    """
    m = labels.shape[1]
    cost = -np.sum(labels * np.log(Yhat) + (1 - labels) * np.log(1 - Yhat)) / m + \
           lambd * sum(np.sum(parameters['W' + str(l + 1)] ** 2) for l in range(len(parameters['layer_sizes'])))/(2 * m)
    return cost


class NeuralNet(object):
    """
    A NeuralNet stores the weights of the model as well as the attributes used to train them.
    """

    def __init__(self):
        self.parameters = {}
        self.hyper_parameters = {}
        self.functions_dict = {
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


    def initialize_weights(self, input_layer_size, layer_sizes, algorithm='relu_specific'):
        """
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
        :return: Initialized weights in the parameter dict.
        """
        self.parameters['input_layer_size'] = input_layer_size
        self.parameters['layer_sizes'] = layer_sizes
        self.parameters['initialization_algorithm'] = algorithm

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


    def train(self, input, labels, epochs, alpha=0.01, activation='relu', lambd=0, dropout_prob=0,
              mini_batch_size=None, beta1=None, beta2=None):
        """
        Performs forward and back propagation steps to optimize the weights contained in parameters.
        :param input: A numpy array of shape (n,m) where n is the number of input units and m is the number of training
                        examples. The data to be trained on.
        :param labels: A numpy array of shape (nl,m) where nl is the number of output units and m is the number of
                        training examples. The actual labels of the training data.
        :param epochs: An int > 0. The number of times to train on the data.
        :param alpha: A non-negative real number greater than zero. The learning rate.
        :param activation: A string describing the activation function to be used for the hidden units. Options are:
                            1) sigmoid - The sigmoid activation function.
                            2) tanh - The tanh activation function.
                            3) relu - The rectified linear unit activation function.
                            4) leaky_relu - The leaky rectified linear unit activation function.
        :param lambd: A non-negative real number. The regularisation parameter.
        :param dropout_prob: A number between [0,1) The probability of a hidden layer unit to be dropped while training.
        :param mini_batch_size: An int. The size of mini batch to use. If None then batch gradient descent is performed.
        :param beta1: A non-negative real number. The beta1 parameter of the RMSProp algorithm. If beta1 is None then
                        then gradients are not computed with an exponetially weighted average. If beta1 and beta2 are
                        present then beta1 is the parameter for the Adam optimizer. Recommended value is 0.9.
        :param beta2: A non-negative real number. The beta2 parameter of the Adam algorithm. If beta1 and beta2 are
                        None then then gradients are not computed with an exponetially weighted average.
                        Recommended value is 0.99.
        :return: Updates the weights in the parameters dict.
        """
        if mini_batch_size is None:
