"""Base class for a neural network."""
import numpy as np
import copy


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


def forward_prop(X, parameters, activation, layer_sizes):
    params = {}
    ls = len(layer_sizes)
    params['Z1'] = parameters['W1'].dot(X) + parameters['B1']
    params['A1'] = function(activation)(params['Z1'])
    for l in range(ls - 2):
        pl = str(l + 1)
        cl = str(l + 2)
        params['Z' + cl] = parameters['W' + cl].dot(params['A' + pl]) + parameters['B' + cl]
        params['A' + cl] = function(activation)(params['Z' + cl])

    params['Z' + str(ls)] = parameters['W' + str(ls)].dot(params['A' + str(ls - 1)]) + parameters['B' + str(ls)]
    if layer_sizes[-1] == 1:
        params['A' + str(ls)] = function('sigmoid')(params['Z' + str(ls)])
    else:
        params['A' + str(ls)] = function('softmax')(params['Z' + str(ls)])
    return params


def compute_cost(Y, Yhat, parameters, layers, lambd=0):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat)) / m + \
            lambd * sum(np.sum(parameters['W' + str(l + 1)] ** 2) for l in range(layers)) / (2 * m)
    return cost


def back_prop(X, Y, parameters, activation, layer_sizes):
    params = {}
    m = X.shape[1]
    ls = len(layer_sizes)
    if layer_sizes[-1] == 1:
        params['dZ' + str(ls)] = parameters['A' + str(ls)] - Y

    for l in reversed(range(ls - 1)):
        params['dW' + str(l + 2)] = params['dZ' + str(l + 2)].dot(parameters['A' + str(l + 1)].T) / m
        params['dB' + str(l + 2)] = np.sum(params['dZ' + str(l + 2)], axis=1, keepdims=True) / m
        params['dZ' + str(l + 1)] = params['dW' + str(l + 2)].T.dot(params['dZ' + str(l + 2)]) * \
                                    function(activation, prime=True)(parameters['Z' + str(l + 1)])
    params['dW1'] = params['dZ1'].dot(X.T) / m
    params['dB1'] = np.sum(params['dZ1'], axis=1, keepdims=True) / m
    return params


def update_weights(parameters, layers, alpha=0.01):
    params = {}
    for l in range(layers):
        params['W' + str(l + 1)] = parameters['W' + str(l + 1)] - alpha * parameters['dW' + str(l + 1)]
        params['B' + str(l + 1)] = parameters['B' + str(l + 1)] - alpha * parameters['dB' + str(l + 1)]
    return params


def num_grad(X, Y, parameters, activation, layer_sizes, lambd=0):
    layers = len(layer_sizes)
    grad_num = []
    eps = 1e-4
    for l in range(1, len(layer_sizes) + 1):
        for r in range(parameters['W' + str(l)].shape[0]):
            for c in range(parameters['W' + str(l)].shape[1]):
                params = copy.deepcopy(parameters)
                params['W' + str(l)][r, c] = params['W' + str(l)][r, c] + eps
                fparams = forward_prop(X, params, activation, layer_sizes)
                params.update(fparams)
                Yhat = params['A' + str(len(layer_sizes))]
                costp = compute_cost(Y, Yhat, params, layers, lambd=lambd)

                params = copy.deepcopy(parameters)
                params['W' + str(l)][r, c] = params['W' + str(l)][r, c] - eps
                fparams = forward_prop(X, params, activation, layer_sizes)
                params.update(fparams)
                Yhat = params['A' + str(len(layer_sizes))]
                costn = compute_cost(Y, Yhat, params, layers, lambd=lambd)
                #print((costp - costn) / (2 * eps))
                #print('costp is {0} and costn is {1}'.format(costp, costn))
                grad_num.append((costp - costn) / (2 * eps))

        for r in range(parameters['B' + str(l)].shape[0]):
            params = copy.deepcopy(parameters)
            params['B' + str(l)][r, 0] = params['B' + str(l)][r, 0] + eps
            fparams = forward_prop(X, params, activation, layer_sizes)
            fparams.update(params)
            Yhat = fparams['A' + str(len(layer_sizes))]
            costp = compute_cost(Y, Yhat, fparams, layers, lambd=lambd)

            params = copy.deepcopy(parameters)
            params['W' + str(l)][r, 0] = params['W' + str(l)][r, 0] - eps
            fparams = forward_prop(X, params, activation, layer_sizes)
            fparams.update(params)
            Yhat = fparams['A' + str(len(layer_sizes))]
            costn = compute_cost(Y, Yhat, fparams, layers, lambd=lambd)
            grad_num.append((costp - costn) / (2 * eps))

    return np.array(grad_num).reshape(1, -1)


def num_grad(X, Y, parameters, activation, layer_sizes, lambd=0):
    layers = len(layer_sizes)
    eps = 1e-4
    wts = np.array([]).reshape(1, 0)
    for l in range(layers):
        wts = np.concatenate((wts, parameters['W' + str(l + 1)].reshape(1, -1)))
        wts = np.concatenate((wts, parameters['B' + str(l + 1)].reshape(1, -1)))
    grads_num = np.zeros(wts.shape)
    for i in wts:





# The NeuralNet Class
class NeuralNet(object):
    """
    A NeuralNet stores the weights of the model as well as the attributes used to train them.
    """

    def __init__(self):
        self.parameters = {}
        self.hyper_parameters = {}
        self.input_layer_size = None
        self.layers = None
        self.layer_sizes = None
        self.initializer = None
        self.activation = None
        self.epochs = 0


    def initialize_weights(self, input_layer_size, layer_sizes, algorithm='relu_specific'):
        """
        This function initializes all weights of the network with respect to the
        algorithm passed.
        :param input_layer_size: An int specifying the number of X units.
        :param layer_sizes: A list of layer sizes (excluding the X layer).
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
        self.input_layer_size = input_layer_size
        self.layer_sizes = layer_sizes
        self.initializer = algorithm
        self.layers = len(layer_sizes)

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


    def train(self, X, Y, alpha=0.01, activation='relu', lambd=0, epochs=1000, dropout_prob=0,
              mini_batch_size=None, beta1=None, beta2=None):
        """
        Performs forward and back propagation steps to optimize the weights contained in parameters.
        :param X: A numpy array of shape (n,m) where n is the number of X units and m is the number of training
                        examples. The data to be trained on.
        :param Y: A numpy array of shape (nl,m) where nl is the number of output units and m is the number of
                        training examples. The actual Y of the training data.
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
        self.activation = activation
        self.hyper_parameters['alpha'] = alpha
        self.hyper_parameters['lambd'] = lambd
        self.epochs += epochs

        for i in range(epochs):
            # Forward prop
            params = forward_prop(X, self.parameters, activation, self.layer_sizes)
            self.parameters.update(params)
            Yhat = self.parameters['A' + str(self.layers)]

            # Compute cost
            cost = compute_cost(Y, Yhat, self.parameters, self.layers, lambd=lambd)
            # if (i + 1) % 100 == 0:
            #     print('The cost after {0} rounds is {1}.'.format(i + 1, cost))

            # Back prop
            params = back_prop(X, Y, self.parameters, activation, self.layer_sizes)
            self.parameters.update(params)

            # Update the weights
            params = update_weights(self.parameters, self.layers, alpha=alpha)
            self.parameters.update(params)

            if i < 10:
                grads_num = \
                    num_grad(X, Y, self.parameters, activation, self.layer_sizes, lambd=lambd)
                grads = np.array([]).reshape(1, 0)
                for l in range(self.layers):
                    grads = np.concatenate((grads, self.parameters['dW' + str(l + 1)].reshape(1, -1)), axis=1)
                    grads = np.concatenate((grads, self.parameters['dB' + str(l + 1)].reshape(1, -1)), axis=1)
                numerator = np.linalg.norm(grads_num - grads)
                denominator = np.linalg.norm(grads_num) + np.linalg.norm(grads)
                # print(grads_num)
                # print(grads)
                #print('Shapes of grads and grads_num are {0} and {1}'.format(grads.shape, grads_num.shape))
                print(np.concatenate((grads.T, grads_num.T), axis=1))
                #print('The grad check ratio is {0}. {1} / {2}'.format(numerator / denominator, numerator, denominator))


    def predict(self, X_test, Y_test=None, cut_off=0.5):
        """
        Make predictions based on a test dataset.
        :param X_test: A numpy array. Data to run model on.
        :param Y_test: A numpy array. Labels of the data if available.
        :param cut_off: A number between [0,1]. The number above which the label is 1.
        :return: A tuple, numpy array of predictions and error if Y_test is not none.
        """
        m = X_test.shape[1]
        params = forward_prop(X_test, self.parameters, self.activation, self.layer_sizes)
        Yhat = params['A' + str(self.layers)]
        predictions = np.where(Yhat > cut_off, 1, 0)

        # Calculating the error
        if Y_test is not None:
            error = np.sum(np.absolute(Y_test - predictions)) / m

        return predictions, error
