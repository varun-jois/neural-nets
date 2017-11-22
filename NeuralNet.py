import copy
import numpy as np
from scipy.stats import truncnorm

# For normalising the input
def get_normalisation_constants(array):
    """
    :param array: A numpy array.
    :return: The constants mean and variance for normalising the train and test data.
    """
    m = array.size
    mean = np.sum(array) / m
    variance = np.sum(array ** 2) / m
    return mean, variance


# The NeuralNet Class
class NeuralNet(object):
    def __init__(self):
        self.parameters = {}
        self.hyper_parameters = {}
        self.details = {}

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
        # array_adj = array - np.max(array)
        # result = np.exp(array_adj)/np.sum(np.exp(array_adj), axis=0, keepdims=True)
        # if np.any(np.isnan(result)):
        #     raise
        mx = np.max(array)
        array_adj = array - mx
        log_term = np.sum(np.exp(array_adj), axis=0, keepdims=True)
        log_term = np.where(log_term == 0, 1e-30, log_term)
        result = np.exp(array - mx - np.log(log_term))
        return result

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

    def _compute_cost(self, y, lambd=0):
        r, m = y.shape
        layers = self.details['layers']
        parameters = self.parameters

        yhat = np.where(parameters['A' + str(layers)] == 0, 1e-100, parameters['A' + str(layers)])
        if r == 1:
            cost = -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / m + \
                    lambd * sum(np.sum(parameters['W' + str(l + 1)] ** 2) for l in range(layers)) / (2 * m)
        else:
            cost = -np.sum(y * np.log(yhat)) / m + \
                   lambd * sum(np.sum(parameters['W' + str(l + 1)] ** 2) for l in range(layers)) / (2 * m)
        return cost

    def _back_prop(self, x, y, activation):
        parameters = self.parameters
        f = self._function
        layers = self.details['layers']
        lambd = self.hyper_parameters['lambd']

        m = y.shape[1]
        parameters['dZ' + str(layers)] = parameters['A' + str(layers)] - y
        for l in reversed(range(layers - 1)):
            parameters['dW' + str(l + 2)] = \
                parameters['dZ' + str(l + 2)].dot(parameters['A' + str(l + 1)].T) / m + \
                lambd / m * parameters['W' + str(l + 2)]
            parameters['dB' + str(l + 2)] = \
                np.sum(parameters['dZ' + str(l + 2)], axis=1, keepdims=True) / m
            parameters['dZ' + str(l + 1)] = \
                parameters['W' + str(l + 2)].T.dot(parameters['dZ' + str(l + 2)]) * \
                parameters['M' + str(l + 1)] * f(activation, prime=True)(parameters['Z' + str(l + 1)])
        parameters['dW1'] = parameters['dZ1'].dot(x.T) / m + \
            lambd / m * parameters['W1']
        parameters['dB1'] = np.sum(parameters['dZ1'], axis=1, keepdims=True) / m

    def _num_grad(self, x, y, lambd, eps=1e-7):
        params = copy.deepcopy(self.parameters)
        grads_num = []
        for l in range(self.details['layers']):
            for r in range(params['W' + str(l + 1)].shape[0]):
                for c in range(params['W' + str(l + 1)].shape[1]):
                    self.parameters = copy.deepcopy(params)
                    self.parameters['W' + str(l + 1)][r, c] = self.parameters['W' + str(l + 1)][r, c] + eps
                    self._forward_prop(x, activation=self.details['activation'])
                    costp = self._compute_cost(y, lambd=lambd)
                    self.parameters = copy.deepcopy(params)
                    self.parameters['W' + str(l + 1)][r, c] = self.parameters['W' + str(l + 1)][r, c] - eps
                    self._forward_prop(x, activation=self.details['activation'])
                    costn = self._compute_cost(y, lambd=lambd)
                    grads_num.append((costp - costn) / (2 * eps))
        for l in range(self.details['layers']):
            for r in range(params['B' + str(l + 1)].shape[0]):
                for c in range(params['B' + str(l + 1)].shape[1]):
                    self.parameters = copy.deepcopy(params)
                    self.parameters['B' + str(l + 1)][r, c] = self.parameters['B' + str(l + 1)][r, c] + eps
                    self._forward_prop(x, activation=self.details['activation'])
                    costp = self._compute_cost(y, lambd=lambd)
                    self.parameters = copy.deepcopy(params)
                    self.parameters['B' + str(l + 1)][r, c] = self.parameters['B' + str(l + 1)][r, c] - eps
                    self._forward_prop(x, activation=self.details['activation'])
                    costn = self._compute_cost(y, lambd=lambd)
                    grads_num.append((costp - costn) / (2 * eps))
        self.parameters = copy.deepcopy(params)
        return np.array(grads_num).reshape(1, -1)
    
    def gd_batch(self, x, y, alpha=0.1, activation='tanh', lambd=0, epochs=1000, grad_check=False, dropout_keep_prob=1):
        self.details['activation'] = activation
        self.details['epochs'] = epochs if 'epochs' not in self.details else self.details['epochs'] + epochs
        self.details['train_size'] = x.shape[1]
        self.hyper_parameters['alpha'] = alpha
        self.hyper_parameters['lambd'] = lambd
        self.hyper_parameters['dropout_keep_prob'] = dropout_keep_prob
        layers = self.details['layers']

        for i in range(epochs):
            # Forward prop
            self._forward_prop(x, activation, dropout_keep_prob=dropout_keep_prob)

            # Compute cost
            cost = self._compute_cost(y, lambd=lambd)
            if (i + 1) % 100 == 0:
                print('The cost after epoch {0} is {1}.'.format(i + 1, cost))

            # Back prop
            self._back_prop(x, y, activation)

            # Gradient checking
            if grad_check is True and dropout_keep_prob == 1 and i < 10:
                grad_arrays = [self.parameters['dW' + str(i + 1)] for i in range(layers)]
                grad_arrays.extend([self.parameters['dB' + str(i + 1)] for i in range(layers)])
                grads = np.array([]).reshape(1, -1)
                for a in grad_arrays:
                    grads = np.concatenate((grads, a.reshape(1, -1)), axis=1)
                grads_num = self._num_grad(x, y, lambd=lambd, eps=1e-4)
                # print(np.concatenate((grads.T, grads_num.T), axis=1))
                numerator = np.linalg.norm(grads_num - grads)
                denominator = np.linalg.norm(grads_num) + np.linalg.norm(grads)
                print('Grad check result is {}'.format(numerator / denominator))

            if grad_check is True and dropout_keep_prob != 1:
                raise ValueError('When implenting gradient checking, dropout_keep_prob must be 1.')

            # Update weights
            for l in range(layers):
                for v in ['W', 'B']:
                    self.parameters[v + str(l + 1)] -= alpha * self.parameters['d' + v + str(l + 1)]

    def gd_mini_batch(self, x, y, alpha=0.1, activation='tanh', lambd=0, epochs=1000, mini_batch_size=64,
                      grad_check=False, dropout_keep_prob=1, optimizer=None, beta1=0.9, beta2=0.999):
        self.details['activation'] = activation
        self.details['epochs'] = epochs if 'epochs' not in self.details else self.details['epochs'] + epochs
        self.details['train_size'] = x.shape[1]
        self.hyper_parameters['alpha'] = alpha
        self.hyper_parameters['lambd'] = lambd
        self.hyper_parameters['dropout_keep_prob'] = dropout_keep_prob
        self.hyper_parameters['mini_batch_size'] = mini_batch_size
        self.hyper_parameters['optimizer'] = optimizer
        layers = self.details['layers']
        train_size = self.details['train_size']

        indices = np.random.permutation(train_size)
        x_shuffled = x[:, indices]
        y_shuffled = y[:, indices]
        batches_num = train_size // mini_batch_size
        x_batches = [x_shuffled[:, i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(batches_num + 1)]
        y_batches = [y_shuffled[:, i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(batches_num + 1)]
        op = {(x + str(l + 1)): 0 for l in range(layers) for x in ['MW', 'MB', 'RW', 'RB']}

        for i in range(epochs):
            costs = []
            for batch, xi, yi in zip(range(len(x_batches)), x_batches, y_batches):
                # Forward prop
                self._forward_prop(xi, activation, dropout_keep_prob=dropout_keep_prob)

                # Compute cost
                cost = self._compute_cost(yi, lambd=lambd)
                costs.append(cost)

                # Back prop
                self._back_prop(xi, yi, activation)

                # Gradient checking
                if grad_check is True and dropout_keep_prob == 1 and i == 0 and batch < 10:
                    grad_arrays = [self.parameters['dW' + str(i + 1)] for i in range(layers)]
                    grad_arrays.extend([self.parameters['dB' + str(i + 1)] for i in range(layers)])
                    grads = np.array([]).reshape(1, -1)
                    for a in grad_arrays:
                        grads = np.concatenate((grads, a.reshape(1, -1)), axis=1)
                    grads_num = self._num_grad(xi, yi, lambd=lambd, eps=1e-4)
                    # print(np.concatenate((grads.T, grads_num.T), axis=1))
                    numerator = np.linalg.norm(grads_num - grads)
                    denominator = np.linalg.norm(grads_num) + np.linalg.norm(grads)
                    print('Grad check result is {}'.format(numerator / denominator))

                if grad_check is True and dropout_keep_prob != 1:
                    raise ValueError('When implenting gradient checking, dropout_keep_prob must be 1.')

                # Update the weights
                if optimizer is None:
                    for l in range(layers):
                        for v in ['W', 'B']:
                            self.parameters[v + str(l + 1)] -= alpha * self.parameters['d' + v + str(l + 1)]

                elif optimizer == 'momentum':
                    for l in range(layers):
                        for v in ['W', 'B']:
                            op['M' + v + str(l + 1)] = beta1 * op['M' + v + str(l + 1)] + \
                                                        (1 - beta1) * self.parameters['d' + v + str(l + 1)]
                            #op['M' + v + str(l + 1)] /= (1 - beta1 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * op['M' + v + str(l + 1)]

                elif optimizer == 'rmsp':
                    for l in range(layers):
                        for v in ['W', 'B']:
                            op['R' + v + str(l + 1)] = beta2 * op['R' + v + str(l + 1)] + \
                                                        (1 - beta2) * self.parameters['d' + v + str(l + 1)] ** 2
                            #op['R' + v + str(l + 1)] /= (1 - beta2 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * self.parameters['d' + v + str(l + 1)] / \
                                                               (np.sqrt(op['R' + v + str(l + 1)]) + 1e-8)

                elif optimizer == 'adam':
                    for l in range(layers):
                        for v in ['W', 'B']:
                            op['M' + v + str(l + 1)] = beta1 * op['M' + v + str(l + 1)] + \
                                                        (1 - beta1) * self.parameters['d' + v + str(l + 1)]
                            #op['M' + v + str(l + 1)] /= (1 - beta1 ** (batch + 1))
                            op['R' + v + str(l + 1)] = beta2 * op['R' + v + str(l + 1)] + \
                                                        (1 - beta2) * self.parameters['d' + v + str(l + 1)] ** 2
                            #op['R' + v + str(l + 1)] /= (1 - beta2 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * op['M' + v + str(l + 1)] / \
                                                               (np.sqrt(op['R' + v + str(l + 1)]) + 1e-8)
            if (i + 1) % 20 == 0:
                print('The average cost for the {0}th epoch was {1}.'.format(i + 1, sum(costs)/len(costs)))

    def predict(self, x, y=None, cut_off=0.5):
        """
        Make predictions based on a test dataset.
        :param x: A numpy array. Data to run model on.
        :param y: A numpy array. Labels of the data if available.
        :param cut_off: A number between [0,1]. The number above which the label is 1.
        :return: A tuple, numpy array of predictions and error if y is not none.
        """
        m = x.shape[1]
        self._forward_prop(x, activation=self.details['activation'])
        yhat = self.parameters['A' + str(self.details['layers'])]
        if x.shape[0] == 1:
            predictions = np.where(yhat > cut_off, 1, 0)
            #error = np.sum(np.absolute(y - predictions)) / m if y is not None else None
        else:
            predictions = np.where(yhat == yhat.max(axis=0), 1, 0)
        error = 1 - (np.sum(np.all(y == predictions, axis=0)) / m) if y is not None else None
        return predictions, error



