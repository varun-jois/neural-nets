from helper_functions import *
import copy
from scipy.stats import truncnorm


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
        self.train_size = None
        self.layer_sizes = None
        self.initializer = None
        self.activation = None
        self.epochs = 0

    def initialize_weights(self, input_layer_size, layer_sizes, initializer='xavier'):
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
        self.initializer = initializer
        self.layers = len(layer_sizes)

        all_layers = [input_layer_size] + layer_sizes
        if initializer == 'zeros':
            for l in range(len(layer_sizes)):
                self.parameters['W' + str(l + 1)] = np.zeros((all_layers[l + 1], all_layers[l]))

        elif initializer == 'random':
            for l in range(len(layer_sizes)):
                self.parameters['W' + str(l + 1)] = np.random.randn(all_layers[l + 1], all_layers[l]) * 0.01

        elif initializer == 'he':
            for l in range(len(layer_sizes)):
                self.parameters['W' + str(l + 1)] = truncnorm(a=-2, b=2, loc=0,
                    scale=np.sqrt(2 / all_layers[l])).rvs(size=(all_layers[l + 1], all_layers[l]))

        elif initializer == 'xavier':
            for l in range(len(layer_sizes)):
                self.parameters['W' + str(l + 1)] = truncnorm(a=-2, b=2, loc=0,
                    scale=np.sqrt(2 / (all_layers[l] + all_layers[l + 1]))).rvs(size=(all_layers[l + 1], all_layers[l]))

        for l in range(len(layer_sizes)):
            self.parameters['B' + str(l + 1)] = np.zeros((all_layers[l + 1], 1))

    def _forward_prop(self, x, activation='tanh', dropout_keep_prob=1):
        self.parameters['Z1'] = self.parameters['W1'].dot(x) + self.parameters['B1']
        if self.layers > 1:
            self.parameters['A1'] = function(activation)(self.parameters['Z1'])
            self.parameters['M1'] = (np.random.rand(*self.parameters['A1'].shape) < dropout_keep_prob) / dropout_keep_prob
            self.parameters['A1'] *= self.parameters['M1']
            for l in range(self.layers - 2):
                pl = str(l + 1)
                cl = str(l + 2)
                self.parameters['Z' + cl] = self.parameters['W' + cl].dot(self.parameters['A' + pl]) + \
                    self.parameters['B' + cl]
                self.parameters['A' + cl] = function(activation)(self.parameters['Z' + cl])

                self.parameters['M' + cl] = (np.random.rand(
                    *self.parameters['A' + cl].shape) < dropout_keep_prob) / dropout_keep_prob
                self.parameters['A' + cl] *= self.parameters['M' + cl]

            self.parameters['Z' + str(self.layers)] = \
                self.parameters['W' + str(self.layers)].dot(self.parameters['A' + str(self.layers - 1)]) + \
                self.parameters['B' + str(self.layers)]
        if self.layer_sizes[-1] == 1:
            self.parameters['A' + str(self.layers)] = function('sigmoid')(self.parameters['Z' + str(self.layers)])
        else:
            self.parameters['A' + str(self.layers)] = function('softmax')(self.parameters['Z' + str(self.layers)])

    def _compute_cost(self, y, lambd=0, e=1e-10):
        m = y.shape[1]
        yhat = self.parameters['A' + str(self.layers)]
        cost = -np.sum(y * np.log(yhat + e) + (1 - y) * np.log(1 - yhat + e)) / m + \
                lambd * sum(np.sum(self.parameters['W' + str(l + 1)] ** 2) for l in range(self.layers)) / (2 * m)
        return cost

    def _back_prop(self, x, y, activation):
        m = y.shape[1]
        ls = self.layers
        if self.layer_sizes[-1] == 1:
            self.parameters['dZ' + str(ls)] = self.parameters['A' + str(ls)] - y

        for l in reversed(range(ls - 1)):
            self.parameters['dW' + str(l + 2)] = \
                self.parameters['dZ' + str(l + 2)].dot(self.parameters['A' + str(l + 1)].T) / m + \
                self.hyper_parameters['lambd'] / m * self.parameters['W' + str(l + 2)]
            self.parameters['dB' + str(l + 2)] = \
                np.sum(self.parameters['dZ' + str(l + 2)], axis=1, keepdims=True) / m
            self.parameters['dZ' + str(l + 1)] = \
                self.parameters['W' + str(l + 2)].T.dot(self.parameters['dZ' + str(l + 2)]) * \
                self.parameters['M' + str(l + 1)] * function(activation, prime=True)(self.parameters['Z' + str(l + 1)])
        self.parameters['dW1'] = self.parameters['dZ1'].dot(x.T) / m + \
            self.hyper_parameters['lambd'] / m * self.parameters['W1']
        self.parameters['dB1'] = np.sum(self.parameters['dZ1'], axis=1, keepdims=True) / m

    def _update_weights(self, alpha=0.01):
        for l in range(self.layers):
            self.parameters['W' + str(l + 1)] -= alpha * self.parameters['dW' + str(l + 1)]
            self.parameters['B' + str(l + 1)] -= alpha * self.parameters['dB' + str(l + 1)]

    def _num_grad(self, x, y, lambd, eps=1e-7):
        params = copy.deepcopy(self.parameters)
        grads_num = []
        for l in range(self.layers):
            for r in range(params['W' + str(l + 1)].shape[0]):
                for c in range(params['W' + str(l + 1)].shape[1]):
                    self.parameters = copy.deepcopy(params)
                    self.parameters['W' + str(l + 1)][r, c] = self.parameters['W' + str(l + 1)][r, c] + eps
                    self._forward_prop(x, activation=self.activation)
                    costp = self._compute_cost(y, lambd=lambd)
                    self.parameters = copy.deepcopy(params)
                    self.parameters['W' + str(l + 1)][r, c] = self.parameters['W' + str(l + 1)][r, c] - eps
                    self._forward_prop(x, activation=self.activation)
                    costn = self._compute_cost(y, lambd=lambd)
                    grads_num.append((costp - costn) / (2 * eps))
        for l in range(self.layers):
            for r in range(params['B' + str(l + 1)].shape[0]):
                for c in range(params['B' + str(l + 1)].shape[1]):
                    self.parameters = copy.deepcopy(params)
                    self.parameters['B' + str(l + 1)][r, c] = self.parameters['B' + str(l + 1)][r, c] + eps
                    self._forward_prop(x, activation=self.activation)
                    costp = self._compute_cost(y, lambd=lambd)
                    self.parameters = copy.deepcopy(params)
                    self.parameters['B' + str(l + 1)][r, c] = self.parameters['B' + str(l + 1)][r, c] - eps
                    self._forward_prop(x, activation=self.activation)
                    costn = self._compute_cost(y, lambd=lambd)
                    grads_num.append((costp - costn) / (2 * eps))
        self.parameters = copy.deepcopy(params)
        return np.array(grads_num).reshape(1, -1)

    def train(self, x, y, alpha=0.1, activation='tanh', lambd=0, epochs=1000, grad_check=False,
              dropout_keep_prob=1, mini_batch_options=None):
        self.train_size = x.shape[1]
        self.activation = activation
        self.epochs += epochs
        beta1 = mini_batch_options['beta1'] if mini_batch_options is not None else None
        beta2 = mini_batch_options['beta2'] if mini_batch_options is not None else None
        self.hyper_parameters['beta1'] = beta1
        self.hyper_parameters['beta2'] = beta2
        self.hyper_parameters['alpha'] = alpha
        self.hyper_parameters['lambd'] = lambd
        self.hyper_parameters['optimizer'] = mini_batch_options['optimizer'] \
            if mini_batch_options is not None else None
        self.hyper_parameters['dropout_keep_prob'] = dropout_keep_prob


        indices = np.random.permutation(self.train_size)
        x_shuffled = x[:, indices]
        y_shuffled = y[:, indices]
        if mini_batch_options is not None:
            batches_num = self.train_size // 64
            x_batches = [x_shuffled[:, i * 64:(i + 1) * 64] for i in range(batches_num + 1)]
            y_batches = [y_shuffled[:, i * 64:(i + 1) * 64] for i in range(batches_num + 1)]
            mbp = {(x + str(l + 1)): 0 for l in range(self.layers) for x in ['MW', 'MB', 'RW', 'RB']}
        else:
            x_batches = [x_shuffled]
            y_batches = [y_shuffled]

        for i in range(epochs):
            if mini_batch_options is not None:
                mbp = {(x + str(l + 1)): 0 for l in range(self.layers) for x in ['MW', 'MB', 'RW', 'RB']}
            for batch, xi, yi in zip(range(len(x_batches)), x_batches, y_batches):
                # Forward prop
                self._forward_prop(xi, activation, dropout_keep_prob=dropout_keep_prob)

                # Compute cost
                cost = self._compute_cost(yi, lambd=lambd)
                if mini_batch_options is not None:
                    if (i + 1) % 50 == 0 and batch == 0:
                        print('The cost after epoch {0} is {1}.'.format(i + 1, cost))
                else:
                    if (i + 1) % 100 == 0:
                        print('The cost after epoch {0} is {1}.'.format(i + 1, cost))

                # Back prop
                self._back_prop(xi, yi, activation)

                # Gradient checking
                if grad_check is True and dropout_keep_prob == 1 and i == 0 and batch < 10:
                    grad_arrays = [self.parameters['dW' + str(i + 1)] for i in range(self.layers)]
                    grad_arrays.extend([self.parameters['dB' + str(i + 1)] for i in range(self.layers)])
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
                if mini_batch_options is not None and mini_batch_options['optimizer'] == 'momentum':
                    if beta1 is None:
                        raise ValueError('beta1 must be present in mini_batch_options if momentum is used. You can ' +
                                         'use the value 0.9.')
                    for l in range(self.layers):
                        for v in ['W', 'B']:
                            mbp['M' + v + str(l + 1)] = beta1 * mbp['M' + v + str(l + 1)] + \
                                                        (1 - beta1) * self.parameters['d' + v + str(l + 1)]
                            mbp['M' + v + str(l + 1)] /= (1 - beta1 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * mbp['M' + v + str(l + 1)]

                elif mini_batch_options is not None and mini_batch_options['optimizer'] == 'rmsp':
                    if beta1 is None:
                        raise ValueError('beta2 must be present in mini_batch_options if rmsp is used. You can ' +
                                         'use the value 0.999.')
                    for l in range(self.layers):
                        for v in ['W', 'B']:
                            mbp['R' + v + str(l + 1)] = beta2 * mbp['R' + v + str(l + 1)] + \
                                                        (1 - beta2) * self.parameters['d' + v + str(l + 1)] ** 2
                            mbp['R' + v + str(l + 1)] /= (1 - beta2 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * self.parameters['d' + v + str(l + 1)] / \
                                                               (np.sqrt(mbp['R' + v + str(l + 1)]) + 1e-8)

                elif mini_batch_options is not None and mini_batch_options['optimizer'] == 'adam':
                    if beta1 is None:
                        raise ValueError('beta1 and beta2 must be present in mini_batch_options if adam is used. ' +
                                         'You can use the values 0.9 and 0.999 respectively.')
                    for l in range(self.layers):
                        for v in ['W', 'B']:
                            mbp['M' + v + str(l + 1)] = beta1 * mbp['M' + v + str(l + 1)] + \
                                                        (1 - beta1) * self.parameters['d' + v + str(l + 1)]
                            mbp['M' + v + str(l + 1)] /= (1 - beta1 ** (batch + 1))
                            mbp['R' + v + str(l + 1)] = beta2 * mbp['R' + v + str(l + 1)] + \
                                                        (1 - beta2) * self.parameters['d' + v + str(l + 1)] ** 2
                            mbp['R' + v + str(l + 1)] /= (1 - beta2 ** (batch + 1))
                            self.parameters[v + str(l + 1)] -= alpha * mbp['M' + v + str(l + 1)] / \
                                                               (np.sqrt(mbp['R' + v + str(l + 1)]) + 1e-8)
                else:
                    for l in range(self.layers):
                        for v in ['W', 'B']:
                            self.parameters[v + str(l + 1)] -= alpha * self.parameters['d' + v + str(l + 1)]

    def predict(self, x, y=None, cut_off=0.5):
        """
        Make predictions based on a test dataset.
        :param x: A numpy array. Data to run model on.
        :param y: A numpy array. Labels of the data if available.
        :param cut_off: A number between [0,1]. The number above which the label is 1.
        :return: A tuple, numpy array of predictions and error if y is not none.
        """
        m = x.shape[1]
        self._forward_prop(x, activation=self.activation)
        yhat = self.parameters['A' + str(self.layers)]
        predictions = np.where(yhat > cut_off, 1, 0)

        # Calculating the error
        if y is not None:
            error = np.sum(np.absolute(y - predictions)) / m

        return predictions, error
