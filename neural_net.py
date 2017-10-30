from helper_functions import *
import copy


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
                self.parameters['W' + str(l + 2)] = np.zeros((layer_sizes[l + 1], layer_sizes[l]))

        elif algorithm == 'random':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * 0.01
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l + 2)] = np.random.randn(layer_sizes[l + 1], layer_sizes[l]) * 0.01

        elif algorithm == 'relu_specific':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * np.sqrt(2 / input_layer_size)
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l + 2)] = np.random.randn(layer_sizes[l + 1], layer_sizes[l]) * \
                                                    np.sqrt(2 / layer_sizes[l])

        elif algorithm == 'xavier':
            self.parameters['W1'] = np.random.randn(layer_sizes[0], input_layer_size) * np.sqrt(1 / input_layer_size)
            for l in range(len(layer_sizes) - 1):
                self.parameters['W' + str(l + 2)] = np.random.randn(layer_sizes[l + 1], layer_sizes[l]) * \
                                                    np.sqrt(1 / layer_sizes[l])

        self.parameters['B1'] = np.zeros((layer_sizes[0], 1))
        for l in range(len(layer_sizes) - 1):
            self.parameters['B' + str(l + 2)] = np.zeros((layer_sizes[l + 1], 1))

    def _forward_prop(self, x, activation='relu'):
        self.parameters['Z1'] = self.parameters['W1'].dot(x) + self.parameters['B1']
        self.parameters['A1'] = function(activation)(self.parameters['Z1'])
        for l in range(self.layers - 2):
            pl = str(l + 1)
            cl = str(l + 2)
            self.parameters['Z' + cl] = self.parameters['W' + cl].dot(self.parameters['A' + pl]) + \
                                        self.parameters['B' + cl]
            self.parameters['A' + cl] = function(activation)(self.parameters['Z' + cl])

        self.parameters['Z' + str(self.layers)] = \
            self.parameters['W' + str(self.layers)].dot(self.parameters['A' + str(self.layers - 1)]) + \
            self.parameters['B' + str(self.layers)]
        if self.layer_sizes[-1] == 1:
            self.parameters['A' + str(self.layers)] = function('sigmoid')(self.parameters['Z' + str(self.layers)])
        else:
            self.parameters['A' + str(self.layers)] = function('softmax')(self.parameters['Z' + str(self.layers)])

    def _compute_cost(self, y, lambd=0):
        m = y.shape[1]
        yhat = self.parameters['A' + str(self.layers)]
        cost = -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / m + \
                lambd * sum(np.sum(self.parameters['W' + str(l + 1)] ** 2) for l in range(self.layers)) / (2 * m)
        return cost

    def _back_prop(self, x, y, activation):
        m = x.shape[1]
        ls = len(self.layer_sizes)
        if self.layer_sizes[-1] == 1:
            self.parameters['dZ' + str(ls)] = self.parameters['A' + str(ls)] - y

        for l in reversed(range(ls - 1)):
            self.parameters['dW' + str(l + 2)] = \
                self.parameters['dZ' + str(l + 2)].dot(self.parameters['A' + str(l + 1)].T) / m
            self.parameters['dB' + str(l + 2)] = \
                np.sum(self.parameters['dZ' + str(l + 2)], axis=1, keepdims=True) / m
            self.parameters['dZ' + str(l + 1)] = \
                self.parameters['W' + str(l + 2)].T.dot(self.parameters['dZ' + str(l + 2)]) * \
                function(activation, prime=True)(self.parameters['Z' + str(l + 1)])
        self.parameters['dW1'] = self.parameters['dZ1'].dot(x.T) / m
        self.parameters['dB1'] = np.sum(self.parameters['dZ1'], axis=1, keepdims=True) / m

    def _update_weights(self, alpha=0.01):
        for l in range(self.layers):
            self.parameters['W' + str(l + 1)] = self.parameters['W' + str(l + 1)] - \
                                                alpha * self.parameters['dW' + str(l + 1)]
            self.parameters['B' + str(l + 1)] = self.parameters['B' + str(l + 1)] - \
                                                alpha * self.parameters['dB' + str(l + 1)]

    def _unroll_arrays(self, array_list):
        params_unrolled = np.array([]).reshape(1, -1)
        for a in array_list:
            params_unrolled = np.concatenate((params_unrolled, a.reshape(1, -1)), axis=1)
        return params_unrolled

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

    def train(self, x, y, alpha=0.01, activation='relu', lambd=0, epochs=1000):
        self.activation = activation
        self.hyper_parameters['alpha'] = alpha
        self.hyper_parameters['lambd'] = lambd
        self.epochs += epochs

        #parameters_copy = copy.deepcopy(self.parameters)
        for i in range(epochs):
            # Forward prop
            self._forward_prop(x, activation)

            # Compute cost
            cost = self._compute_cost(y, lambd=lambd)
            if (i + 1) % 100 == 0:
                print('The cost after {0} rounds is {1}.'.format(i + 1, cost))

            # Back prop
            self._back_prop(x, y, activation)
            if i < 10:
                grad_arrays = [self.parameters['dW' + str(i + 1)] for i in range(self.layers)]
                grad_arrays.extend([self.parameters['dB' + str(i + 1)] for i in range(self.layers)])
                grads = self._unroll_arrays(grad_arrays)
                grads_num = self._num_grad(x, y, lambd=lambd, eps=1e-4)
                #print(np.concatenate((grads.T, grads_num.T), axis=1))
                nuerator = np.linalg.norm(grads_num - grads)
                denominator = np.linalg.norm(grads_num) + np.linalg.norm(grads)
                print('Grad check result is {}'.format(nuerator / denominator))

            # Update the weights
            self._update_weights(alpha=alpha)
            
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
