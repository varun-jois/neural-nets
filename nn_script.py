from neural_net import NeuralNet
#from NeuralNet import NeuralNet
from helper_functions import *
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import copy
import _pickle

np.random.seed(0)
X, y = sklearn.datasets.make_moons(1000, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=10, c=y, cmap=plt.cm.Spectral)

X = X.T
Y = y.reshape(1, 1000)
mu, var = get_normalisation_constants(X)
X = (X - mu) / var
# X = X[:, :2]
# Y = Y[:, :2]

np.random.seed(0)
nn = NeuralNet()
nn.initialize_weights(2, [3,3,3,3,3,3,1], initializer='xavier')
nn.train_batch(X, Y, epochs=1000, alpha=1, activation='tanh', lambd=1, grad_check=True, dropout_keep_prob=1)
# nn.train_mini_batch(X, Y, epochs=2000, alpha=.001, activation='tanh', lambd=0, grad_check=True, mini_batch_size=16,
#                     optimizer='adam')
p1 = nn.parameters
with open('p1.p', 'wb') as f:
    _pickle.dump(p1, f)

p, err = nn.predict(X, Y)
print(err)
np.sum(p==Y)
nn.epochs

###############################################

from NeuralNet import NeuralNet, get_normalisation_constants
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import _pickle

np.random.seed(0)
X, y = sklearn.datasets.make_moons(1000, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=10, c=y, cmap=plt.cm.Spectral)
X = X.T
Y = y.reshape(1, 1000)
mu, var = get_normalisation_constants(X)
X = (X - mu) / var

np.random.seed(0)
nn = NeuralNet()
nn.initializer_xavier(2, [3,3,3,3,3,3,1])
nn.gd_batch(X, Y, epochs=1000, alpha=1, grad_check=True, lambd=1)
p2 = nn.parameters

p, err = nn.predict(X, Y)
print(err)

with open('p1.p', 'rb') as f:
    p1 = _pickle.load(f)

all([np.array_equal(p1[a], p2[b]) for a, b in zip(sorted(list(p1.keys())), sorted(list(p2.keys())))])



