from neural_net import NeuralNet
from helper_functions import *
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import copy

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
#plt.scatter(X[:,0], X[:,1], s=10, c=y, cmap=plt.cm.Spectral)

X = X.T
Y = y.reshape(1, 200)
mu, var = get_normalisation_constants(X)
X = (X - mu) / var
# X = X[:, :2]
# Y = Y[:, :2]

nn = NeuralNet()
nn.initialize_weights(2, [5, 5, 5, 5, 5, 5, 1], initializer='xavier')
mo = {'beta1': 0.9, 'beta2': 0.999, 'optimizer': 'adam'}
nn.train(X, Y, epochs=1000, alpha=.09, activation='tanh', lambd=0, grad_check=True)

p, err = nn.predict(X, Y)
print(err)
np.sum(p==Y)
nn.epochs

a = np.array([1, 2]).reshape(1,-1)
b = a / 1e-100
c = 1 / (1 + np.exp(-1000000000))
0 * np.inf
np.array([np.nan,np.nan]) + np.array([np.nan,np.nan])
np.array([np.nan,np.nan]) + np.array([np.inf,np.nan])

np.sum(np.array([1e10, 9e1000]) ** 2)

