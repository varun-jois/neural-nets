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

#np.random.seed(3)
nn = NeuralNet()
nn.initialize_weights(2, [5, 1], algorithm='xavier')
nn.train(X, Y, epochs=500, alpha=1, activation='tanh', lambd=0, grad_check=True)

p, err = nn.predict(X, Y)
print(err)
np.sum(p==Y)
nn.epochs

W1 = params['W1']
W2 = params['W2']
B1 = params['B1']
B2 = params['B2']
Z1 = W1.dot(X) + B1
A1 = function('relu')(Z1)
Z2 = W2.dot(A1) + B2
A2 = function('sigmoid')(Z2)

dZ2 = A2 - Y
dW2 = dZ2.dot(A1.T) / 200
dB2 = np.sum(dZ2, axis=1, keepdims=True) / 200
dZ1 = W2.T.dot(dZ2) * function('relu', prime=True)(Z1)
dW1 = dZ1.dot(X.T) / 200
dB1 = np.sum(dZ1, axis=1, keepdims=True) / 200

dZ2_2 = nn.parameters['dZ2']
dW2_2 = nn.parameters['dW2']
dB2_2 = nn.parameters['dB2']
dZ1_2 = nn.parameters['dZ1']
dW1_2 = nn.parameters['dW1']
dB2_2 = nn.parameters['dB2']



dW2 == dW2_2
dB2 == dB2_2
dW1 == dW1_2
dB2 == dB2_2






