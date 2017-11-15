from NeuralNet import NeuralNet, get_normalisation_constants
import sklearn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import _pickle

np.random.seed(0)
X, y = sklearn.datasets.make_moons(1000, noise=0.2)
plt.scatter(X[:,0], X[:,1], s=10, c=y)
# X, y = sklearn.datasets.make_blobs(1000, n_features=2, centers=3, random_state=2)
# plt.scatter(X[:,0], X[:,1], s=10, c=y)
lb = sklearn.preprocessing.LabelBinarizer()

X = X.T
Y = lb.fit_transform(y).T
mu, var = get_normalisation_constants(X)
X = (X - mu) / var

np.random.seed(0)
nn = NeuralNet()
nn.initializer_xavier(2, [5, 5, 1])
nn.gd_batch(X, Y, epochs=3000, alpha=1, grad_check=True, lambd=0, activation='tanh')
p2 = nn.parameters

p, err = nn.predict(X, Y)
print(err)

with open('p1.p', 'rb') as f:
    p1 = _pickle.load(f)

all([np.array_equal(p1[a], p2[b]) for a, b in zip(sorted(list(p1.keys())), sorted(list(p2.keys())))])



