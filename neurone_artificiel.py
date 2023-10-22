import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimension de X:', X.shape)
print('dimension de y:', y.shape)

#plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
#plt.show

def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return(W, b)

W, b = init(X)
print(W.shape)
print(b)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

A = model(X, W, b)
print(A.shape)

def log_loss(A, y):
    return (1/len(y)) * np.sum(-y * np.log(A) - (1-y) * np.log(1 - A))

print(log_loss(A, y))

def gradients(A, X, y):
    dw = 1/len(y) * np.dot(X.T, A - y)
    db = 1/len(y) * np.sum(A - y)
    return(dw, db)

dw, db = gradients(A, X, y)
print(dw.shape)
print(db.shape)

def update(dw, db, W, b, learning_rate):
    W = W - learning_rate * dw
    b = b - learning_rate * db
    return (W, b)

def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
    # initialize parameters
    W, b = init(X)

    loss = []

    for i in range (n_iter):
        A = model(X, W, b)
        loss.append(log_loss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    plt.plot(loss)
    plt.show()

artificial_neuron(X, y)

