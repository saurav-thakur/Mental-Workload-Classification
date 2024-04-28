import numpy as np
import math
import scipy.io
from sklearn.metrics import accuracy_score

np.random.seed(40)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_true, y_pred):
    
    m = y_true.shape[0]  # total no. of examples
    
    # clipping y_pred to avoid log(0) or log(1)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return loss


def forward_propagation(X,weight,bias):
    z = np.dot(weight.T , X ) + bias
    sigmoid_output = sigmoid(z)
    return sigmoid_output


def backprop(X, y_true, y_pred):
    m = X.shape[1]  # Number of training examples

    # Computing the derivatives
    dl_w = 1/m * np.dot(X, (y_pred - y_true).T)
    dl_b = 1/m * np.sum(y_pred - y_true)

    return dl_w, dl_b

def load_data(data_path):
    mat_data = scipy.io.loadmat(data_path)
    data = mat_data["data"]
    label = mat_data["label"]

    # data prep
    data = data.reshape(62*512,360)
    labels = label.reshape(360,1)
    labels = np.squeeze(labels)

    return data,labels
