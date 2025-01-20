import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array(pd.read_csv('train.csv'))
np.random.shuffle(data)

m, n = data.shape

test = data[0:1000].T
y_test = test[0]
x_test = (test[1:n])/255

train = data[1000:m].T
y_train = train[0]
x_train = (train[1:n])/255


def init_params():
    w1 = np.random.rand(10, 784)  *0.05
    b1 = np.random.rand(10,1) * 0.05
    w2 = np.random.rand(10,10) * 0.05
    b2 = np.random.rand(10,1) * 0.05

    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z,0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def for_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1,x) + b1
    a1 = ReLU(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = softmax(z2)

    return z1, a1, z2, a2


def one_hot(y):
    one_hot_y = np.zeros((np.size(y), np.max(y) + 1))
    one_hot_y[np.arange(np.size(y)), y] = 1

    return one_hot_y.T

def dReLU(z):
    return z > 0

def back_prop(z1, a1, z2, a2, w2, x, y):
    one_hot_y = one_hot(y)

    dz2 = 1/m * (a2 - one_hot_y)
    dw2 = np.dot(dz2 , a1.T)
    db2 = np.sum(dz2)

    dz1 = np.dot(w2.T, dz2) * dReLU(z1)
    dw1 =  np.dot(dz1, x.T)
    db1 =  np.sum(dz1)

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2

    return w1, b1, w2, b2


def get_pred(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    accuracy = np.sum(predictions == y)/np.size(y)
    print(accuracy)
    return accuracy


def grad_descent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()

    for i in range(iterations):
        z1, a1, z2, a2 = for_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 50 == 0:
            print('iteration: ', i)
            print('Accuracy: ', get_accuracy(get_pred(a2), y))

    return w1, b1, w2, b2


w1, b1, w2, b2 = grad_descent(x_train, y_train, 500, 0.1)

def make_pred(w1, b1, w2, b2, x):
    z1, a1, z2, a2 = for_prop(w1, b1, w2, b2, x)
    prediction = get_pred(a2)
    return prediction

def test_pred(index, w1, b1, w2, b2):
    current_image = x_train[:, index, None]
    prediction = make_pred(w1, b1, w2, b2, x_train[:, index, None])
    label = y_train[index]
    print('prediction: ', prediction)
    print('Label: ', label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_pred(0, w1, b1, w2, b2)
test_pred(1, w1, b1, w2, b2)
test_pred(2, w1, b1, w2, b2)
test_pred(3, w1, b1, w2, b2)


test_predictions = make_pred(w1, b1, w2, b2, x_test)
get_accuracy(test_predictions, y_test)

