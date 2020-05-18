import numpy as np
import matplotlib.pyplot as plt
import os
from helper import *

"""
Homework2: perceptron classifier
"""


def sign(x):
    return 1 if x > 0 else -1


# -------------- Implement your code Below -------------#

def show_images(data):
    """
    This function is used for plot image and save it.

    Args:
    data: Two images from train data with shape (2, 16, 16). The shape represents total 2
          images and each image has size 16 by 16.

    Returns:
        Do not return any arguments, just save the images you plot for your report.
    """
    i = 1
    for x in data:
        plt.imshow(x, cmap="Greys")
        plt.title(f'Image {i}')
        plt.imsave(f'../output/figure_a{i}.png', x, cmap="Greys")
        plt.close()
        i += 1


def show_features(data, label):
    """
    This function is used for plot a 2-D scatter plot of the features and save it.

    Args:
    data: train features with shape (1561, 2). The shape represents total 1561 samples and
          each sample has 2 features.
    label: train data's label with shape (1561,1).
           1 for digit number 1 and -1 for digit number 5.

    Returns:
    Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
    """
    # using symmetry as x-axis, average intensity as y-axis
    plt.scatter(data[:, 0][label == 1], data[:, 1][label == 1], c='r', marker='*')
    plt.scatter(data[:, 0][label == -1], data[:, 1][label == -1], c='b', marker='+')
    plt.legend([1, 5], title='Digit Number')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.title('Train Set - Perceptron')
    #plt.show()
    plt.savefig('../output/figure_b.png')
    plt.close()


def perceptron(data, label, max_iter, learning_rate):
    """
    The perceptron classifier function.

    Args:
    data: train data with shape (1561, 3), which means 1561 samples and
            each sample has 3 features.(1, symmetry, average intensity)
    label: train data's label with shape (1561,1).
            1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update

    Returns:
        w: the seperater with shape (1, 3). You must initialize it with w = np.zeros((1,d))
    """
    w = np.zeros((1, 3))
    [n, _] = data.shape
    for _ in range(max_iter):
        for i in range(n):
            x = data[i, :]
            h = sign(np.dot(w, x))
            if h != label[i]:
                w += learning_rate * label[i] * x
    return w

def show_result(data, label, w):
    """
    This function is used for plot the test data with the separators and save it.

    Args:
    data: test features with shape (424, 2). The shape represents total 424 samples and
           each sample has 2 features.
    label: test data's label with shape (424,1).
           1 for digit number 1 and -1 for digit number 5.

    Returns:
    Do not return any arguments, just save the image you plot for your report.
    """
    # plotting data
    plt.scatter(data[:, 0][label == 1], data[:, 1][label == 1], c='r', marker='*')
    plt.scatter(data[:, 0][label == -1], data[:, 1][label == -1], c='b', marker='+')
    plt.legend([1, 5], title='Digit Number')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.title('Test Set - Perceptron')
    # plotting separator
    # using symmetry as x-axis, average intensity as y-axis
    # replace x1 with x, and x2 with y to find the equation of the separator
    w0, w1, w2 = w[0, 0], w[0, 1], w[0, 2]
    # y = -w1/w2*x - w0/w2
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    xs = np.linspace(x_min, x_max)
    plt.plot(xs, -w1 / w2 * xs - w0 / w2, c='g')
    #plt.show()
    plt.savefig('../output/figure_d.png')
    plt.close()

# -------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
    n, _ = data.shape
    mistakes = 0
    for i in range(n):
        if sign(np.dot(data[i, :], np.transpose(w))) != label[i]:
            mistakes += 1
    return (n - mistakes) / n


def test_perceptron(max_iter, learning_rate):
    # get data
    traindataloc, testdataloc = "../data/train.txt", "../data/test.txt"
    train_data, train_label = load_features(traindataloc)
    test_data, test_label = load_features(testdataloc)
    # train perceptron
    w = perceptron(train_data, train_label, max_iter, learning_rate)
    train_acc = accuracy_perceptron(train_data, train_label, w)
    # test perceptron model
    test_acc = accuracy_perceptron(test_data, test_label, w)
    return w, train_acc, test_acc
