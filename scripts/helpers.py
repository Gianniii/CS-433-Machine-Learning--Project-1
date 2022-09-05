# Helper functions for implementations.py and run.py

import numpy as np
from proj1_helpers import predict_labels

# Given function helper for SGD
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# Functions done in labs 5
# **********************************************************************************************************
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def max_likelihood_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    l1 = np.log(1 + np.exp(tx @ w))
    l2 = y * (tx @ w)
    loss = np.sum(l1 - l2)
    return loss


def mle_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx @ w)
    grad = tx.T @ (pred - y)
    return grad


# **********************************************************************************************************
def compute_mse(y, tx, w):
    """Calculate the MSE loss."""
    N = y.shape[0]
    cost = np.square((y - tx @ w)).sum() / (2 * N)
    return cost


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - tx @ w
    return (-1 / N) * tx.T @ e


# **********************************************************************************************************
# Helper function for run.py

def standardize(x, x_test):
    """Standardize a data set, ignoring nans"""

    # Compute relevant statistics
    x[x == -999] = np.nan
    x_test[x_test == -999] = np.nan
    mean_x = np.nanmean(x, axis=0)
    std_x = np.nanstd(x, axis=0)

    # Normalize
    x_n = (x - mean_x) / std_x
    x_test_n = (x_test - mean_x) / std_x

    # Put back the -999
    x[np.isnan(x)] = -999

    return x_n, x_test_n


def replace_nans(x, x_test):
    """Replaces the -999 values by the median of the relevant feature"""
    x[x == -999] = np.nan
    x_test[x_test == -999] = np.nan

    # Compute relevant statistic
    medians = np.nanmedian(x, axis=0)

    # Replace
    for j in np.arange(x.shape[1]):
        x[np.isnan(x[:, j]), j] = medians[j]
        x_test[np.isnan(x_test[:, j]), j] = medians[j]

    return x, x_test


def expand_features(x):
    """Adds some new features"""
    expanded = np.zeros([x.shape[0], 600]) # 600 is larger than what we need but
    # it doesn't matter to have a large number of zero columns
    print(expanded.shape)
    n_col = x.shape[1]

    # Add pairwise multiplications of original columns
    p = 0
    for i in np.arange(n_col):
        for j in np.arange(i, n_col):
            expanded[:, p] = x[:, i] * x[:, j]
            p += 1

    # Add logs of original columns
    for i in np.arange(n_col):
        expanded[:, p + i] = np.log(1 + np.abs(x[:, i]))

    return expanded


def split_data(y, tX):
    # split ratio 20% / 80%
    split = int(y.shape[0]/5)
    return y[0:split], tX[0:split, :], y[split:], tX[split:, :]


def accuracy(w, y_test, tX_test):
    """Percentage of correct predictions"""
    y_pred = predict_labels(w, tX_test)
    return 100.0 * np.sum(np.abs(y_test + y_pred)) / (2 * y_test.shape[0])
