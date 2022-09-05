import numpy as np
from helpers import batch_iter, compute_mse, max_likelihood_loss
from helpers import compute_gradient, mle_gradient, sigmoid


# Linear regression using gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        w = w - gamma * grad
        # avoid printing at each iteration
        if n_iter % 20 == 0:
            print(f"Loss at %d iteration: %f" % (n_iter, loss))

    return w, loss


# Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):

            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_mse(minibatch_y, minibatch_tx, w)
            if n_iter % 20 == 0:
                print(f"Loss at %d iteration: %f" % (n_iter, loss))

    return w, loss


# Least squares regression using normal equations
def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T @ tx
    b = tx.T @ y
    w_opt = np.linalg.solve(a, b)
    mse = compute_mse(y, tx, w_opt)
    return w_opt, mse


# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx + aI
    b = tx.T @ y
    w_ridge = np.linalg.solve(a, b)
    mse_ridge = compute_mse(y, tx, w_ridge) + lambda_ * w_ridge.T @ w_ridge
    return w_ridge, mse_ridge


# Logistic regression using gradient descent or SGD
def logistic_regression(y, tx, initial_w, max_iters, gamma, mode="GD"):
    w = initial_w
    # we want y in {0, 1} instead of {-1, 1}
    y = (y + 1.0) / 2.0
    if mode == "GD":
        for n_iter in range(max_iters):
            grad = mle_gradient(y, tx, w)

            w = w - gamma * grad
            loss = max_likelihood_loss(y, tx, w)
            if n_iter % 20 == 0:
                print(f"Loss at %d iteration: %f" % (n_iter, loss))

    elif mode == "SGD":
        batch_size = 1
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1
            ):
                grad = mle_gradient(minibatch_y, minibatch_tx, w)
                w = w - gamma * grad
                loss = max_likelihood_loss(minibatch_y, minibatch_tx, w)
                if n_iter % 20 == 0:
                    print(f"Loss at %d iteration: %f" % (n_iter, loss))

    return w, loss


# Regularized logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, mode="GD"):
    w = initial_w
    # we want y in {0, 1} instead of {-1, 1}
    y = (y + 1.0) / 2.0
    if mode == "GD":
        for n_iter in range(max_iters):
            grad = mle_gradient(y, tx, w)

            w = w - gamma * grad
            loss = max_likelihood_loss(y, tx, w) + lambda_ * w.T @ w
            if n_iter % 20 == 0:
                print(f"Loss at %d iteration: %f" % (n_iter, loss))
            # avoid going in negative loss
            if loss < 0:
                return w, loss

    elif mode == "SGD":
        batch_size = 1
        for n_iter in range(max_iters):
            for minibatch_y, minibatch_tx in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1
            ):
                grad = mle_gradient(minibatch_y, minibatch_tx, w)
                w = w - gamma * grad
                loss = (
                    max_likelihood_loss(minibatch_y, minibatch_tx, w)
                    + lambda_ * w.T @ w
                )
                if n_iter % 20 == 0:
                    print(f"Loss at %d iteration: %f" % (n_iter, loss))

    return w, loss
