# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *


from proj1_helpers import *

print("Begin loading dataset...")

DATA_TRAIN_PATH = "../data/train.csv"
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = "../data/test.csv"  # TODO: download test data and supply path here
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print("Done!")



# Standardize
tX, tX_test = standardize(tX, tX_test)

# Add intercept column
tX = np.insert(tX, 0, np.ones(tX.shape[0]), axis=1)
tX_test = np.insert(tX_test, 0, np.ones(tX_test.shape[0]), axis=1)

# Replace -999 with the median
tX, tX_test = replace_nans(tX, tX_test)

# Add new features
tX = expand_features(tX)
tX_test = expand_features(tX_test)


print(np.where(tX[0] == 0))

initial_w = np.zeros(tX.shape[1])

# Logistic regression using expanded features
w_logit, loss_logit = reg_logistic_regression(y, tX, lambda_= 0.1, initial_w=initial_w, max_iters=15000, gamma=0.07 /y.shape[0])
acc_logit = accuracy(w_logit, y, tX)
print(acc_logit)



OUT= 'sub.csv'
create_csv_submission(ids_test, predict_labels(w_logit, tX_test), OUT)

print("End")