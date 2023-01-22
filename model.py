
from sklearn.metrics import mean_squared_error
import math
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

def get_folds(n, K):
    ### Get the appropriate number of fold labels
    n_fold = math.ceil(n / K) # Number of observations per fold (rounded up)
    fold_ids_raw = list(range(1, K+1)) * n_fold
    fold_ids = fold_ids_raw[:n]
    random.shuffle(fold_ids)
    return fold_ids

def rescale(x1, x2):
    for col in range(x1.shape[1]):
        a = np.min(x2[:,col])
        b = np.max(x2[:,col])
        x1[:,col] = (x1[:,col]-a)/(b-a)
    return x1

