
import math
import random
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor



def get_folds(n, K):
    ### Get the appropriate number of fold labels
    n_fold = math.ceil(n / K) # Number of observations per fold (rounded up)
    fold_ids_raw = list(range(1, K+1)) * n_fold
    fold_ids = fold_ids_raw[:n]
    random.shuffle(fold_ids)
    return fold_ids



def rescale(x1, x2):
    for col in range(x1.shape[1]):
        a = np.min(x2[:, col])
        b = np.max(x2[:, col])
        x1[:, col] = (x1[:, col] - a) / (b - a)
    return x1


# Get data from csv file
cpi = pd.read_csv('cpi.csv') # per month from 1913-01-01 to 2022-12-01, total 1320 observations
unemployment = pd.read_csv('unemployment.csv') # per month from 1948-01-01 to 2022-12-01, total 900 observations
gdp = pd.read_csv('gdp.csv') # per quarter from 1947-01-01 to 2021-07-01, total 303 observations
fundrate = pd.read_csv('fundrate.csv') # per month from 1954-07-01 to 2022-12-01, total 822 observations
retail = pd.read_csv('retail.csv') # per month from 1992-01-01 to 2022-12-01, total 372 observations
durables = pd.read_csv('durables.csv') # per month from 1992-02-01 to 2022-11-01, total 370 observations
SP500 = pd.read_csv('SP500.csv') #  per day from 2016-01-04 to 2023-01-13, total 1771 observations




#%%
