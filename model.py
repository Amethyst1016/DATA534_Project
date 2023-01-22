
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

# Get data
cpi_url = 'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=O6LFU5LE4ZVYXL1H'
inflation_url = 'https://www.alphavantage.co/query?function=INFLATION&apikey=O6LFU5LE4ZVYXL1H'
unemployment_url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=O6LFU5LE4ZVYXL1H'

cpi_df = pd.read_json(cpi_url)
inflation_df = pd.read_json(inflation_url)
unemployment_df = pd.read_json(unemployment_url)

# Extract date and value from data
cpi_df['date'] = cpi_df['data'].apply(lambda x: x['date'])
cpi_df['value'] = cpi_df['data'].apply(lambda x: x['value'])
cpi_df = cpi_df[['date', 'value']]

inflation_df['date'] = inflation_df['data'].apply(lambda x: x['date'])
inflation_df['value'] = inflation_df['data'].apply(lambda x: x['value'])
inflation_df = inflation_df[['date', 'value']]

unemployment_df['date'] = unemployment_df['data'].apply(lambda x: x['date'])
unemployment_df['value'] = unemployment_df['data'].apply(lambda x: x['value'])
unemployment_df = unemployment_df[['date', 'value']]


#%%
