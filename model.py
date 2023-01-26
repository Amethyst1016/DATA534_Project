
import math
import pylab
import random
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.svm import SVR
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#%%


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
cpi = pd.read_csv('data/cpi.csv') # per month from 1913-01-01 to 2022-12-01, total 1320 observations
unemployment = pd.read_csv('data/unemployment.csv') # per month from 1948-01-01 to 2022-12-01, total 900 observations
gdp = pd.read_csv('data/gdp.csv') # per quarter from 1947-01-01 to 2022-07-01, total 303 observations
fund_rate = pd.read_csv('data/fundrate.csv') # per month from 1954-07-01 to 2022-12-01, total 822 observations
retail = pd.read_csv('data/retail.csv') # per month from 1992-01-01 to 2022-12-01, total 372 observations
durables = pd.read_csv('data/durables.csv') # per month from 1992-02-01 to 2022-11-01, total 370 observations
SP500 = pd.read_csv('data/SP500.csv') # per day from 2016-01-04 to 2023-01-13, total 1771 observations

########## Data wrangling

# SP500
# Convert the 'Date' column to the 'Y-M' format
SP500['date'] = pd.to_datetime(SP500['Date']).dt.to_period('M')
# Calculate the average value for each month for column 'Close'
SP500 = SP500.groupby('date').mean()
# Subset the DataFrame with dates between 2016-01 and 2022-12
SP500 = SP500['2016-01':'2022-12']
# Only keep the column 'Close'
SP500 = SP500[['Close']]
# Rename the column 'Close' to 'SP500'
SP500.rename(columns={'Close': 'SP500'}, inplace=True)
# per month from 2016-01 to 2022-12, total 84 observations

# CPI
# Drop the first column 'Unnamed: 0'
cpi = cpi.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
cpi['date'] = pd.to_datetime(cpi['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-12
cpi.set_index('date', inplace=True)
cpi = cpi['2022-12':'2016-01']
cpi = cpi.sort_index(ascending=True)
# Rename the column 'value' to 'CPI'
cpi.rename(columns={'value': 'CPI'}, inplace=True)
# per month from 2016-01 to 2022-12, total 84 observations

# unemployment
# Drop the first column 'Unnamed: 0'
unemployment = unemployment.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
unemployment['date'] = pd.to_datetime(unemployment['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-12
unemployment.set_index('date', inplace=True)
unemployment = unemployment['2022-12':'2016-01']
unemployment = unemployment.sort_index(ascending=True)
# Rename the column 'value' to 'Unemployment'
unemployment.rename(columns={'value': 'Unemployment'}, inplace=True)
# per month from 2016-01 to 2022-12, total 84 observations

# gdp
# Drop the first column 'Unnamed: 0'
gdp = gdp.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
gdp['date'] = pd.to_datetime(gdp['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-07
gdp.set_index('date', inplace=True)
gdp = gdp['2022-07':'2016-01']
gdp = gdp.sort_index(ascending=True)
# Rename the column 'value' to 'GDP'
gdp.rename(columns={'value': 'GDP'}, inplace=True)
# per quarter from 2016-01 to 2022-07, total 27 observations

# fund_rate
# Drop the first column 'Unnamed: 0'
fund_rate = fund_rate.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
fund_rate['date'] = pd.to_datetime(fund_rate['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-12
fund_rate.set_index('date', inplace=True)
fund_rate = fund_rate['2022-12':'2016-01']
fund_rate = fund_rate.sort_index(ascending=True)
# Rename the column 'value' to 'Fund_rate'
fund_rate.rename(columns={'value': 'Fund_rate'}, inplace=True)
# per month from 2016-01 to 2022-12, total 84 observations

# retail
# Drop the first column 'Unnamed: 0'
retail = retail.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
retail['date'] = pd.to_datetime(retail['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-12
retail.set_index('date', inplace=True)
retail = retail['2022-12':'2016-01']
retail = retail.sort_index(ascending=True)
# Rename the column 'value' to 'Retail'
retail.rename(columns={'value': 'Retail'}, inplace=True)
# per month from 2016-01 to 2022-12, total 84 observations

# durables
# Drop the first column 'Unnamed: 0'
durables = durables.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
durables['date'] = pd.to_datetime(durables['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-07
durables.set_index('date', inplace=True)
durables = durables['2022-07':'2016-01']
durables = durables.sort_index(ascending=True)
# Rename the column 'value' to 'Durables'
durables.rename(columns={'value': 'Durables'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

# Merge all dataframes (exclude GDP for now)
df = pd.concat([SP500, cpi, unemployment, fund_rate, retail], axis=1)

#%%

########## Analysis of the data ##########
# Correlation matrix
df.corr()
#%%

# Scatter plot
for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        col1 = df.columns[i]
        col2 = df.columns[j]
        sns.scatterplot(x=col1, y=col2, data=df)
        plt.title(col1 + ' vs ' + col2)
        plt.show()
#%%

# Time series plot
for column in df.columns:
    df[column].plot()
    plt.title(column)
    plt.show()

#%%

########## Data preprocessing ##########

#### Fit time series cross-validation models
X = df.drop(columns='SP500')
Y = df['SP500']
tscv = TimeSeriesSplit(n_splits=10)
# This allows you to split the data into training and test sets by specifying the number of splits,
# and it ensures that the splits are done in a time-sensitive manner so that the training sets are
# always before the test sets in time.

# tscv.split(X) returns the indices of the training and test sets for each split
# X.iloc[train_index] returns the training set for each split
# X.iloc[test_index] returns the test set for each split
# Y.iloc[train_index] returns the training set for each split
# Y.iloc[test_index] returns the test set for each split


for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Linear regression
    lm = sm.OLS(Y_train, X_train).fit()
    lm_mse = mean_squared_error(Y_test, lm.predict(X_test))
    print('Linear regression MSE: ', lm_mse)

    # Decision tree
    Tree = DecisionTreeRegressor().fit(X_train, Y_train)
    Tree_mse = mean_squared_error(Y_test, Tree.predict(X_test))
    print('Decision tree MSE: ', Tree_mse)

    # Random forest
    RF = RandomForestRegressor().fit(X_train, Y_train)
    RF_mse = mean_squared_error(Y_test, RF.predict(X_test))
    print('Random forest MSE: ', RF_mse)

    # Gradient boosting
    GB = GradientBoostingRegressor().fit(X_train, Y_train)
    GB_mse = mean_squared_error(Y_test, GB.predict(X_test))
    print('Gradient boosting MSE: ', GB_mse)

    # Support vector machine
    SVM = SVR().fit(X_train, Y_train)
    SVM_mse = mean_squared_error(Y_test, SVM.predict(X_test))
    print('Support vector machine MSE: ', SVM_mse)

    # KNN
    KNN = KNeighborsRegressor().fit(X_train, Y_train)
    KNN_mse = mean_squared_error(Y_test, KNN.predict(X_test))
    print('KNN MSE: ', KNN_mse)

    print('--------------------------------------')

#### Make a table of MSE
MSE = pd.DataFrame({'MSE': [lm_mse, Tree_mse, RF_mse, GB_mse, SVM_mse, KNN_mse]},
                     index=['Linear Regression', 'Decision Tree', 'Random Forest',
                            'Gradient Boosting', 'Support Vector Machine', 'KNN'])
MSE
#%%

