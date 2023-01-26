
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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
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
cpi = pd.read_csv('cpi.csv') # per month from 1913-01-01 to 2022-12-01, total 1320 observations
unemployment = pd.read_csv('unemployment.csv') # per month from 1948-01-01 to 2022-12-01, total 900 observations
gdp = pd.read_csv('gdp.csv') # per quarter from 1947-01-01 to 2022-07-01, total 303 observations
fund_rate = pd.read_csv('fundrate.csv') # per month from 1954-07-01 to 2022-12-01, total 822 observations
retail = pd.read_csv('retail.csv') # per month from 1992-01-01 to 2022-12-01, total 372 observations
durables = pd.read_csv('durables.csv') # per month from 1992-02-01 to 2022-11-01, total 370 observations
SP500 = pd.read_csv('SP500.csv') # per day from 2016-01-04 to 2023-01-13, total 1771 observations

########## Data wrangling

# SP500
# Convert the 'Date' column to the 'Y-M' format
SP500['date'] = pd.to_datetime(SP500['Date']).dt.to_period('M')
# Calculate the average value for each month for column 'Close'
SP500 = SP500.groupby('date').mean()
# Subset the DataFrame with dates between 2016-01 and 2022-07
SP500 = SP500['2016-01':'2022-07']
# Only keep the column 'Close'
SP500 = SP500[['Close']]
# Rename the column 'Close' to 'SP500'
SP500.rename(columns={'Close': 'SP500'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

# CPI
# Drop the first column 'Unnamed: 0'
cpi = cpi.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
cpi['date'] = pd.to_datetime(cpi['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-07
cpi.set_index('date', inplace=True)
cpi = cpi['2022-07':'2016-01']
cpi = cpi.sort_index(ascending=True)
# Rename the column 'value' to 'CPI'
cpi.rename(columns={'value': 'CPI'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

# unemployment
# Drop the first column 'Unnamed: 0'
unemployment = unemployment.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
unemployment['date'] = pd.to_datetime(unemployment['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-07
unemployment.set_index('date', inplace=True)
unemployment = unemployment['2022-07':'2016-01']
unemployment = unemployment.sort_index(ascending=True)
# Rename the column 'value' to 'Unemployment'
unemployment.rename(columns={'value': 'Unemployment'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

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
# Subset the DataFrame with dates between 2016-01 and 2022-07
fund_rate.set_index('date', inplace=True)
fund_rate = fund_rate['2022-07':'2016-01']
fund_rate = fund_rate.sort_index(ascending=True)
# Rename the column 'value' to 'Fund_rate'
fund_rate.rename(columns={'value': 'Fund_rate'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

# retail
# Drop the first column 'Unnamed: 0'
retail = retail.drop(columns=['Unnamed: 0'])
# Convert the 'date' column to the 'Y-M' format
retail['date'] = pd.to_datetime(retail['date']).dt.to_period('M')
# Subset the DataFrame with dates between 2016-01 and 2022-07
retail.set_index('date', inplace=True)
retail = retail['2022-07':'2016-01']
retail = retail.sort_index(ascending=True)
# Rename the column 'value' to 'Retail'
retail.rename(columns={'value': 'Retail'}, inplace=True)
# per month from 2016-01 to 2022-07, total 79 observations

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
df = pd.concat([SP500, cpi, unemployment, fund_rate, retail, durables], axis=1)

#%%

########## Analysis

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
#### Fit a linear regression model
X = df.drop(columns=['SP500'])
Y = df['SP500']
# create a LinearRegression object and fit the model to the data
# Get the model summary
lm = sm.OLS(Y, X).fit()
# Get the model summary
lm.summary()
# Plot the residuals
sns.residplot(lm.predict(X), Y)
#%%
# Visualize the residuals and check the normality assumptions.
stats.probplot(lm.resid, dist="norm", plot=pylab)
pylab.show()
# Check MSE
lm_mse = mean_squared_error(Y, lm.predict(X))
lm_mse

#%%
#### Fit a decision tree model
Tree = DecisionTreeRegressor().fit(X, Y)
# Plot the residuals
sns.residplot(Tree.predict(X), Y)
# Check MSE
Tree_mse = mean_squared_error(Y, Tree.predict(X))
Tree_mse

#%%
#### Fit a random forest model
RF = RandomForestRegressor().fit(X, Y)
# Plot the residuals
sns.residplot(RF.predict(X), Y)
# Check MSE
RF_mse = mean_squared_error(Y, RF.predict(X))
RF_mse

#%%
#### Fit a gradient boosting model
GB = GradientBoostingRegressor().fit(X, Y)
# Plot the residuals
sns.residplot(GB.predict(X), Y)
# Check MSE
GB_mse = mean_squared_error(Y, GB.predict(X))
GB_mse


#%%
#### Fit a neural network model
NN = MLPRegressor().fit(X, Y)
# Plot the residuals
sns.residplot(NN.predict(X), Y)
# Check MSE
NN_mse = mean_squared_error(Y, NN.predict(X))
NN_mse

#%%
#### Fit a support vector machine model
SVM = SVR().fit(X, Y)
# Plot the residuals
sns.residplot(SVM.predict(X), Y)
# Check MSE
SVM_mse = mean_squared_error(Y, SVM.predict(X))
SVM_mse

#%%


