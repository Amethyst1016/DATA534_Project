## code for generating testing model
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import TimeSeriesSplit

def drop_unnamed(df):
    # Drop the column 'Unnamed: 0'
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

def format_date(df):
    # Convert the 'date' column to the 'Y-M' format and set as index
    df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
    df.set_index('date', inplace=True)

def df_subset(df, name):
    # Subset the DataFrame based on dates
    if name == 'GDP':
        idx = df['2015-10':].index
    else:
        idx = df['2015-12':].index
    df.drop(idx, inplace=True)
    df.sort_index(ascending=True, inplace=True)

def df_rename(df, name):
    # Rename column
    df.rename(columns={'value': name}, inplace=True)

### Read data
cpi = pd.read_csv('data/cpi.csv') # per month from 1913-01-01 to 2022-12-01, total 1320 observations
unemployment = pd.read_csv('data/unemployment.csv') # per month from 1948-01-01 to 2022-12-01, total 900 observations
gdp = pd.read_csv('data/gdp.csv') # per quarter from 1947-01-01 to 2022-07-01, total 303 observations
fund_rate = pd.read_csv('data/fundrate.csv') # per month from 1954-07-01 to 2022-12-01, total 822 observations
retail = pd.read_csv('data/retail.csv') # per month from 1992-01-01 to 2022-12-01, total 372 observations
durables = pd.read_csv('data/durables.csv') # per month from 1992-02-01 to 2022-11-01, total 370 observations
SP500 = pd.read_csv('data/SP500.csv') # per day from 2016-01-04 to 2023-01-13, total 1771 observations
SP500_whole = pd.read_csv('data/SP500_whole.csv') # per day from 1927-12-30 to 2023-01-27, total 23883 observations
### Data Wrangling
for df, name in zip([cpi, unemployment, gdp, fund_rate, durables, retail],
                    ['CPI', 'Unemployment', 'GDP', 'Fund_rate', 'Durables', 'Retail']):
    drop_unnamed(df)
    format_date(df)
    df_subset(df, name)
    df_rename(df, name)

# SP500 is a special one
SP500 = SP500.rename({'Date':'date'}, axis=1)
format_date(SP500)
# Calculate the average value for each month for column 'Close'
SP500 = SP500.groupby('date').mean()
# Subset the DataFrame with dates between 2016-01 and 2022-12
SP500 = SP500['2016-01':'2022-12']
# Only keep the column 'Close'
SP500 = SP500[['Close']]
# Rename the column 'Close' to 'SP500'
SP500.rename(columns={'Close': 'SP500'}, inplace=True)

df = pd.concat([SP500, cpi, unemployment, fund_rate, retail], axis=1)

X = df.drop(columns='SP500')
Y = df['SP500']
tscv = TimeSeriesSplit(n_splits=10)

import random
random.seed(777)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Linear regression
    LM = LinearRegression().fit(X_train, Y_train)
    LM_mse = mean_squared_error(Y_test, LM.predict(X_test))
    #print('Linear regression MSE: ', lm_mse)

    # Random forest
    RF = RandomForestRegressor().fit(X_train, Y_train)
    RF_mse = mean_squared_error(Y_test, RF.predict(X_test))
    #print('Random forest MSE: ', RF_mse)


#%%

LM_mse = mean_squared_error(Y_test, LM.predict(X_test))
# 1446860.3721969668

LM_r_square = LM.score(X_train, Y_train) # train data
r2_score(Y_train, LM.predict(X_train))
# 0.9208607427107347

LM_intercept = LM.intercept_
# -9895.264936868789

LM_coef= LM.coef_
# 47.907904
# -6.010215
# -96.414447
# 0.001737

LM.feature_names_in_
# CPI
# Unemployment
# Fund_rate
# Retail

residuals = Y_test - LM.predict(X_test)
residuals
# 2022-06,-1322.328242
# 2022-07,-1247.200902
# 2022-08,-953.468052
# 2022-09,-1208.457830
# 2022-10,-1373.966119
# 2022-11,-1124.550100
# 2022-12,-1141.092924

#%%


#%%
RF_mse = mean_squared_error(Y_test, RF.predict(X_test))
# 109488.0671571658

Rf_r_square = RF.score(X_train, Y_train) # train data
r2_score(Y_train, RF.predict(X_train))
# 0.9968843113436046

RF.feature_names_in_
# 0,CPI
# 1,Unemployment
# 2,Fund_rate
# 3,Retail


RF.feature_importances_
# 0,0.953768
# 1,0.027691
# 2,0.011759
# 3,0.006782

RF_residuals = Y_test - RF.predict(X_test)
RF_residuals
# 2022-06,-307.211151
# 2022-07,-304.882254
# 2022-08,-48.488441
# 2022-09,-361.454430
# 2022-10,-491.454480
# 2022-11,-288.669240
# 2022-12,-348.176278
#%%


######## create a class for the 2 models
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LmModel:
    """
    A class to perform linear regression.

    Attributes
    ----------
    X_train : DataFrame
        The training data for the independent variables.
    Y_train : DataFrame
        The training data for the dependent variable.
    X_test : DataFrame
        The test data for the independent variables.
    Y_test : DataFrame
        The test data for the dependent variable.
    model : LinearRegression
        The linear regression model.

    Methods
    -------
    get_mse():
        Obtain the mean squared error.
    get_r_square():
        Obtain the coefficient of determination.
    get_coef():
        Obtain the coefficients of the linear regression model.
    get_intercept():
        Obtain the intercept of the linear regression model.
    get_residuals():
        Obtain the residuals of the linear regression model.
    get_equation():
        Obtain the equation of the linear regression model.
    """

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = LinearRegression().fit(self.X_train, self.Y_train)

    def get_mse(self):
        # Obtain the mean squared error.
        self.mse = mean_squared_error(self.Y_test, self.model.predict(self.X_test))
        return self.mse

    def get_r_square(self):
        # Obtain the coefficient of determination.
        self.r_square = r2_score(self.Y_train, self.model.predict(self.X_train))
        return self.r_square

    def get_coef(self):
        # Obtain the coefficients from the linear regression model.
        self.coef = self.model.coef_
        return self.coef

    def get_intercept(self):
        # Obtain the intercept from the linear regression model.
        self.intercept = self.model.intercept_
        return self.intercept

    def get_residuals(self):
        # Obtain the residuals.
        self.residuals = self.Y_test - self.model.predict(self.X_test)
        return self.residuals

    def get_equation(self):
        # Obtain the equation of the linear regression model.
        self.equation = 'y = ' + str(self.intercept)
        for i in range(len(self.coef)):
            self.equation += ' + ' + str(self.coef[i]) + ' * ' + self.X_train.columns[i]
        return self.equation
#%% test the class
lm_try = LmModel(X_train, Y_train, X_test, Y_test)
print(lm_try.get_mse())
print(lm_try.get_r_square())
print(lm_try.get_coef())
print(lm_try.get_intercept())
print(lm_try.get_residuals())
print(lm_try.get_equation())



#%%
class RfModel:
    """
    A class to perform random forest regression.

    Attributes
    ----------
    X_train : DataFrame
        The training data for the independent variables.
    Y_train : DataFrame
        The training data for the dependent variable.
    X_test : DataFrame
        The test data for the independent variables.
    Y_test : DataFrame
        The test data for the dependent variable.

    Methods
    -------
    get_mse():
        Obtain the mean squared error.
    get_r_square():
        Obtain the coefficient of determination.
    get_feature_importances():
        Obtain the feature importances.
    get_residuals():
        Obtain the residuals.
    """
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = RandomForestRegressor().fit(self.X_train, self.Y_train)

    def get_mse(self):
        # Obtain the mean squared error.
        self.mse = mean_squared_error(self.Y_test, self.model.predict(self.X_test))
        return self.mse

    def get_r_square(self):
        # Obtain the coefficient of determination.
        self.r_square = r2_score(self.Y_train, self.model.predict(self.X_train))
        return self.r_square

    def get_feature_importances(self):
        # Obtain the feature importances.
        self.feature_importances = self.model.feature_importances_
        return self.feature_importances

    def get_residuals(self):
        # Obtain the residuals.
        self.residuals = self.Y_test - self.model.predict(self.X_test)
        return self.residuals


#%% test the class
random.seed(777)
rf_try = RfModel(X_train, Y_train, X_test, Y_test)
print(rf_try.get_mse())
print(rf_try.get_r_square())
print(rf_try.get_feature_importances())
print(rf_try.get_residuals())

#%%
