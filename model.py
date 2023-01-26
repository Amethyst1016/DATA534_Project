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
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import altair as alt

def get_folds(n, K):
    ### Get the appropriate number of fold labels
    n_fold = math.ceil(n / K) # Number of observations per fold (rounded up)
    fold_ids_raw = list(range(1, K+1)) * n_fold
    fold_ids = fold_ids_raw[:n]
    random.shuffle(fold_ids)
    return fold_ids

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
df

### Analysis
# Correlation matrix
df.corr()

# Scatter plots
columns = np.asarray(df.columns)
scatter = alt.Chart(df, height=150, width=150).mark_point(
).encode(
    alt.X(alt.repeat('column'), type = 'quantitative', scale=alt.Scale(zero=False), title=''),
    alt.Y(alt.repeat('row'), type = 'quantitative', scale=alt.Scale(zero=False), title='')
).repeat(
    row = columns, column = columns
)

# Trend
df_no_index = df.reset_index()
df_no_index['date'] = df_no_index['date'].dt.to_timestamp().apply(lambda x: x.strftime('%Y-%m'))
trend = alt.Chart(df_no_index, height=100, width=150).mark_line().encode(
    alt.X('date:T'),
    alt.Y(alt.repeat('repeat'), type = 'quantitative', scale=alt.Scale(zero=False))
).repeat(
    repeat = columns, columns = 3
)

### Data Preprocessing
# Fit time series cross-validation models
X = df.drop(columns='SP500')
Y = df['SP500']
tscv = TimeSeriesSplit(n_splits=10)
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

    # KNN
    KNN = KNeighborsRegressor().fit(X_train, Y_train)
    KNN_mse = mean_squared_error(Y_test, KNN.predict(X_test))
    print('KNN MSE: ', KNN_mse)
    
    print('--------------------------------------')
    
# Make a scatter plot of the actual vs predicted values
plt.scatter(Y_test, lm.predict(X_test))
plt.title('Linear regression')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

plt.scatter(Y_test, Tree.predict(X_test))
plt.title('Decision tree')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

plt.scatter(Y_test, RF.predict(X_test))
plt.title('Random forest')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

plt.scatter(Y_test, GB.predict(X_test))
plt.title('Gradient boosting')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

plt.scatter(Y_test, KNN.predict(X_test))
plt.title('KNN')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

#### Fit an ARIMA (AutoRegressive Integrated Moving Average) model
# Split the data into training and testing sets
SP500_train = SP500[:'2021-12']
SP500_test = SP500['2022-01':]
# fit ARIMA model, specifying the order of the model
ARIMA_model = ARIMA(SP500_train, order=(1, 1, 1))
ARIMA_model_fit = ARIMA_model.fit()
print(ARIMA_model_fit.summary())

# make predictions
arima_predict = ARIMA_model_fit.predict(start=len(SP500_train), end=len(SP500_train)+11, typ='levels')
arima_mse = mean_squared_error(SP500_test, arima_predict)

# Make a scatter plot of the actual vs predicted values
plt.scatter(SP500_test, arima_predict)
plt.title('ARIMA')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# fit Auto ARIMA model, order is automatically selected
auto_arima_model = auto_arima(SP500_train, start_p=1, start_q=1,
                            max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
# Print model summary
print(auto_arima_model.summary())

# Make predictions on the test set
auto_arima_predict = auto_arima_model.predict(n_periods=len(SP500_test))
auto_arima_mse = mean_squared_error(SP500_test, auto_arima_predict)

# Make a scatter plot of the actual vs predicted values
plt.scatter(SP500_test, auto_arima_predict)
plt.title('Auto ARIMA')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Make a table of MSE adding ARIMA and Auto ARIMA models
MSE = pd.DataFrame({'MSE': [lm_mse, Tree_mse, RF_mse, GB_mse, KNN_mse, arima_mse, auto_arima_mse]},
                        index=['Linear Regression', 'Decision Tree', 'Random Forest',
                                 'Gradient Boosting', 'KNN', 'ARIMA', 'Auto ARIMA'])
# Make a MSE plot
MSE.plot(kind='bar')
plt.title('MSE of different models')
plt.show()

# Make a relative MSE plot
MSE['Relative MSE'] = MSE['MSE'] / MSE['MSE'].min()
MSE.plot(kind='bar', y='Relative MSE')
plt.title('Relative MSE of different models')
plt.show()