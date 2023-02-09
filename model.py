import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LinearModel:
    
    """
    A class to perform linear regression.

    Attributes
    ----------
    df: Dataframe
        Input dataframe containing date, independent variables and dependent variable.
    X_col: List
        The name of independent variables.
    y_col: str
        The name of dependent variable.
    test_size: float
        The proportion of data assigned to test set.

    Methods
    -------
    get_summary():
        Obtain MSE, coefficients, intercept, residuals, equation, and R-Squared.
        Return a dictionary containing all coefficients.
    get_parameters():
        Return a dataframe of variables and coefficients ordered by absolute value.
    """

    def __init__(self, df, X_col, y_col, test_size):
        self.df = df
        self.n = round(len(self.df)*test_size)
        self.X_col = X_col
        self.X = self.df[self.X_col]
        self.y_col = y_col
        self.y = self.df[[self.y_col]]
        self.X_train = self.X.iloc[self.n:,:]
        self.y_train = self.y.iloc[self.n:,:]
        self.X_test = self.X.iloc[:self.n,:]
        self.y_test = self.y.iloc[:self.n,:]
        self.model = LinearRegression().fit(self.X_train, self.y_train)
        self.simple = False
        if len(X_col) == 1:
            self.simple = True

        if not type(self.df) == pd.core.frame.DataFrame:
            raise TypeError('df must be a dataframe.')
        if not type(self.X_col) == list:
            raise TypeError('X_col must be a list.')
        if not type(self.y_col) == str:
            raise TypeError('y_col must be a string.')
        if not type(test_size) == float:
            raise TypeError('test_size must be a float.')
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1.')
        if not self.X_col:
            raise ValueError('X_col must not be empty.')
        if not self.y_col:
            raise ValueError('y_col must not be empty.')
        if not self.X_col[0] in self.df.columns:
            raise ValueError('X_col must be in df.')
        if not self.y_col in self.df.columns:
            raise ValueError('y_col must be in df.')



    def get_summary(self):
        try:
            self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
            self.coef = self.model.coef_.tolist()
            self.intercept = self.model.intercept_.tolist()
            self.residuals = self.y_test - self.model.predict(self.X_test)
            self.equation = self.y_col + ' = ' + str(self.intercept[0])
            for i in range(len(self.coef)):
                self.equation += ' + ' + str(self.coef[i][0]) + ' * ' + self.X_train.columns[i]
            self.R_Squared = self.model.score(self.X_train, self.y_train)
            return dict({'MSE': self.mse,
                         'Coefficients': self.coef,
                         'Intercept': self.intercept,
                         'Residuals': self.residuals,
                         'Equation': self.equation,
                         'R-Squared': self.R_Squared})
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None


    
    def get_combined_df(self, model='LinearModel'):
        try:
            if self.simple:
                df_prediction = self.df[['date',self.X_col[0]]]
                df_true = self.df[['date',self.X_col[0], self.y_col]]
            else:
                df_prediction = self.df[['date']]
                df_true = self.df[['date',self.y_col]]
            prediction = []
            for i in self.model.predict(self.X_test):
                prediction.append(i[0])
            for i in self.model.predict(self.X_train):
                prediction.append(i[0])
            df_prediction[self.y_col] = prediction
            df_prediction['type'] = f'{model}_prediction'
            df_true['type'] = 'true value'
            return pd.concat([df_true, df_prediction], ignore_index=True)
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None

    
    def get_parameters(self):
        try:
            variables = pd.DataFrame(self.X_col, columns=['variable'])
            coefs = pd.DataFrame(self.coef[0], columns=['coefficient'])
            params = pd.concat([variables, coefs], axis=1)
            order = np.argsort(abs(coefs)['coefficient'])
            return params.reindex(order).reset_index(drop=True)
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None
        
class RfModel:
    """
    A class to perform random forest regression.

    Attributes
    ----------
    df: Dataframe
        Input dataframe containing date, independent variables and dependent variable.
    X_col: List
        The name of independent variables.
    y_col: str
        The name of dependent variable.
    test_size: float
        The proportion of data assigned to test set.

    Methods
    -------
    get_summary():
        Obtain MSE, feature importance, residuals, and R-Squared.
        Return a dictionary containing all coefficients.
    get_combined_df():
        Return a dataframe with date, value of true and prediction, type of value.
   get_parameters():
        Return a dataframe of feature importance ordered by absolute value.
    """
    def __init__(self, df, X_col, y_col, test_size):
        self.df = df
        self.n = round(len(self.df)*test_size)
        self.X_col = X_col
        self.X = self.df[self.X_col]
        self.y_col = y_col
        self.y = self.df[[self.y_col]]
        self.X_train = self.X.iloc[self.n:,:]
        self.y_train = self.y.iloc[self.n:,:]
        self.X_test = self.X.iloc[:self.n,:]
        self.y_test = self.y.iloc[:self.n,:]
        self.model = RandomForestRegressor().fit(self.X_train, self.y_train)

        if not type(self.df) == pd.core.frame.DataFrame:
            raise TypeError('df must be a dataframe.')
        if not type(self.X_col) == list:
            raise TypeError('X_col must be a list.')
        if not type(self.y_col) == str:
            raise TypeError('y_col must be a string.')
        if not type(test_size) == float:
            raise TypeError('test_size must be a float.')
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1.')
        if not self.X_col:
            raise ValueError('X_col must not be empty.')
        if not self.y_col:
            raise ValueError('y_col must not be empty.')
        if not self.X_col[0] in self.df.columns:
            raise ValueError('X_col must be in df.')
        if not self.y_col in self.df.columns:
            raise ValueError('y_col must be in df.')

    def get_summary(self):
        try:
            self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
            self.feature_importances = self.model.feature_importances_
            self.residuals = self.y_test - self.model.predict(self.X_test).reshape(self.n,1)
            self.R_Squared = self.model.score(self.X_train, self.y_train)
            return dict({'MSE': self.mse, 'Feature importances': self.feature_importances, 'Residuals': self.residuals,
                         'R-Squared': self.R_Squared})
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None
    
    def get_combined_df(self, model='RandomForest'):
        try:
            df_prediction = self.df[['date']]
            prediction = []
            for i in self.model.predict(self.X_test).reshape(self.n,1):
                prediction.append(i[0])
            for i in self.model.predict(self.X_train).reshape(len(self.df) - self.n,1):
                prediction.append(i[0])
            df_prediction[self.y_col] = prediction
            df_prediction['type'] = f'{model}_prediction'
            df_true = self.df[['date',self.y_col]]
            df_true['type'] = 'true value'
            return pd.concat([df_true, df_prediction], ignore_index=True)
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None

    
    def get_parameters(self):
        try:
            variables = pd.DataFrame(self.X_col, columns=['variable'])
            importances = pd.DataFrame(self.feature_importances.tolist(), columns=['importance'])
            params = pd.concat([variables, importances], axis=1)
            order = np.argsort(abs(importances)['importance'])
            return params.reindex(order).reset_index(drop=True)
        except ValueError as e:
            print("An error occurred:", e)
            return None
        except TypeError as e:
            print("An error occurred:", e)
            return None
        except AttributeError as e:
            print("An error occurred:", e)
            return None
        except RuntimeError as e:
            print("An error occurred:", e)
            return None
#%%
