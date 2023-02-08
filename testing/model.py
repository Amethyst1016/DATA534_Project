import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class LinearModel:


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

    def get_summary(self):
        self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        self.coef = self.model.coef_.tolist()
        self.intercept = self.model.intercept_.tolist()
        self.residuals = self.y_test - self.model.predict(self.X_test)
        self.equation = self.y_col + ' = ' + str(self.intercept[0])
        for i in range(len(self.coef)):
            self.equation += ' + ' + str(self.coef[i][0]) + ' * ' + self.X_train.columns[i]
        self.R_Squared = self.model.score(self.X_train, self.y_train)
        return dict({'MSE': self.mse, 'Coefficients': self.coef, 'Intercept': self.intercept, 'Residuals': self.residuals, 
                     'Equation': self.equation, 'R-Squared': self.R_Squared})
    
    def get_combined_df(self, model='LinearModel'):
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
    
    def get_parameters(self):
        variables = pd.DataFrame(self.X_col, columns=['variable'])
        coefs = pd.DataFrame(self.coef[0], columns=['coefficient'])
        params = pd.concat([variables, coefs], axis=1)
        order = np.argsort(abs(coefs)['coefficient'])
        return params.reindex(order).reset_index(drop=True)
        
class RfModel:

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

    def get_summary(self):
        self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        self.feature_importances = self.model.feature_importances_
        self.residuals = self.y_test - self.model.predict(self.X_test).reshape(self.n,1)
        self.R_Squared = self.model.score(self.X_train, self.y_train)
        return dict({'MSE': self.mse, 'Feature importances': self.feature_importances, 'Residuals': self.residuals, 
                     'R-Squared': self.R_Squared})
    
    def get_combined_df(self, model='RandomForest'):
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
    
    def get_parameters(self):
        variables = pd.DataFrame(self.X_col, columns=['variable'])
        importances = pd.DataFrame(self.feature_importances.tolist(), columns=['importance'])
        params = pd.concat([variables, importances], axis=1)
        order = np.argsort(abs(importances)['importance'])
        return params.reindex(order).reset_index(drop=True)