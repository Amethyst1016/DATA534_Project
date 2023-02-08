### model.py
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
############################################################################################################

#%%

from model import *
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class TestLinearModel(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'date': ['2022-01-06', '2022-01-05', '2022-01-04',
                                         '2022-01-04', '2022-01-02', '2022-01-01'],
                                'x1': [1, 2, 3,
                                       1.5, 2.5, 3.5],
                                'x2': [4, 5, 6,
                                       4.5, 5.5, 6.5],
                                'y': [7, 8, 9,
                                      7.5, 8.5, 9.5]})
        self.X_col = ['x1', 'x2']
        self.y_col = 'y'
        self.test_size = 0.5
        self.n = round(len(self.df)*self.test_size)

        self.X = self.df[self.X_col]
        self.y = self.df[self.y_col]
        self.X_train = self.X.iloc[self.n:, :]
        self.y_train = self.y.iloc[self.n:, :]
        self.X_test = self.X.iloc[:self.n, :]
        self.y_test = self.y.iloc[:self.n, :]
        self.model = LinearRegression().fit(self.X_train, self.y_train)
        self.simple = False
        if len(self.X_col) == 1:
            self.simple = True


    def test_get_summary(self):

        self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        self.coef = self.model.coef_tolist()
        self.intercept = self.model.intercept_tolist()
        self.residuals = self.y_test - self.model.predict(self.X_test)
        self.equation = self.y_col + ' = ' + str(self.intercept[0])
        for i in range(len(self.coef)):
            self.equation += ' + ' + str(self.coef[i][0]) + ' * ' + self.X_train.columns[i]
        self.R_Squared = self.model.score(self.X_train, self.y_train)


        self.assertAlmostEqual(self.mse, 0.0)
        self.assertListEqual(self.coef, [[1.0], [1.0]])
        self.assertAlmostEqual(self.intercept, [0.0])
        self.assertListEqual(self.residuals, [0.0, 0.0, 0.0])
        self.assertEqual(self.equation, 'y = 0.0 + 1.0 * x1 + 1.0 * x2')
        self.assertAlmostEqual(self.R_Squared, 1.0)


    def test_get_combined_df(self, model='LinearModel'):
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

        combined_df = pd.concat([df_true, df_prediction], ignore_index=True)

        self.assertAlmostEqual(combined_df, (12, 4))
        self.assertListEqual(combined_df['date'].tolist(),
                             ['2022-01-06', '2022-01-05', '2022-01-04',
                              '2022-01-03', '2022-01-02', '2022-01-01'])

        self.assertListEqual(combined_df[self.y_col].tolist(), [7.0, 8.0, 9.0, 7.0, 8.0, 9.0])
        self.assertListEqual(combined_df['type'].tolist(), ['true value', 'true value', 'true value',
                                                            'LinearModel_prediction', 'LinearModel_prediction',
                                                            'LinearModel_prediction'])

    def test_get_parameters(self):
        self.variables = pd.DataFrame(self.X_col, columns=['variable'])
        self.coefs = pd.DataFrame(self.coef[0], columns=['coefficient'])
        self.params = pd.concat([self.variables, self.coefs], axis=1)
        self.order = np.argsort(abs(self.coefs)['coefficient'])

        self.assertEqual(self.params, (2, 2))
        self.assertListEqual(self.params['variable'].tolist(), ['x1', 'x2'])
        self.assertListEqual(self.params['coefficient'].tolist(), [1.0, 1.0])




if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)

#%%
