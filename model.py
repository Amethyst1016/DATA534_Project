
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
    get_summary():
        Obtain MSE, coefficients, intercept, residuals, equation, train score, and test score.
        Return a dictionary containing all coefficents.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = LinearRegression().fit(self.X_train, self.y_train)

    def get_summary(self):
        self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_
        self.residuals = self.y_test - self.model.predict(self.X_test)
        self.equation = 'y = ' + str(self.intercept)
        for i in range(len(self.coef)):
            self.equation += ' + ' + str(self.coef[i]) + ' * ' + self.X_train.columns[i]
        self.train_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)
        return dict({'MSE': self.mse, 'Coefficients': self.coef, 'Intercept': self.intercept, 'Residuals': self.residuals, 
                     'Equation': self.equation, 'Train score': self.train_score, 'Test score': self.test_score})
    
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
    get_summary():
        Obtain MSE, feature importances, residuals, train score, and test score.
        Return a dictionary containing all coefficents.
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = RandomForestRegressor().fit(self.X_train, self.y_train)

    def get_summary(self):
        self.mse = mean_squared_error(self.y_test, self.model.predict(self.X_test))
        self.feature_importances = self.model.feature_importances_
        self.residuals = self.y_test - self.model.predict(self.X_test)
        self.train_score = self.model.score(self.X_train, self.y_train)
        self.test_score = self.model.score(self.X_test, self.y_test)
        return dict({'MSE': self.mse, 'Feature importances': self.feature_importances, 'Residuals': self.residuals, 
                     'Train score': self.train_score, 'Test score': self.test_score})