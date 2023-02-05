
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
