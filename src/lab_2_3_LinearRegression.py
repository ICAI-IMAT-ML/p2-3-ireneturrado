# Import here whatever you may need
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class LinearRegressor:
    """
    Linear Regression model that can perform both simple and multiple linear regression.

    Attributes:
        coefficients (np.ndarray): Coefficients of the independent variables in the regression model.
        intercept (float): Intercept of the regression model.
    """

    def __init__(self):
        """Initializes the LinearRegressor model with default coefficient and intercept values."""
        self.coefficients = None
        self.intercept = None

    def fit_simple(self, X, y):
        """
        Fit the model using simple linear regression (one independent variable).

        This method calculates the coefficients for a linear relationship between
        a single predictor variable X and a response variable y.

        Args:
            X (np.ndarray): Independent variable data (1D array).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if np.ndim(X) > 1:
            X = X.reshape(1, -1)

        # TODO: Train linear regression model with only one coefficient
        self.coefficients = None
        self.intercept = None

        # y = b + w*x, donde b es el intercepto y w el coeficiente
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        #calculo w como se ha visto en clase, w = Sxy / Sx^2
        Sxy = np.sum((X - x_mean) * (y - y_mean)) 
        Sxx = np.sum((X - x_mean) ** 2)
        w = Sxy / Sxx

        #calculamos el intercepto b, usando la fórmula vista en clase que minimiza el error b = y_mean - w*x_mean
        b = y_mean - w * x_mean

        #guardo los parámetros
        self.coefficients = w
        self.intercept = b

    # This part of the model you will only need for the last part of the notebook
    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array where each column is a variable).
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # TODO: Train linear regression model with multiple coefficients
        self.intercept = None
        self.coefficients = None
        
        if np.ndim(X) == 1: #por si acaso X no es 2D, para evitar errores al multiplicar y trasponer hago comprobacion
            X = X.reshape(-1, 1)

        #segun los cálculos realizados a mano, w = (X^T*X)^(-1)*X^T*y
        #ademas para que incluya tambien el intercepto (antes b ahora lo podemos llamar w0) añado una col de 1s a X
        X_aumentada= np.hstack([np.ones((X.shape[0], 1)), X]) #añado columna de 1s a la izq de la matriz X para calcular el intercepto

        #calculamos w con la formula: w = (X^T*X)^(-1)*X^T*y
        #en numpy: trasponemos con .T y la @ se usa para multiplicar matrices. Para invertir una matriz np.linalg.inv(matriz)
        Xt_X = X_aumentada.T @ X_aumentada
        Xt_y = X_aumentada.T @ y
        w = np.linalg.inv(Xt_X) @ Xt_y

        #guardo los parametros calculados
        self.intercept = w[0]  #w0 es el intercepto
        self.coefficients = w[1:]  #el resto w son los coeficientes: w1, w2, etc.

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1: #y = b + X*w
            # TODO: Predict when X is only one variable
            predictions = self.intercept + X*self.coefficients
        else: #y = b + X*w
            # TODO: Predict when X is more than one variable
            predictions = self.intercept + X@self.coefficients

        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # R^2 Score
    # TODO: Calculate R^2
    # r^2 = 1 - (rss/tss)
    rss = np.sum((y_true - y_pred)**2)
    tss = np.sum((y_true - np.mean(y_true))**2)
    r_squared = 1 - (rss/tss)

    # Root Mean Squared Error
    # TODO: Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Error
    # TODO: Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


# ### Scikit-Learn comparison


def sklearn_comparison(x, y, linreg):
    """Compares a custom linear regression model with scikit-learn's LinearRegression.

    Args:
        x (numpy.ndarray): The input feature data (1D array).
        y (numpy.ndarray): The target values (1D array).
        linreg (object): An instance of a custom linear regression model. Must have
            attributes `coefficients` and `intercept`.

    Returns:
        dict: A dictionary containing the coefficients and intercepts of both the
            custom model and the scikit-learn model. Keys are:
            - "custom_coefficient": Coefficient of the custom model.
            - "custom_intercept": Intercept of the custom model.
            - "sklearn_coefficient": Coefficient of the scikit-learn model.
            - "sklearn_intercept": Intercept of the scikit-learn model.
    """
    ### Compare your model with sklearn linear regression model
    # TODO : Import Linear regression from sklearn
    from sklearn.linear_model import LinearRegression

    # Assuming your data is stored in x and y
    # TODO : Reshape x to be a 2D array, as scikit-learn expects 2D inputs for the features
    x_reshaped = x.reshape(-1,1)

    # Create and train the scikit-learn model
    # TODO : Train the LinearRegression model
    sklearn_model = LinearRegression() #creo una instancia de LinearRegression de sklearn
    sklearn_model.fit(x_reshaped, y)

    # Now, you can compare coefficients and intercepts between your model and scikit-learn's model
    print("Custom Model Coefficient:", linreg.coefficients)
    print("Custom Model Intercept:", linreg.intercept)
    print("Scikit-Learn Coefficient:", sklearn_model.coef_[0])
    print("Scikit-Learn Intercept:", sklearn_model.intercept_)
    return {
        "custom_coefficient": linreg.coefficients,
        "custom_intercept": linreg.intercept,
        "sklearn_coefficient": sklearn_model.coef_[0],
        "sklearn_intercept": sklearn_model.intercept_,
    }

def anscombe_quartet():
    """Loads Anscombe's quartet, fits custom linear regression models, and evaluates performance.

    Returns:
        tuple: A tuple containing:
            - anscombe (pandas.DataFrame): The Anscombe's quartet dataset.
            - datasets (list): A list of unique dataset identifiers in Anscombe's quartet.
            - models (dict): A dictionary where keys are dataset identifiers and values
              are the fitted custom linear regression models.
            - results (dict): A dictionary containing evaluation metrics (R2, RMSE, MAE)
              for each dataset.
    """
    # Load Anscombe's quartet
    # These four datasets are the same as in slide 19 of chapter 02-03: Linear and logistic regression
    anscombe = sns.load_dataset("anscombe")

    # Anscombe's quartet consists of four datasets
    # TODO: Construct an array that contains, for each entry, the identifier of each dataset
    datasets = anscombe["dataset"].unique() #los identificadores estan en la columna dataset de anscombe, visto a parte. 
    #con unique cogemos cada nombre solo una vez, ya estan ordenados

    models = {}
    results = {"R2": [], "RMSE": [], "MAE": []}
    for dataset in datasets:

        # Filter the data for the current dataset
        # TODO
        data = anscombe[anscombe["dataset"]==dataset]

        # Create a linear regression model
        # TODO
        model = LinearRegressor()

        # Fit the model
        # TODO
        X = data["x"]  # Predictor, make it 1D for your custom model
        y = data["y"] # Response
        model.fit_simple(X, y)

        # Create predictions for dataset
        # TODO
        y_pred = model.predict(X)

        # Store the model for later use
        models[dataset] = model

        # Print coefficients for each dataset
        print(
            f"Dataset {dataset}: Coefficient: {model.coefficients}, Intercept: {model.intercept}"
        )

        evaluation_metrics = evaluate_regression(y, y_pred)

        # Print evaluation metrics for each dataset
        print(
            f"R2: {evaluation_metrics['R2']}, RMSE: {evaluation_metrics['RMSE']}, MAE: {evaluation_metrics['MAE']}"
        )
        results["R2"].append(evaluation_metrics["R2"])
        results["RMSE"].append(evaluation_metrics["RMSE"])
        results["MAE"].append(evaluation_metrics["MAE"])
    return anscombe, datasets, models, results

# Go to the notebook to visualize the results
