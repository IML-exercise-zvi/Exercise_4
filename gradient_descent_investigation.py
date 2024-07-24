import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from loss_functions import misclassification_error
from cross_validate import cross_validate
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's y-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="blue")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(value, weight, **kwargs):
        values.append(value)
        weights.append(weight)
    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module in [L1, L2]:
        results = {}
        for eta in etas:
            callback, vals, weights = get_gd_state_recorder_callback() #record the objective value and parameters
            model = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            model.fit(module(weights=np.copy(init)), np.array([0]), np.array([0]))
            results[eta] = (vals, weights)

            if eta == .01:  # Plotting descent path for the learning rate of 0.01 as requested
                plot = plot_descent_path(module, np.array([init] + weights), title=f" - Fixed Learning Rate: {eta}")
                plot.write_html(f"GD_{module.__name__}_fixed_rate_path.html")

        # Plotting convergence of the algorithm for different learning rates
        fig = go.Figure(layout=go.Layout(xaxis=dict(title="Gradient Descent Iteration"),
                                         yaxis=dict(title="Norm"),
                                         title=f"Convergence of {module.__name__}"))
        colors = ['red', 'green', 'blue', 'purple']
        for eta, color in zip(results.keys(), colors):
            fig.add_trace(go.Scatter(x=list(range(len(results[eta][0]))), y=results[eta][0], mode="lines", name="eta={eta}", line=dict(color=color)))
        fig.write_html(f"GD_{module.__name__}_fixed_rate_convergence.html")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    X_train, y_train, X_test, y_test = load_data() # Load data and split into train and test sets

    # Fitting logistic regression with fixed learning rate
    callback, losses, weights = get_gd_state_recorder_callback()
    max_iter = 20000
    lr = 1e-4
    gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter, callback=callback) # Initialize Gradient Descent
    model = LogisticRegression(solver=gd).fit(X_train.values, y_train.values) # Fit logistic regression model

    y_prob = model.predict_proba(X_train.values) # Predict probabilities of positive class

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="purple", dash='dash'), showlegend=False),
               go.Scatter(x=fpr, y=tpr, mode='lines', text=thresholds, showlegend=False,
                          hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
              layout=go.Layout(title=rf"$\text{{ROC Curve - Logistic Regression - AUC}}={auc(fpr, tpr):.3f}$",
                               width=400, height=400, xaxis=dict(title="FPR"), yaxis=dict(title="TPR")))\
            .write_html(f"gd_logistic_roc_lr{lr:.4f}.html")

    model.alpha_ = thresholds[np.argmax(tpr-fpr)] # Set threshold to maximize TPR-FPR

    lamdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty, lambdas in [("l1", lamdas),
                             ("l2", lamdas)]:
        # Running cross validation
        scores = np.zeros((len(lambdas), 2))
        for i, lamda in enumerate(lambdas):

            gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
            logistic = LogisticRegression(solver=gd, penalty=penalty, lam=lamda, alpha=.5)
            scores[i] = cross_validate(estimator=logistic, X=X_train.values, y=y_train.values,
                                       scoring=misclassification_error)

        # Selecting optimal lambda
        fig = go.Figure([go.Scatter(x=lambdas, y=scores[:, 0], name="Train Error", line=dict(color='purple')),
                         go.Scatter(x=lambdas, y=scores[:, 1], name="Validation Error", line=dict(color='green'))],
                        layout=go.Layout(
                            title="Train and Validation errors for Logistic Regression",
                            xaxis=dict(title="Lambda"),
                            yaxis=dict(title="Error Rate")))
        fig.write_html(f"{penalty}_logistic_regression_errors.html")

        # fitting a model with the best lambda on the entire train set
        lam_opt = lambdas[np.argmin(scores[:, 1])]
        gd = GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter)
        model = LogisticRegression(solver=gd, penalty=penalty, lam=lam_opt, alpha=.5).fit(X_train.values, y_train.values)

        # Get predictions
        train_pred = model.predict(X_train.values)
        test_pred = model.predict(X_test.values)

        print(f"Optimal lambda for {penalty} penalty: {lam_opt}, Train Error: {misclassification_error(y_train.values, train_pred):.3f}, Test Error: {misclassification_error(y_test.values, test_pred)}")



if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
