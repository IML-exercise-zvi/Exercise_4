import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import  BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR



#  from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

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
        Plot's x-axis range

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
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
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
    for name, module in {"L1": L1, "L2": L2}.items():
        results = {}
        for eta in etas:
            # Retrieve a new callback function
            callback, vals, weights = get_gd_state_recorder_callback()
            # Use GD to minimize the given objective, with specified fixed learning rate
            GradientDescent(learning_rate=FixedLR(eta), callback=callback).fit(module(weights=np.copy(init)), None, None)
            results[eta] = (vals, weights)
            # Plot algorithm's descent path
            plot_descent_path(module, np.array([init] + weights), f"{name} - Learning Rate: {eta} ")\
                .write_image(f"../figures/gd_{name}_eta_{eta}.png")

        # Plot algorithm's convergence for the different values of eta
        fig = go.Figure(layout=go.Layout(xaxis=dict(title="GD Iteration"),
                                         yaxis=dict(title="Norm"),
                                         title=f"{name} GD Convergence For Different Learning Rates"))
        for eta, (v, _) in results.items():
            fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, mode="lines", name=rf"$\eta={eta}$"))
        fig.write_image(f"gd_{name}_fixed_rate_convergence.png")




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
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    predictions = logistic_regression.predict(X_test)
    accuracy = np.mean(predictions == y_test)

    _, values, weights = get_gd_state_recorder_callback()
    fig = go.Figure([go.Scatter(x=list(range(len(values))), y=values, mode='lines', name='Convergence Rate')])
    fig.update_layout(title='Convergence Rate of Logistic Regression', xaxis_title='Iteration', yaxis_title='Objective Value')
    fig.show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify
    # values of regularization parameter
    l1_model = LogisticRegression(penalty='l1')
    l2_model = LogisticRegression(penalty='l2')

    # Cross-validation to determine the best regularization parameter
    # Note: Implement cross-validation logic here

    l1_model.fit(X_train, y_train)
    l2_model.fit(X_train, y_train)

    l1_predictions = l1_model.predict(X_test)
    l2_predictions = l2_model.predict(X_test)

    l1_accuracy = np.mean(l1_predictions == y_test)
    l2_accuracy = np.mean(l2_predictions == y_test)

    # Plotting results
    l1_fig = go.Figure([go.Scatter(x=list(range(len(l1_model.loss_history))), y=l1_model.loss_history, mode='lines', name='L1 Convergence Rate')])
    l1_fig.update_layout(title='Convergence Rate of L1-Regularized Logistic Regression', xaxis_title='Iteration', yaxis_title='Objective Value')
    l1_fig.show()

    l2_fig = go.Figure([go.Scatter(x=list(range(len(l2_model.loss_history))), y=l2_model.loss_history, mode='lines', name='L2 Convergence Rate')])
    l2_fig.update_layout(title='Convergence Rate of L2-Regularized Logistic Regression', xaxis_title='Iteration', yaxis_title='Objective Value')
    l2_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
