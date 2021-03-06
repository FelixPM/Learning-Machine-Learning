"""Linear Regression code
Find slope and intercept
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(how_many_points, variance, step=2, correlation='pos'):
    val = 1
    ys = []
    for i in range(how_many_points):
        yt = val + random.randrange(-variance, variance)
        ys.append(yt)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    """
    Calculates slope and intercept from input points
    :param xs: x points
    :param ys: y points
    :return: slope m, intercept b
    """
    get_m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) /
             ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    get_b = np.mean(ys) - get_m * np.mean(xs)
    return get_m, get_b


def squared_error(y_orig, y_line):
    """
    Total sum of squared errors
    :param y_orig: original y points
    :param y_line: fitted y points
    :return: int square error
    """
    return sum((y_line - y_orig) ** 2)


def coefficient_of_determination(y_orig, y_line):
    """
    :param y_orig: original y points
    :param y_line: fitted y points
    :return: int coefficient of determination
    """
    y_mean_line = np.array([np.mean(y_orig) for _ in y_orig])
    square_error_regression = squared_error(y_orig, y_line)
    square_error_y_mean = squared_error(y_orig, y_mean_line)
    return 1 - (square_error_regression / square_error_y_mean)


# Points
# x = np.array([1, 2, 3, 4, 5, 6])
# y = np.array([5, 4, 6, 5, 6, 7])

x, y = create_dataset(40, 10, 2, 'pos')

# Fit and predict
m, b = best_fit_slope_and_intercept(x, y)

regression_line = np.array([(m * x1) + b for x1 in x])

predict_x = 45
predict_y = m * predict_x + b

r_squared = coefficient_of_determination(y, regression_line)
print(r_squared)

# Plot
plt.scatter(x, y)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(x, regression_line)
plt.show()
