"""Classification example using sci-kit learn svm
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation, svm

# read data, replace missing values, drop useless id column
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# x = columns excluding class/features; y = class/label
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# split data
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

# train and fit classifier using sci-kit svm support vector classification
# Parameters:
#     C: Penalty parameter C of the error term
#     kernel:  ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#     degree: Degree of the polynomial kernel function (‘poly’).
#     gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
#     coef0: Independent term in kernel function
#     probability: Whether to enable probability estimates.
#                  This must be enabled prior to calling fit, and will slow down that method.
#     shrinking, tol, cache_size, class_weight
#     decision_function_shape : ‘ovo’, ‘ovr’ or None
#     random_state
clf = svm.SVC()
clf.fit(x_train, y_train)

# test accuracy
accuracy = clf.score(x_test, y_test)
print(accuracy)

# made up point to test the classifier
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)
