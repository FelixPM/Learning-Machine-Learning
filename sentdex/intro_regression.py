"""Predicting stock prices using machine learning
"""
import datetime
import math
import pickle
import Quandl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression


style.use('ggplot')

# import alphabet stocks data set into data frame
# df = Quandl.get('WIKI/GOOGL')
# with open('wiki_googl.pickle', 'wb') as file:
#     pickle.dump(df, file)
pickle_in = open('wiki_googl.pickle', 'rb')
df = pickle.load(pickle_in)
# file.close()

# Grab desired features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Create new features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0  # High Low percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0  # Percent change

# Filter out features
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'  # label variable

# Fill missing data
df.fillna(-99999, inplace=True)

# set label to be forecast_col certain percent of data len

forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

# Using numpy array X=features Y=label
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]

df.dropna(inplace=True)
Y = np.array(df['label'])

# split data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

# Fit classifier on train data
# clf = LinearRegression()
# clf.fit(X_train, Y_train)
#
# # Save classifier as pickle
# with open('linear_regression.pickle', 'wb') as file:
#     pickle.dump(clf, file)
pickle_in = open('linear_regression.pickle', 'rb')
clf = pickle.load(pickle_in)

# Test classifier on test data
accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
