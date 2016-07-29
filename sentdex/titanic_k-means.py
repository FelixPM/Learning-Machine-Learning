"""K-means algorithm from sci-kit learn on the titanic dataset
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')
# print(df.head())

df.drop(['body', 'name'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

# print(df.head())


def handle_non_numerical_data(dfi):
    columns = dfi.columns.values
    for column in columns:
        text_digit_val = {}

        def convert_to_int(val):
            return text_digit_val[val]
        if dfi[column].dtype != np.int64 and dfi[column].dtype != np.float64:
            column_contents = dfi[column].values.tolist()
            unique_elements = set(column_contents)
            label = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = label
                    label += 1
            dfi[column] = list(map(convert_to_int, dfi[column]))
    return dfi

df = handle_non_numerical_data(df)
# print(df.head())


x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

amount_correct = correct/len(x)
print(max(amount_correct, 1 - amount_correct)*100)
