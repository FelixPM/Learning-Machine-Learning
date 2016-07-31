"""Mean Shift algorithm from sci-kit learn on the titanic dataset
"""

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn import preprocessing

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

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

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i, label in enumerate(labels):
    original_df['cluster_group'].iloc[i] = label

survival_rates = {}
n_clusters_ = len(np.unique(labels))

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

for i in survival_rates:
    print(original_df[original_df['cluster_group'] == i].describe())
