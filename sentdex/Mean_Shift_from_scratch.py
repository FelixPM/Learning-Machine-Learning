"""Mean Shift algorithm in python from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
import random

centers = random.randrange(2, 4)
style.use('ggplot')

x, y = make_blobs(n_samples=30, centers=centers, n_features=2)

# x = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [8, 2],
#               [10, 2],
#               [9, 3],
#               [9, 11]])

# plt.scatter(x[:, 0], x[:, 1], s=150)
# plt.show()
colors = 10*['g', 'r', 'c', 'b', 'k']


class MeanShift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):

        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i, j in enumerate(data):
            centroids[i] = j
        weights = [i for i in range(self.radius_norm_step)][::-1]
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.000000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                if i in uniques:
                    uniques.remove(i)

            prev_centroids = dict(centroids)
            centroids = {}
            for i, j in enumerate(uniques):
                centroids[i] = np.array(j)

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

            if optimized:
                break

        self.centroids = centroids
        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = (distances.index(min(distances)))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification

clf = MeanShift()
clf.fit(x)

centroids1 = clf.centroids

for classification1 in clf.classifications:
    color = colors[classification1]
    for featureset1 in clf.classifications[classification1]:
        plt.scatter(featureset1[0], featureset1[1], marker='x', color=color, s=150, linewidths=5)

for c in centroids1:
    plt.scatter(centroids1[c][0], centroids1[c][1], color='k', marker='*', s=150)

plt.show()
