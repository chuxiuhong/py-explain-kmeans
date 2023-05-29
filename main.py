from sklearn.cluster import KMeans
import numpy as np
from explainkmeans import ExplainKmeansTree
from utils import read_csv
import copy

ratio = 1
data, label = read_csv("data/vehicle.csv")
label = label[:int(data.shape[0] * ratio)]
data = data[:int(data.shape[0] * ratio)]
scaled_data = copy.deepcopy(data)
for i in range(scaled_data.shape[1]):
    scaled_data[:, i] = (scaled_data[:, i] - np.min(scaled_data[:, i])) / (np.max(scaled_data[:, i]) - np.min(scaled_data[:, i]))
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(scaled_data)
centers = kmeans.cluster_centers_
et = ExplainKmeansTree()
et.centers = centers
et.expand()
et.traverse(data)
