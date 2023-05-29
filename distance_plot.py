from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
from explainkmeans import ExplainKmeansTree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from utils import read_csv


data,labels = read_csv("data/vehicle.csv")
plot_x, plot_y = [], []

for i in range(data.shape[1]):
    data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i]) - np.min(data[:,i]))



for c in range(1, 10):
    kmeans = KMeans(n_clusters=c, random_state=0, n_init="auto").fit(data)
    centers = kmeans.cluster_centers_
    distances = 0
    for i in range(len(data)):
        distances += np.linalg.norm(centers[kmeans.labels_[i]] - data[i])
    plot_x.append(c)
    plot_y.append(distances)

fig, ax = plt.subplots()
ax.plot(plot_x, plot_y)

ax.set(xlabel='cluster size', ylabel='sum of distances',
       title='Kmeans distance')
ax.grid()

fig.savefig("output/distance.png")
plt.show()
