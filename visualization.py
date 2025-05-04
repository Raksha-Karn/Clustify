from sklearn.datasets import make_blobs
from main import KMeansClustering
import matplotlib.pyplot as plt

data = make_blobs(n_samples=100, n_features=2, centers=3)
random_points = data[0]

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(random_points)
print(data[1])

plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)
plt.show()
