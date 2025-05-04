import numpy as np
import matplotlib.pyplot as plt

clustered_data = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7,5], [6, 4], [1, 2], [4, 9]])
print(clustered_data.shape)


class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
