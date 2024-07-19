import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import norm, pinv

class RBF:
    def __init__(self, input_dim, num_centers, output_dim):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim

        self.centers = np.random.uniform(-1, 1, (self.num_centers, self.input_dim))
        self.beta = 1.0
        self.W = np.random.randn(self.num_centers, self.output_dim)
    
    def _basisfunc(self, c, d):
        return np.exp(-self.beta * norm(c - d) ** 2)
    
    def _calc_activations(self, X):
        G = np.zeros((X.shape[0], self.num_centers), dtype=np.float64)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
    def train(self, X, Y):
        kmeans = KMeans(n_clusters=self.num_centers).fit(X)
        self.centers = kmeans.cluster_centers_
        G = self._calc_activations(X)
        self.W = np.dot(pinv(G), Y)
    
    def predict(self, X):
        G = self._calc_activations(X)
        Y = np.dot(G, self.W)
        return Y


