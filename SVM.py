import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import norm

class RBF:
    def __init__(self, input_dim, num_centers, output_dim, learning_rate=0.01, lambda_param=0.01):
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param

        self.centers = np.random.uniform(-1, 1, (self.num_centers, self.input_dim))
        self.beta = 1.0
        self.W = np.random.randn(self.num_centers, self.output_dim)
    
    def _basisfunc(self, c, d):
        c = np.array(c)  
        d = np.array(d)  
        return np.exp(-self.beta * norm(c - d) ** 2)
    
    def _calc_activations(self, X):
        G = np.zeros((X.shape[0], self.num_centers), dtype=np.float64)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G
    
    def train(self, X, Y, epochs=50):
        kmeans = KMeans(n_clusters=self.num_centers).fit(X)
        self.centers = kmeans.cluster_centers_
        G = self._calc_activations(X)

        for epoch in range(epochs):
            G = self._calc_activations(X)

            Y_pred = np.dot(G, self.W)

            margin = 1 - Y * Y_pred

            error = -Y * (margin > 0).astype(float)

            dEdW = np.dot(G.T, error) + self.lambda_param * self.W

            self.W -= self.learning_rate * dEdW
    
    def predict(self, X):
        G = self._calc_activations(X)
        Y = np.dot(G, self.W)
        return np.sign(Y)
