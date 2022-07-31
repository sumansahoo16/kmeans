import numpy as np 


class KMEANS():
    def __init__(self, n_clusters = 2, n_iter = 250, seed = 0):

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.seed = seed

        self.centroids = None
        

    def fit(self, X):

        random_points = np.random.randint(low = 0, high = X.shape[0], size = self.n_clusters)

        self.centroids = X[random_points] 

        for i in range(self.n_iter):

            residue = np.array([ ((X - self.centroids[i]) ** 2).mean(axis = 1) for i in range(self.n_clusters)])
            residue = np.argmax(residue, axis = 0)

            self.centroids = self.centroids * 0.0

            for j in range(X.shape[0]):
                self.centroids[residue[j]] += X[j]

            values, counts = np.unique(residue, return_counts=True)


            for j in range(self.n_clusters):
                self.centroids[j] /= counts[j]

        return 0

    def centroids(self):
        return self.centroids

if __name__ == '__main__':

    model = KMEANS(n_clusters = 7)

    X = np.random.rand(2500, 31)

    model.fit(X)

    print(model.centroids)
