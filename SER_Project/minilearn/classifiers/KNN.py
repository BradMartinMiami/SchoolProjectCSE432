import numpy as np

class KNNClassifierMM:

    def __init__(self, k=5):
        self.k = k

        self.X = None
        self.y = None
        self.classes = None

    def fit(self, X, y):

        #Get the columns into the arrays and then class names to classify
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.classes = np.unique(self.y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        predictions = []

        for i in range (X.shape[0]):
            first = X[i]

            difference = self.X - first
            squared = difference ** 2
            distance = np.sqrt(squared.sum(axis=1))

            sort = np.argsort(distance)
            nearest = sort[:self.k]

            neighbors_class = self.y[nearest]
        
            values, counts = np.unique(neighbors_class, return_counts=True)
            winner = values[np.argmax(counts)]
            predictions.append(winner)
        
        return np.array(predictions)
    
    