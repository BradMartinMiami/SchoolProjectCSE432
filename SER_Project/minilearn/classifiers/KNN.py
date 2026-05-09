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
    

    def predict_proba(self, X):
        # for ROC curves we need a probability per class
        # for kNN we just use the fraction of the k neighbors that are each class
        X = np.array(X, dtype=float)
        n_test = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_test, n_classes))
 
        for i in range(n_test):
            test_point = X[i]
 
            # same distance calculation as predict()
            diffs = self.X - test_point
            dists = np.sqrt((diffs ** 2).sum(axis=1))
 
            sorted_idx = np.argsort(dists)
            nearest = sorted_idx[:self.k]
            neighbor_labels = self.y[nearest]
 
            # for each class, what fraction of the k neighbors are that class
            for j in range(n_classes):
                probs[i, j] = np.mean(neighbor_labels == self.classes[j])
 
        return probs
    
    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        return np.mean(preds == y)
    

    
