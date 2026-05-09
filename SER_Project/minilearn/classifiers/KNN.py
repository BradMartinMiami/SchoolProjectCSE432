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