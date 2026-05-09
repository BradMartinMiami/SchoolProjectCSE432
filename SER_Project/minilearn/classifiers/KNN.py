import numpy as np

class KNNClassifierMM:

    #Default constructor with the K being the number of neighbors and then also 
    #the X,y, and classes will be filled in, in fit.
    def __init__(self, k=5):
        self.k = k

        self.X = None
        self.y = None
        self.classes = None

    def fit(self, X, y):

        #Get the columns into the arrays and then class names to classify
        #the self.X get the features into and array and Y gets the labels.
        #different then other algos because KNN is a lazy learner and just memorizes all
        self.X = np.array(X, dtype=float)
        self.y = np.array(y)
        self.classes = np.unique(self.y)

    #the heart of the KNN
    def predict(self, X):
        X = np.array(X, dtype=float)
        #where to save predicitons to
        predictions = []

        #this loop is intuitvely complicated. Basically what is happening is that you have one point your testing
        #every single time. the distance is them computed against every other feature. Those points are then 
        #organized by how close they are to the point. Then we get the winner by checking the N-closest labels
        #and then taking that and classifying our predicition of the point.
        for i in range (X.shape[0]):
            test_point = X[i]

            #difference from all features vs test-point
            difference = self.X - test_point
            #square the distance of features to get rid of negatives and -s
            squared = difference ** 2
            #we sum the row full of numbers and SQRT it to get the distance from our test point
            distance = np.sqrt(squared.sum(axis=1))

            #sorting by closest distance
            sort = np.argsort(distance)
            #nearest is the sort of the closest # of ks
            nearest = sort[:self.k]

            neighbors_class = self.y[nearest]
            #counting the emotion class of the closest ks
            values, counts = np.unique(neighbors_class, return_counts=True)
            #getting the winner from number of max class
            winner = values[np.argmax(counts)]
            #append predictions with features
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
    

    
