# Logistic regression from scratch
# week 7

import numpy as np

#logisitc regression just takes a row of features and then outputs a probability for each class,
#and then picks to class with the highest probability.
class LogisticRegressionMM:

    #the consturctor, learning rate is just the amount you go per iteration.
    #n_iters is the amount of times you go through the list and train basically.
    #like the amount of time you do the learning rate
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

        # set when we call fit()
        #W is just the weights and classes is the classes.
        self.W = None
        self.classes = None


    def fit(self, X, y):
        #just conveting into the numpy arrays
        X = np.array(X, dtype=float)
        y = np.array(y)

        # add a column of 1s so the bias gets learned along with the weights
        # (saves us tracking a separate b vector)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        #Saves the classes, the number of thos classes, adn then the number of samples/features
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        #so basically we are creating dummy variables here. Instead of having like angry it would change it 
        #into a 0, to be able to pass onto the algorithm.
        label_idx = np.searchsorted(self.classes, y)
        #now y becomes a 2d matrix of those vectors with a 1 in a certain spot representing the true emotion class
        #you have to pass the label into it because before it is just an identity matrix and now the position of the 0
        #lines up to the the sorted position in the classes.
        Y = np.eye(n_classes)[label_idx]

        #setting seed so this is reproduciable
        np.random.seed(0)
        #Builds a matrix of the number of features by classes and then buts random small
        #weights into each space. It then is very small because of the .01, need small probabilities
        #so everything works correctly and dosent generate insane numbers
        self.W = np.random.randn(n_features, n_classes) * 0.01

        # gradient descent
        #runs the number of times of iterations
        for step in range(self.n_iters):

            # softmax: subtract max from each row before exp
            # so we dont overflow when scores get big
            #score is the sum of the multiplication of the array of features by our weights
            #you do this for every emotion and the sum is kept for next time
            scores = X.dot(self.W)
            #now we are changing the scores into probabilities so we subtract score from max to keep low
            scores = scores - scores.max(axis=1, keepdims=True)
            #we then expenitate eacg
            exp_scores = np.exp(scores)
            #now we divide each by the sum of the row to get the probability adding up to 1
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)


            grad = X.T.dot(probs - Y) / n_samples

            self.W = self.W - self.lr * grad


    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        scores = X.dot(self.W)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs


    def predict(self, X):
        probs = self.predict_proba(X)
        best_idx = np.argmax(probs, axis=1)
        return self.classes[best_idx]


    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        return np.mean(preds == y)