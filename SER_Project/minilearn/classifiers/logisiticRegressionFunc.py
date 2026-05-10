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
        self.learning = lr
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

            #there is a lot happenign in this block
            #So the probs - y is the error matrix / a matrix of weights of the predicted - the true probability
            #it has the same shape of our W. so now we do X.T.dot because we have to transpose X due to dot product
            #X right now is not the size of Y. so we flip it because both have the same number of features.
            #for example if X is (2400, 496) and probs - y is (2400, 8), then it would turn into (496, 8)
            #so this new matrix is like all of the values that need to be pushed or pulled by the algorithm
            #if you have an example row for angry that has negatives in three spots those are hurting angry and should push more towards
            #angry, it does this for every class. It does this over and over across features.
            grad = X.T.dot(probs - Y) / n_samples

            #so now the W matrix is storing everything. We need to make the gradient change. So we multiply the gradient by the learning 
            #rate and then subtract it from the W to change future scores in the next loop. So everything we have been working on comes into this loop
            self.W = self.W - self.learning * grad

    #now that W is trained we can get the probabilites from softmax with one pass
    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        #same bias stacked on the first row
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        #same process here
        scores = X.dot(self.W)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probs

    #uses our previous method to get the prediction for each data point
    def predict(self, X):
        probs = self.predict_proba(X)
        best_idx = np.argmax(probs, axis=1)
        return self.classes[best_idx]

    #how well it was precdicited
    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        return np.mean(preds == y)