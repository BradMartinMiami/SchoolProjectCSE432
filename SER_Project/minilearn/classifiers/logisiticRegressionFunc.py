# Logistic regression from scratch
# week 7

import numpy as np


class LogisticRegressionMM:

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

        # set when we call fit()
        self.W = None
        self.classes = None


    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)

        # add a column of 1s so the bias gets learned along with the weights
        # (saves us tracking a separate b vector)
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape

        label_idx = np.searchsorted(self.classes, y)
        Y = np.eye(n_classes)[label_idx]

        np.random.seed(0)
        self.W = np.random.randn(n_features, n_classes) * 0.01

        # gradient descent
        for step in range(self.n_iters):

            # softmax: subtract max from each row before exp
            # so we dont overflow when scores get big
            scores = X.dot(self.W)
            scores = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
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