# Logistic regression from scratch
# week 7

import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2

        # these get set when we call fit()
        self.W = None
        self.classes = None
        self.losses = []


    def fit(self, X, y):
        # convert to numpy in case they pass lists or pandas stuff
        X = np.array(X, dtype=float)
        y = np.array(y)

        # add a column of 1s for the bias (so we don't have to track b separately)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack([ones_col, X])

        # figure out classes
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # one-hot encode y
        # ex: if y=[3,1,5] and classes are 1-8, turn each into a row of 8
        Y = np.zeros((n_samples, num_classes))
        for i in range(n_samples):
            label = y[i]
            for j in range(num_classes):
                if self.classes[j] == label:
                    Y[i, j] = 1
                    break

        # init weights small and random
        # tried zeros first but model didn't learn anything (all classes got same score)
        np.random.seed(0)
        self.W = np.random.randn(n_features, num_classes) * 0.01

        # gradient descent loop
        for step in range(self.n_iters):

            # forward pass
            scores = X.dot(self.W)

            # softmax with the max trick so exp() doesn't blow up
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # cross entropy loss (just for tracking, not used in the update)
            # add a small number inside log so we never do log(0)
            log_probs = np.log(probs + 1e-12)
            loss = -np.mean(np.sum(Y * log_probs, axis=1))

            # add L2 penalty if we're using it
            if self.l2 > 0:
                # skip the first row because that's the bias
                weight_squared = np.sum(self.W[1:] ** 2)
                loss = loss + 0.5 * self.l2 * weight_squared

            self.losses.append(loss)

            # gradient
            # this is the clean part - softmax + cross entropy gives just (P - Y)
            error = probs - Y
            grad = X.T.dot(error) / n_samples

            # add regularization gradient
            if self.l2 > 0:
                reg_grad = self.l2 * self.W
                reg_grad[0] = 0   # don't regularize bias
                grad = grad + reg_grad

            # update step
            self.W = self.W - self.lr * grad


    def predict_proba(self, X):
        X = np.array(X, dtype=float)

        # add bias column same as in fit
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack([ones_col, X])

        scores = X.dot(self.W)

        # softmax again (same as in fit, copied bc its short)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return probs


    def predict(self, X):
        probs = self.predict_proba(X)
        # pick the class with highest probability
        best_idx = np.argmax(probs, axis=1)
        return self.classes[best_idx]


    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        # mean of True/False = fraction correct
        return np.mean(preds == y)