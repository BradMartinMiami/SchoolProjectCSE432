import numpy as np
from collections import Counter


class CART:

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):
        counts = Counter(y)
        total = len(y)

        gini_score = 1

        for label in counts:
            prob = counts[label] / total
            gini_score -= prob ** 2

        return gini_score


    def most_common(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]


    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        return X_left, y_left, X_right, y_right

    def find_best_split(self, X, y):
        best_gini = 999
        best_feature = None
        best_threshold = None

        rows, cols = X.shape

        for feature in range(cols):
            values = np.unique(X[:, feature])

            for threshold in values:
                X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_gini = self.gini(y_left)
                right_gini = self.gini(y_right)

                weighted_gini = (len(y_left) / len(y)) * left_gini + (len(y_right) / len(y)) * right_gini

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth):
        # stopping conditions
        if depth >= self.max_depth:
            return self.most_common(y)

        if len(np.unique(y)) == 1:
            return self.most_common(y)

        if len(y) < self.min_samples_split:
            return self.most_common(y)

        feature, threshold = self.find_best_split(X, y)

        if feature is None:
            return self.most_common(y)

        X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)

        left_branch = self.build_tree(X_left, y_left, depth + 1)
        right_branch = self.build_tree(X_right, y_right, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_branch,
            "right": right_branch
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.tree = self.build_tree(X, y, 0)

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = tree["feature"]
        threshold = tree["threshold"]

        if row[feature] <= threshold:
            return self.predict_row(row, tree["left"])
        else:
            return self.predict_row(row, tree["right"])

    def predict(self, X):
        X = np.array(X)

        predictions = []

        for row in X:
            pred = self.predict_row(row, self.tree)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)