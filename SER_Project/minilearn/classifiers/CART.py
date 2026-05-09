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

    