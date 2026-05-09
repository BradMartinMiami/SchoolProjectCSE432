import numpy as np

class Cart:

    def __init__(self, max_depth = 8, min_split = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.tree = None

    