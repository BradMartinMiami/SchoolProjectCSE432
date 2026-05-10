import numpy as np

def k_fold_split(X, y, k=5, shuffle=True, random_state=None):
    X = np.array(X)
    y = np.array(y)
    