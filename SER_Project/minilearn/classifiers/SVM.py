import numpy as np
from scipy.optimize import minimize

class SVMMM:

    def __init__(self, C=1, kernel="linear", gamma= "scale", degree=3, coef=0.0, learning=0.001, max_iter=1000):

        #the C is the amount of regularization
        self.C = C
        #Which kernel to use since we have multiple
        self.kernel = kernel
        #
        self.gamma = gamma
        self.degree = degree
        self.coef = coef
        self.learning = learning
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.X = None
        self.y = None

    #Because there are three different kernels we have to assign values depending on each one
    def kernel(self, X1, X2):

        if (self.kernel == "linear"):
            return np.dot(X1, X2)
        
        elif self.kernel == "rbf":
            if self.gamma == "scale":
                gamma = 1.0 / X1.shape[0]
            elif self.gamma == "auto":
                gamma = 1.0 / X1.shape[0]
            else:
                gamma = self.gamma
            diff = X1 - X2
            return np.exp(-gamma * np.dot(diff, diff))
        
        elif self.kernel == "poly":
            return (np.dot(X1, X2) + 1) ** self.degree
        
    def compute_kernel(self, X, i):
        n1, n2 = X.shape[0], i.shape[0]
        
    

        
        

