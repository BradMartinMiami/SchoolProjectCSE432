import numpy as np

class SVMMM:

    def __init__(self, C=1.0,learning=0.001, max_iters=1000):

        #the C is the amount of regularization
        self.C = C
        #Which kernel to use since we have multiple
        self.learning = learning
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.classes_ = None

    #Because there are three different kernels we have to assign values depending on each one
    def fit(self, X, y):
        # Convert labels to -1 and 1
        self.classes_ = np.unique(y)
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range (self.max_iters):
            for i in range(n_samples):
                margin = y_binary[i] * (np.dot(X[i], self.w) + self.b)
                
                if margin >= 1:
                    # only regularization
                    self.w = self.w - self.learning * self.w
                else:
                    # regularization plus hinge loss update
                    self.w = self.w - self.learning * (self.w - self.C * y_binary[i] * X[i])
                    self.b = self.b + self.learning * self.C * y_binary[i]
        
        return self
    
    def predict(self, X):
        scores = np.dot(X, self.w) + self.b
        
        predictions = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        
        return predictions
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    




    

        
        

