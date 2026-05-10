import numpy as np

#We assume evertyhing follows a normal distribution and 
#
class GaussianNaiveBayesMM:

    def __init__(self):
        self.classname = None
        self.classcount = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classname = np.unique(y)
        numberofclass = len(self.classname)
        numberoffeatures = X.shape[1]
        numberofSamples = X.shape[0]

        self.mean = np.zeros((numberofclass, numberoffeatures))
        self.variance = np.zeros((numberofclass, numberoffeatures))
        self.classcount = np.zeros(numberofclass)

        for i in range(numberofclass):
            c = self.classname[i]
            xcount = X[y==c]
            self.mean[i] = xcount.mean(axis=0)
            self.variance[i] = xcount.var(axis=0)
            self.classcount[i] = xcount.shape[0] / numberofSamples
        
        self.variance = self.variance + 1e-9
            
    def predict(self, X):
        X = np.array(X, dtype=float)
        n_test = X.shape[0]
        n_classes = len(self.classname)
        scores = np.zeros((n_test, n_classes))
        for i in range(n_classes):
            mean = self.mean[i]
            var = self.variance[i]
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            log_lik = log_pdf.sum(axis=1)
            scores[:, i] = log_lik + np.log(self.classcount[i])
        best_idx = np.argmax(scores, axis=1)
        return self.classname[best_idx]
    
    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        n_test = X.shape[0]
        n_classes = len(self.classname)
 
        scores = np.zeros((n_test, n_classes))
 
        for i in range(n_classes):
            mean = self.mean[i]
            var = self.variance[i]
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            scores[:, i] = log_pdf.sum(axis=1) + np.log(self.classcount[i])
        
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        return np.mean(preds == y)
 
    

