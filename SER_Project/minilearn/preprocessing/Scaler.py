import numpy as np

class StandardScalerMM:
    #basic constructor when called with two variables
    def _init_(self):
        self.mean = None
        self.std = None

    #this method saves the mean and std value to be used by the transform
    def fit(self, X):
        #makes an array of the values
        X = np.array(X, dtype=float)
        #calculates down the row
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.std[(self.std == 0) | np.isnan(self.std)] = 1
        
        return self
    
    #basically using the saved mean and std to be able to standardized the data
    def transform(self, X):
        X = np.array(X, dtype=float)

        X_scaled = (X - self.mean) / self.std

        return X_scaled
