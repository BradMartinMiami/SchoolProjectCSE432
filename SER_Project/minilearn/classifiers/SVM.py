import numpy as np
from scipy.optimize import minimize

class SVMMM:

    def __init__(self, C=1, kernel="linear", gamma= "scale", degree=3, coef=0.0, learning=0.001, max_iters=1000):

        #the C is the amount of regularization
        self.C = C
        #Which kernel to use since we have multiple
        self.kernel = kernel
        #
        self.gamma = gamma
        self.degree = degree
        self.coef = coef
        self.learning = learning
        self.max_iters = max_iters
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None

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
        
    def compute_kernel(self, X, Z):
        n1, n2 = X.shape[0], Z.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X[i], Z[j])
        return K
    
    def fit(self, X, y):
        """Fit SVM using simple sub-gradient descent with hinge loss"""
        # Convert labels to -1 and 1
        self.classes_ = np.unique(y)
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y_binary
        
        # For linear kernel, use primal form with w directly
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            self.b = 0
            
            for i in range(self.max_iters):
                # Compute margins
                margins = y_binary * (np.dot(X, self.w) - self.b)
                
                # Find misclassified or within margin (hinge loss gradient)
                # Subgradient of max(0, 1 - y(wx + b))
                for i in range(n_samples):
                    if margins[i] < 1:  # point violates margin
                        # Update w and b
                        self.w = self.w - self.learning * (self.w - self.C * y_binary[i] * X[i])
                        self.b = self.b - self.learning * (-self.C * y_binary[i])
                    else:
                        # Only regularization
                        self.w = self.w - self.learning * self.w
                
                # Reduce learning rate over time
                self.learning *= 0.99
        else:
            # For non-linear kernels, use representer theorem approach
            # This is a simplified version using optimization
            def objective(alphas):
                # Simplified dual objective with hinge loss approximation
                K = self.compute_kernel(X, X)
                pred = np.dot(K, alphas)
                hinge_loss = np.maximum(0, 1 - y_binary * pred)
                return np.sum(hinge_loss) + 0.5 * np.dot(alphas.T, np.dot(K, alphas)) / self.C
            
            # Start with small alphas
            initial_alphas = np.zeros(n_samples)
            
            # Use scipy optimizer
            result = minimize(objective, initial_alphas, method='L-BFGS-B')
            alphas = result.x
            
            # Find support vectors (non-zero alphas)
            sv_threshold = 1e-5
            sv_indices = np.abs(alphas) > sv_threshold
            
            self.X_train = X[sv_indices]
            self.y_train = y_binary[sv_indices]
            self.alphas = alphas[sv_indices]
            
            # Calculate bias using support vectors
            if len(self.alphas) > 0:
                K_sv = self.compute_kernel(self.X_train, self.X_train)
                margins = self.y_train * np.dot(K_sv, self.alphas)
                self.b = np.mean(self.y_train - margins)
            else:
                self.b = 0
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.kernel == 'linear':
            scores = np.dot(X, self.w) - self.b
        else:
            K = self.compute_kernel(X, self.X_train)
            scores = np.dot(K, self.alphas) - self.b
        
        # Convert back to original labels
        predictions = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        return predictions
    
    def score(self, X, y):
        """Calculate accuracy"""
        return np.mean(self.predict(X) == y)
    




    

        
        

