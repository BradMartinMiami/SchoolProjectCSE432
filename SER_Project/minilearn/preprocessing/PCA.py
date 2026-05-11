import numpy as np

#PCA is a dimenson reduction technique. For out large amount of features
#it should give us back far fewer features to use for our models. PCA
#finds components which are new combined features to capture highly correlated
#features and use them more efficiently. It finds using variance

class PCA:
    #basic construction.
    #The first component is the is that of maximum variance, and the 2nd is the 2nd most
    def __init__(self, n_components=2):
        #This is the amount of componets to keep
        self.n_components = n_components
        #These four are placeholders for the information that will
        #come in the fit method
        #Mean is per feature average
        self.mean = None
        #The directions of component matrix
        self.components = None
        #How much variance each component has
        self.explained_variance = None
        #variance as the fraction of the total
        self.explained_variance_ratio = None

    #the fit method for the PCA
    def fit(self, X):
        X = np.array(X)

        #get the mean of each feature across X       
        self.mean = np.mean(X, axis=0)
        #We now have the feautre set subtracted by the mean
        #We need this because PCA is calculated by variance
        #so when we center and bring the mean to 0 the PCA can find
        #variation
        X_centered = X - self.mean

        # covariance matrix
        #basically find the correlation between features using this
        #covariance matrix and get the Variance of the feature on the diaganol
        cov_matrix = np.cov(X_centered, rowvar=False)

        # eigenvalues and eigenvectors
        #lot to unpack here, the eigenvalues is a 1d matrix or vector that
        #tells you how much variance is the data has along each eigenvector direction
        #The eigenvectoes is the size of the cov_matrix and each column is a directon 
        #Eigenvectors are the principal componets that have the direction
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort from biggest eigenvalue to smallest, because .eigh returns the opposite way
        sorted_indexes = np.argsort(eigenvalues)[::-1]
        #This now stores biggest first and the vectors now store the same order as values.
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:, sorted_indexes]

        # keep only the first n components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        #get the variance through the eigenvalues and store the explained ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

        return self

    def transform(self, X):
        #array
        X = np.array(X)
        #Centering the new data on the already found mean
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    