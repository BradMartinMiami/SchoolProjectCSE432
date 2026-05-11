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

        

    