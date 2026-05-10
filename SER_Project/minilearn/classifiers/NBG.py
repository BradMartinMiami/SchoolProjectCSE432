import numpy as np

#We assume evertyhing follows a normal distribution and 
#is independent of each other (my data set is not)
class GaussianNaiveBayesMM:
    #basic constructor
    def __init__(self):
        #class names
        self.classname = None
        #number of classes
        self.classcount = None
        #avergae feature value per class in a matrix
        self.mean = None
        #feature spread per class in a matrix
        self.variance = None


    def fit(self, X, y):
        #put both X and y into arrays
        X = np.array(X, dtype=float)
        y = np.array(y)
        #unique classes in a list
        self.classname = np.unique(y)
        #counts those number
        numberofclass = len(self.classname)
        #getting number of features
        numberoffeatures = X.shape[1]
        #getting number of samples
        numberofSamples = X.shape[0]

        #initalizaing all three of these variables with matrix the size of 
        #the class size by number of features
        self.mean = np.zeros((numberofclass, numberoffeatures))
        self.variance = np.zeros((numberofclass, numberoffeatures))
        self.classcount = np.zeros(numberofclass)

        #this is where everything happens. We loop through the number of classes
        for i in range(numberofclass):
            # C becomes the first class in the list
            c = self.classname[i]
            # This is a little magic, basically y==c finds all of the places in X where the emotion is equal to the one grabbed in C and returns the new matrix
            xcount = X[y==c]
            # This line computes all of the values per feature in columns down
            #it then grabs the mean after crushing these together
            self.mean[i] = xcount.mean(axis=0)
            #does the same thing but for varinace
            self.variance[i] = xcount.var(axis=0)
            #finds what fraction of the data is this class.
            self.classcount[i] = xcount.shape[0] / numberofSamples
        #have to do this beacuse is variance is 0 the whole thing breaks
        self.variance = self.variance + 1e-9
    
    #the predict model is a little complicated so were going line by line
    def predict(self, X):
        #arrays of the X
        X = np.array(X, dtype=float)
        #number of samples
        n_test = X.shape[0]
        #number of classes
        n_classes = len(self.classname)
        #matrix of scores setup with all 0s
        scores = np.zeros((n_test, n_classes))
        for i in range(n_classes):
            #the mean for said class
            mean = self.mean[i]
            #the variance for said class
            var = self.variance[i]
            #formula for GNB
            log_pdf = -0.5 * np.log(2 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
            #gives us the sum across for features
            log_lik = log_pdf.sum(axis=1)
            #add the score to the scores 
            scores[:, i] = log_lik + np.log(self.classcount[i])
        #pick best score
        best_idx = np.argmax(scores, axis=1)
        #return classname of best score
        return self.classname[best_idx]
    
    #same thing as method aboce
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
        
        #just turns into probabilities like in softmax
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs
    
    def score(self, X, y):
        preds = self.predict(X)
        y = np.array(y)
        return np.mean(preds == y)
 
    

