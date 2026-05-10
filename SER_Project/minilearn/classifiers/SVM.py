import numpy as np

#works by finding the most amount of margin between two different groups
#if you have two points on a graph what is the best line you can use to draw between them
#the distance between the two nearest points is considered the margin


class SVMMM:

    #constructor for the SVMMM
    def __init__(self, C=1.0,learning=0.001, max_iters=1000):

        #The C is the amount of punishment for misclassification
        self.C = C
        #The learning rate is the same as the learning for our logisitic regression
        self.learning = learning
        #Amount of iterations
        self.max_iters = max_iters
        #Weight vector
        self.weight = None
        #bias
        self.b = None
        #classes
        self.classes_ = None

    #the training part of the SVM. Since it says in the directions simple linear
    #the SVM will only be trained on 2 classes.
    def fit(self, X, y):
        #find the unique classes. 
        self.classes_ = np.unique(y)
        # Convert labels to -1 and 1
        # if the label = first class then -1 and 1 otherwise
        actualsign = np.where(y == self.classes_[0], -1, 1)
        
        #getting number of samples and features to be used later
        numsamples, numfeatures = X.shape
        #just creating the weight vector with 0s over the nyumber of features
        self.weight = np.zeros(numfeatures)
        #bias starts at 0
        self.b = 0
        #a for loop that loops the number of iterations
        for k in range (self.max_iters):
            #in each loop go over every sample
            for i in range(numsamples):
                #In this we do the dot product of the row of the features by the model weights for those features
                #we add bias and multiply by the actualsign of the row. We do this because if the class is +1 and we multiply and get
                #a positive value that means it is actually correct. the same thing works for the negative 1 argument for class 2.
                margin = actualsign[i] * (np.dot(X[i], self.weight) + self.b)
                
                #if margin is above 1 the point is already correct
                if margin >= 1:
                    # Now that we are sure of our correctness we make the weights smaller
                    #so over many iterations we dont have these huge weights that over fit
                    self.weight = self.weight - self.learning * self.weight
                else:
                    #we need this because it means the point has been put in incorrectly. So we must update the weights
                    #basically what were doing is putting the information into the right direction
                    #so C is the punishment rate. We basically multiplt the feature set by C times the actualsign
                    #we then multiply this by the learning rate and subtract by the original weights. so if they
                    #were negative actualsign value it would now be more negative then before.
                    self.weight = self.weight - self.learning * (self.weight - self.C * actualsign[i] * X[i])
                    #gets nuged based on the actualsign so really just adds on the first round
                    #something like -.001 or +.001 depending on what actualsign is
                    self.b = self.b + self.learning * self.C * actualsign[i]
        
        return self
    
    #works the same as logisitc regression. already have the
    #Finished self.weight so we can predict
    def predict(self, X):

        scores = np.dot(X, self.weight) + self.b
        
        predictions = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        
        return predictions
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    #for ROC scores
    def decision_function(self, X):
        """Return raw scores (before sign thresholding)."""
        X = np.array(X)
        scores = np.dot(X, self.w) + self.b
        return scores
    




    

        
        

