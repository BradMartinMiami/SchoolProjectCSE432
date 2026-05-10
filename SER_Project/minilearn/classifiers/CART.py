import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class CART:

    #constructor for the CART algorithm here
    def __init__(self, max_depth=5, minsplit=2):
        #max depth so it dosent grow forever
        #dont split less then 2 samples (impossible)
        #self.tree will be used after fit
        self.max_depth = max_depth
        self.minsplit = minsplit
        self.tree = None

    #in cart we are going for the lowest amount of GINI
    #Gini helps us know how much splitting we have to do
    def gini(self, y):
        #captures the amount of labels for each
        counts = Counter(y)
        #the amount of rows
        total = len(y)

        #start from 1
        gini_score = 1

        #gets the probability per emotion label
        #calculates the gini score by probability Squared and subtracts from the gini of 1
        for label in counts:
            prob = counts[label] / total
            gini_score -= prob ** 2

        return gini_score

    #just finds the most common class, we need this for when a leaf makes a predicition
    def most_common(self, y):
        counts = Counter(y)
        #so counter is from collections and basically it makes a disctornary of 
        #object counting occurances. you have to do (1)[0][0] to get the highest occurance
        #and then [0][0] to get the name of the class in the tuple it is stored in
        return counts.most_common(1)[0][0]

    #Basically just creates the split for each tree. So based off a number
    #it splits the features into above and below that number
    def split(self, X, y, feature, middlenum):
        #the split between features given a number
        leftside = X[:, feature] <= middlenum
        rightside = X[:, feature] > middlenum

        #need each to keep the data set for left and right
        xleft = X[leftside]
        yleft = y[leftside]

        #need each to keep the data set for left and right
        xright = X[rightside]
        yright = y[rightside]

        return xleft, yleft, xright, yright

    #finds the best split over all iterations after trying everything out.
    def find_best_split(self, X, y):
        #We use these as place holders to be filled by our for loops
        best_gini = 999
        best_feature = None
        best_middlenum = None

        #need to grab the number of columns out of the feature set
        rows, cols = X.shape
        #this for loop is basicually say for every feature in our columns
        for feature in range(cols):
            #for the first feature we try every set as a threshold value
            values = np.unique(X[:, feature])

            #For every unique number in values loop through each one
            for middlenum in values:
                #splits our feature into two different sides using our split method
                xleft, yleft, xright, yright = self.split(X, y, feature, middlenum)

                #incase threshold is bad an neither side ran any data
                if len(yleft) == 0 or len(yright) == 0:
                    continue
                
                #calculates the gini for each side
                leftg = self.gini(yleft)
                rigthg = self.gini(yright)

                #Combine into the weighted average gini across both sides
                weightg = (len(yleft) / len(y)) * leftg + (len(yright) / len(y)) * rigthg

                #if its the best replace the place holder values
                if weightg < best_gini:
                    best_gini = weightg
                    best_feature = feature
                    best_middlenum = middlenum

        return best_feature, best_middlenum

    #never thought I would be dealing with recursion again after java class. But this function recursively works
    #in recursion you need conditions that kill the function and then return the data
    def build_tree(self, X, y, depth):
        # stopping conditions
        #if the depth is bigger then max_depth stop
        if depth >= self.max_depth:
            return self.most_common(y)
        #if all leafs got the same label its perfect no need to keep splitting
        if len(np.unique(y)) == 1:
            return self.most_common(y)
        #if less then min split then stop
        if len(y) < self.minsplit:
            return self.most_common(y)

        #find the best split from our best split algo
        feature, middlenum = self.find_best_split(X, y)
        #if it didnt find anything then return leaf
        if feature is None:
            return self.most_common(y)

        #use the best feature and best threshold to split the data
        xleft, yleft, xright, yright = self.split(X, y, feature, middlenum)

        #now that we have the first go and build each side of the tree down
        #will either return a dict or a class label if at a leaf. We have it as dicts
        #on each side so that you can change left or right into another dict of another feature
        #then keep doing that forever until we reach the actual tree
        left_branch = self.build_tree(xleft, yleft, depth + 1)
        right_branch = self.build_tree(xright, yright, depth + 1)

        return {
            "feature": feature,
            "middlenum": middlenum,
            "left": left_branch,
            "right": right_branch
        }

    def fit(self, X, y):
        #arrays like always
        X = np.array(X)
        y = np.array(y)
        #build the tree
        self.tree = self.build_tree(X, y, 0)

    #some recursion again for row prediciton
    def predict_row(self, row, tree):
        #if not a dict of a tree it is then just a label and must be returned
        if not isinstance(tree, dict):
            return tree
        #So we first have to grab the feature and the middlenum of the tree dict
        feature = tree["feature"]
        middlenum = tree["middlenum"]
        #if the feature in the row is less then the threshold then we go down a level to the left
        #if it bigger we go to the right
        if row[feature] <= middlenum:
            return self.predict_row(row, tree["left"])
        else:
            return self.predict_row(row, tree["right"])
        #this method keeps going until we get to the leaf value which is itself the prediction

    #just predicting each one and putting it into an array
    def predict(self, X):
        X = np.array(X)

        predictions = []
        #append the predicitons for each sample using our predict row
        for row in X:
            pred = self.predict_row(row, self.tree)
            predictions.append(pred)

        return np.array(predictions)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    #This whole method is so that we can draw our tree which is kind of crazy
    #if you think about it, it uses recursion and then mathplot lib
    def plot_tree(self, figsize=(20, 10)):
    
        # Helper: count leaves under any subtree
        def count_leaves(tree):
            #if it is not a tree dict or have a feature it is a leaf. Base case
            if not isinstance(tree, dict) or "feature" not in tree:
                return 1
            #if not a leaf go down another level on each side recusrively
            return count_leaves(tree["left"]) + count_leaves(tree["right"])
    
        # Helper: get tree depth
        def get_depth(tree):
            #again check if current node is a leaf
            if not isinstance(tree, dict) or "feature" not in tree:
                return 1
            #Keep going if not a leaf and recursively get depth
            return 1 + max(get_depth(tree["left"]), get_depth(tree["right"]))
    
        #variables that hold the leaves and depth
        total_leaves = count_leaves(self.tree)
        total_depth = get_depth(self.tree)


        #creates the plot
        fig, ax = plt.subplots(figsize=figsize)
        #X-axis from 0 to total number of leaves
        ax.set_xlim(0, total_leaves)
        #Y-axis from 0 to total depth
        ax.set_ylim(0, total_depth)
        #because its a tree we dont want to show the numbers on the axises
        ax.axis('off')

        #main method for drawing the tree
        #Tree is the tree dict we are passing through
        #X-min is the left boundary, with x_max is the right boundary
        #y is the height
        def draw(tree, x_min, x_max, y):    
            # Center x for this node
            #this is so it gets drawn in the middle of the space
            x = (x_min + x_max) / 2
        
            # This code cheks if it is a leaf.
            #if it is a leaf it grabs the predicted class
            #it also makes it as a green box
            if not isinstance(tree, dict) or "feature" not in tree:
                ax.text(x, y, f"Predict\n{tree}", 
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', edgecolor='darkgreen'),
                        fontsize=9)
                return x
        
            # Internal node
            #This grabs the feature being split
            feature = tree["feature"]
            #the value it is being split at
            middlenum = tree["middlenum"]
            #This draws the decision node, it makes the text centered
            #and also using the {middlenum} it shows the split value againt the feature
            #makes it light blue
            ax.text(x, y, f"Feature {feature}\n<= {middlenum:.3f}",
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', edgecolor='steelblue'),
                    fontsize=9)
        
            #This is used to check the amount of space avaliable under each
            #So it finds the amkount of leaves under the left and right branch
            #So more space is allocated depending on more leaves
            left_leaves = count_leaves(tree["left"])
            right_leaves = count_leaves(tree["right"])
            #This decides where the left branch ends and the right branch begins for plotting
            split_point = x_min + (left_leaves / (left_leaves + right_leaves)) * (x_max - x_min)
        
            # Recurse through and draw the left and right branches of each node
            #you subtract one from the height because your going down a level
            x_left = draw(tree["left"], x_min, split_point, y - 1)
            x_right = draw(tree["right"], split_point, x_max, y - 1)

            #This draws the lines to the left and the right of each node
            ax.plot([x, x_left], [y - 0.2, y - 0.8], 'k-', linewidth=1)
            ax.plot([x, x_right], [y - 0.2, y - 0.8], 'k-', linewidth=1)
            #This makes it so we draw the yes and no labels on each branch
            ax.text((x + x_left) / 2 - 0.1, y - 0.5, "YES", fontsize=8, color='green', ha='right')
            ax.text((x + x_right) / 2 + 0.1, y - 0.5, "NO", fontsize=8, color='red', ha='left')

            #return current position
            return x

        #plot it
        draw(self.tree, 0, total_leaves, total_depth - 0.5)
        plt.title("Custom CART Decision Tree", fontsize=14)
        plt.tight_layout()
        plt.show()
    