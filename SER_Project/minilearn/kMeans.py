import numpy as np

#Whats interesting about K means is that it does not look at the
#labels it only looks at our features and tries to find groupings
class KMeans:

    #Constructor for K-means. THe N_clusters is the number of these groupings
    #Max_iters is the amount of iterations to run, and random state is randomness
    def __init__(self, n_clusters=8, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        #Centroids are the cluster centers. you want points to group around these
        #centroids. It starts random on those and finds them over time
        self.centroids = None
        #The final cluster for each sample
        self.labels_ = None

    #Basically what were doing is assiginging all of the features to their nearest centroid
    #Then we keep moving each centroid to the average of its points and then repeat
    #Until centroids stop moving all together or max iters stop
    def fit(self, X):
        #An array of X
        X = np.array(X)

        #If it isnt none we get take from the see dso reproducible
        if self.random_state is not None:
            np.random.seed(self.random_state)

        #Gets the number of rows
        n_samples = X.shape[0]

        # randomly choose starting centroids from actual rows in the data
        random_indexes = np.random.choice(n_samples,self.n_clusters,replace=False)
        #Uses the indices to grab 8 random rows to act as the intial centroids
        self.centroids = X[random_indexes]

        #Loop for the amount of iterations
        for i in range(self.max_iters):

            # assign each row to the closest centroid, returns an array of those cluster ids
            labels = self._assign_clusters(X)

            # move centroids to the average of their assigned points
            new_centroids = self._update_centroids(X, labels)

            # stop if centroids barely changed, no need to keep going
            #because were at a good stop
            if np.allclose(self.centroids, new_centroids):
                break
            #Every loop assign the new centroids as the ones for next loop
            self.centroids = new_centroids

        #Assign the final centroids/ clustering output
        self.labels_ = self._assign_clusters(X)

        return self

    #this is helping us do a lot in the fit method. This method
    #helps us actually assign points to each cluster
    def _assign_clusters(self, X):
        #The list to be updated
        labels = []
        #Take a row of features at a time
        for row in X:
            #List of distrances to be filled in
            distances = []
            #For each centroid in our list of centroids
            for centroid in self.centroids:
                #This is the euclidian distance formula from KNN. Basically subtract
                #feature by feature, square the difference, sun and then SQRT per each centroid
                distance = np.sqrt(np.sum((row - centroid) ** 2))
                #We add the distance
                distances.append(distance)
            #Then for this feature set we take the smallest distance and then
            #Assign it to its closest cluster
            closest_cluster = np.argmin(distances)
            labels.append(closest_cluster)

        return np.array(labels)

    #Another method that helps us is the update_centroids so that we can move
    #Centroids per iteration closer to the average of its assgined points
    def _update_centroids(self, X, labels):
        #Empty list to be filled
        new_centroids = []
        #loop through the number of clusters
        for cluster_num in range(self.n_clusters):
            #This is a boolean array where we take the rows assigned to this cluster
            points_in_cluster = X[labels == cluster_num]

            #keep the same so we dont get a divide by 0 error
            if len(points_in_cluster) == 0:
                # if no points are assigned, keep the old centroid
                new_centroids.append(self.centroids[cluster_num])
            #Computes the mean of each rows and then has a number per feature
            #This is the new centroid for that cluster
            else:
                new_centroids.append(points_in_cluster.mean(axis=0))

        return np.array(new_centroids)

    #Predicts using our assign clusters method
    def predict(self, X):
        X = np.array(X)
        return self._assign_clusters(X)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_