import numpy as np

#Basic function definition basically how test train split works but with more folds
#so we add K which is number of folds
def k_fold_split(X, y, k=5, shuffle=True, random_state=None):
    X = np.array(X)
    y = np.array(y)
    #number of rows in features
    numberofsamples = len(X)
    #get the index of the aranged number of feature rows
    indices = np.arange(numberofsamples)
    #need this for shuffle later on so it is reproducible by the number
    if random_state is not None:
        np.random.seed(random_state)

    #get all of the class names uninquely.
    classes = np.unique(y)

    #This creates k empty lists, so for 5 folds would create 5 lists. The first
    #list through the last list is where the each folds test  will go into
    #for example for fold 0 it would be test on 0 and train on 1,2,3,4
    #second would go fold 1 it would be test on 1 and train 0,2,3,4
    #training gets computed later
    folds = [[] for _ in range(k)]

    #Most important part of the whole operation
    #first loop through the classes one at a time
    for class_label in classes:
        #this is a boolean array in which we have implemented previously
        #just returns the rows where the class is the one were looping through
        class_indices = np.where(y == class_label)[0]
        #randomizes the indices of those rows
        if shuffle:
            np.random.shuffle(class_indices)
        #Splits the indices into the amount of folds. So if 400 happy indices
        #and k=5, you get 5 chunks of ~80 happy indices each
        split_indices = np.array_split(class_indices, k)
        #take each chunk and assign it to the matching fold's test bucket
        #so fold 0 gets chunk 0 of happy, fold 1 gets chunk 1 of happy, etc.
        #after doing this for every class each fold ends up with a proportional
        #slice of every class
        for fold_num in range(k):
            folds[fold_num].extend(split_indices[fold_num])
    #now we need to package the train/test pairs
    final_folds = []
    #all sample indices from 0 to n-1, used to compute train indices below
    all_indices = np.arange(len(X))
    #loop through each fold and build its (train, test) pair
    for fold_num in range(k):
        #the test indices are the ones we filled into folds[fold_num] above
        test_indices = np.array(folds[fold_num])
        #train indices are everything else. setdiff1d does all_indices - test_indices
        #so we get back every index that isnt in the test set
        train_indices = np.setdiff1d(all_indices, test_indices)
        #store as a tuple so we can get it as (test,train)
        final_folds.append((train_indices, test_indices))
    #return list of k tuples, each with (train indices, test indices)
    return final_folds


