import numpy as np

def train_test_split(X,y,test_size = 0.2, random_state=None, shuffle=True):

    #must make these again to be able to use arrays
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError("Data aint the same across X and Y. somethings wrong")
    
    rows = np.arange(len(X))

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(rows)

    test_count = int(len(X) * test_size)

    test_rows = rows[:test_count]
    train_rows = rows[test_count:]

    X_train = X[train_rows]
    X_test = X[test_rows]

    y_train = y[train_rows]
    y_test = y[test_rows]

    return X_train, X_test, y_train, y_test
