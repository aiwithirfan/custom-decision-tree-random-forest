import numpy as np

def gini_impurity(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)


def information_gain(parent, left, right):
    weight_left = len(left) / len(parent)
    weight_right = len(right) / len(parent)

    return gini_impurity(parent) - (
        weight_left * gini_impurity(left) +
        weight_right * gini_impurity(right)
    )


def best_split(X, y):
    best_gain = -1
    best = None

    n_features = X.shape[1]

    for feature in range(n_features):
        values = np.unique(X[:, feature])

        for threshold in values:
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]

            if len(left) == 0 or len(right) == 0:
                continue

            gain = information_gain(y, left, right)

            if gain > best_gain:
                best_gain = gain
                best = (feature, threshold)

    return best
