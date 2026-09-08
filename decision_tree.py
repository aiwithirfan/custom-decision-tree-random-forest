import numpy as np
from .gini import best_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class CustomDecisionTree:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._grow(X, y, 0)

    def _grow(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)

        if depth >= self.max_depth or len(classes) == 1:
            return Node(value=classes[np.argmax(counts)])

        split = best_split(X, y)

        if split is None:
            return Node(value=classes[np.argmax(counts)])

        feature, threshold = split

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        return Node(
            feature,
            threshold,
            self._grow(X[left_idx], y[left_idx], depth+1),
            self._grow(X[right_idx], y[right_idx], depth+1)
        )

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])