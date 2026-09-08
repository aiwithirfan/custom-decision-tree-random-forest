import numpy as np
from .decision_tree import CustomDecisionTree

class CustomRandomForest:

    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(
                len(X), len(X), replace=True
            )

            tree = CustomDecisionTree(self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array(
            [tree.predict(X) for tree in self.trees]
        )

        result = []
        for col in predictions.T:
            values, counts = np.unique(col, return_counts=True)
            result.append(values[np.argmax(counts)])

        return np.array(result)