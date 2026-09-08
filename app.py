import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from custom_tree.random_forest import CustomRandomForest

st.title("Custom Random Forest From Scratch")

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

if st.button("Train Model"):
    model = CustomRandomForest(n_trees=5, max_depth=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    st.write("Accuracy:", accuracy_score(y_test, pred))
    st.write(pd.DataFrame({
        "Actual": y_test,
        "Predicted": pred
    }))