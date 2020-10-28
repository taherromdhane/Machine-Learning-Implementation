from RandomForestClassifier import RandomForestClassifier
import pandas as pd

data = pd.read_csv("simple_test_data.csv")
target = "Survived"

categorical_cols = [col for col in data.columns if col != "Survived"]

data_one_hot = pd.concat([pd.get_dummies(data[col], prefix = col) for col in categorical_cols] + [data.drop(categorical_cols, axis=1)], axis = 1)
#print(data_one_hot.head())

Tree = RandomForestClassifier(n_estimators = 20, alpha = 0.5, min_samples_leaf = 0, d = 1, max_samples = 0.5, max_features = 0.8)
Tree.fit(data = data_one_hot, target = "Survived", verbose = 1)

y = Tree.predict(data_one_hot.drop("Survived", axis=1))

#print(Tree)

print(Tree.predict_proba(data_one_hot.drop("Survived", axis=1)))

print((y == data_one_hot["Survived"]).value_counts()[True] / len(y))

print(pd.Series(y).value_counts())