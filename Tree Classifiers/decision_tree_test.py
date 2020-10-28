from DecisionTreeClassifier import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv("simple_test_data.csv")
target = "Survived"

categorical_cols = [col for col in data.columns if col != "Survived"]

data_one_hot = pd.concat([pd.get_dummies(data[col], prefix = col) for col in categorical_cols] + [data.drop(categorical_cols, axis=1)], axis = 1)
print(data_one_hot.head())

Tree = DecisionTreeClassifier(alpha = 0.5, min_samples_leaf = 5, d = 0)
Tree.fit(data = data_one_hot, target = "Survived", verbose = True)

y = Tree.predict(data_one_hot.drop("Survived", axis=1))

#print(Tree)

print((y == data_one_hot["Survived"]).value_counts()[True] / len(y))

print(y.value_counts())