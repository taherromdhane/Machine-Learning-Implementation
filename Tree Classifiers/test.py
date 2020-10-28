from DecisionTreeClassifier import DecisionTreeClassifier
import pandas as pd

data = pd.read_csv("simple_test_data.csv")

Tree = DecisionTreeClassifier(alpha = 0.5, min_samples_leaf = 5, d = 0)
Tree.build(data = data, target = "Survived", verbose = True)

y = Tree.predict(data[[col for col in data.columns if col != "Survived"]])

#print(Tree)

print((y == data["Survived"]).value_counts()[True] / len(y))