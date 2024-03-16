from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
import sys


file_path = sys.argv[1]
model_path = sys.argv[2]

data = pd.read_csv(file_path)
x_train = data[data.columns[0:-1]]
y_train = data["class"]

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

with open(model_path, 'wb') as file:
    pickle.dump(model, file)