import pickle
import pandas as pd
import sys

file_path = sys.argv[1]
model_path = sys.argv[2]

with open(model_path, 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv(file_path)
x_test = data[data.columns[0:-1]]
y_test = data["class"]
score = model.score(x_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))