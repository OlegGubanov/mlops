import pandas as pd
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve


iris_dataset = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
urlretrieve(iris_dataset)

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(iris_dataset, names=columns)

train, test = train_test_split(df, test_size=0.3)
train.to_csv("train/train.csv", index=False)
test.to_csv("test/test.csv", index=False)