import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve

train_path = sys.argv[1]
test_path = sys.argv[2]

iris_dataset = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
urlretrieve(iris_dataset)

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(iris_dataset, names=columns)

train, test = train_test_split(df, test_size=0.3)
train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)