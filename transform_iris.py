import pandas as pd

data = pd.read_csv("input_data/original/iris/iris.data", header=None)
data[4] = data[4].astype('category').cat.codes
data.index.name = 'ID'
data.to_csv("iris_modified.csv")