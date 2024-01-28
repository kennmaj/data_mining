import pandas as pd

data_red = pd.read_csv("input_data/original/wine+quality/winequality-red.csv", delimiter=";", skiprows=1, header=None)
data_white = pd.read_csv("input_data/original/wine+quality/winequality-white.csv", delimiter=";", skiprows=1, header=None)

data = pd.concat([data_red, data_white], ignore_index=True)
data.index.name = 'ID'
data.to_csv("input_data/data/wine_quality.csv")