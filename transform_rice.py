import pandas as pd

data = pd.read_csv("input_data/original/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff", skiprows=16, header=None)
data[7] = data[7].astype('category').cat.codes
data.index.name = 'ID'
data.to_csv("input_data/data/Rice_Cammeo_Osmancik_modified.csv")