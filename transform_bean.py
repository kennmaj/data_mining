import pandas as pd

data = pd.read_csv("input_data/original/DryBeanDataset/Dry_Bean_Dataset.arff", skiprows=25, header=None)
data[16] = data[16].astype('category').cat.codes
data.index.name = 'ID'
data.to_csv("input_data/data/Dry_Bean_Dataset_modified.csv")