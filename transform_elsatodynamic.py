import pandas as pd

data = pd.read_csv("input_data/original/2d+elastodynamic+metamaterials/data.csv")[["BandGapLocation","BandGapWidth"]]
data.columns = [0, 1]
data.index.name = 'ID'
data.to_csv("input_data/data/2d_elastodynamic_metamaterials.csv")