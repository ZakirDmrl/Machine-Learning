import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/SampleSuperstore.csv")
# print(data.head())
# print(data.shape)
# print(data.describe())
# print(data.columns)
# print(data.isnull().sum()) 

# print(data[data["Ship Mode"] == "Second Class"].value_counts)
data.groupby("Ship Mode").boxplot()

plt.show()