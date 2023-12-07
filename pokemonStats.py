import numpy as np
import pandas as pd

data = pd.read_csv("MakinaOgrenmesi/Pokemon.csv")
# print(data.head())
# print(data.index)
# print(len(data))

# print(data.HP)

# print(data.HP.mean())
# print(data.HP.max())
# print(data[data["Attack"] == data["Attack"].max()])

# print(data[data["Name"] == "Pikachu"]["Defense"].iloc[0])


print(data.groupby("Type 1").mean())

print(data["Type 1"].nunique())

print(data["Type 1"].value_counts())