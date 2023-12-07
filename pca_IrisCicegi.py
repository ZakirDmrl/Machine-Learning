import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Iris.csv")

x = data.iloc[:, 0:-1]  
y = data.iloc[:, -1]  
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(x)

from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
print(principalDf.head())

son = pd.concat([principalDf,data[["Species"]]], axis=1)

# Create separate DataFrames for each class
setosa = son[son["Species"] == "Iris-setosa"]
versicolor = son[son["Species"] == "Iris-versicolor"]
virginica = son[son["Species"] == "Iris-virginica"]

plt.figure(figsize=(8,6))

# Scatter plot for Iris Versicolour
plt.scatter(versicolor['Principal Component 1'], versicolor['Principal Component 2'], label="Versicolor")
plt.scatter(setosa['Principal Component 1'], setosa['Principal Component 2'], label="Setosa")
plt.scatter(virginica['Principal Component 1'], virginica['Principal Component 2'], label="Virginica")



plt.title('Scatter Plot for Iris Versicolour')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
