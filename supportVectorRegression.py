import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/ploy.csv")

# print(data.head())

x = data["zaman"].values.reshape(-1,1)
y = data["sıcaklık"].values.reshape(-1,1)

# print(x)
# print(y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x1 = sc.fit_transform(x)
# print(x1)
y1 = sc.fit_transform(y)
# print(y1)

from sklearn.svm import SVR

sv = SVR(kernel="sigmoid") # default değeri kernel = 'rbf,degree = 3 ,Bunun dışında linear,poly,sigmoid,precomputed
sv.fit(x1,y1)

plt.scatter(x1,y1)
plt.plot(x1,sv.predict(x1),color="red")
plt.show()