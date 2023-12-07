import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/ploy.csv")
x = data["zaman"].values.reshape(-1,1)
y = data["sıcaklık"].values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
plt.scatter(x,y)
plt.plot(x,lr.predict(x),color="red")
plt.show() # Doğru bir sonuç çıkmadı

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree=6)
xpl = pr.fit_transform(x)


lr2 = LinearRegression()

lr2.fit(xpl, y)
yhead = lr2.predict(xpl)

plt.scatter(x, y)
plt.plot(x,lr.predict(x),color="red", label="Polynomial Regression")
plt.plot(x, lr2.predict(xpl), color="green", label="Linear Regression")
plt.legend()
plt.show()

