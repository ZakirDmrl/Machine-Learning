import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/reklam.csv")
# print(data.columns)

x = data.iloc[:, 1:4].values  # Tüm satırları ve 1. ile 3. sütunlar arasındaki verileri seçiyoruz.
y = data["satış"].values.reshape(-1, 1)  # "satış" sütununu seçip yeniden şekillendiriyoruz.
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=22)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

yhead = lr.predict(xtest)

print(lr.predict([[230,38,70]]))