import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Basit Doğrusal Regresyon
data = pd.read_csv("MakinaOgrenmesi/maas.csv")

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("tecrübe")
plt.ylabel("maas")
# plt.show()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=33)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(xtrain,ytrain)

yhead = lr.predict(xtest)

print(lr.predict([[1.8]])) #Tahmin

plt.scatter(x,y)
plt.plot(x,lr.predict(x),color="red")
plt.show()

from sklearn.metrics import r2_score
r2_score(ytest,yhead)
print(r2_score)