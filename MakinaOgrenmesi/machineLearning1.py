import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.scatter([1,2,3,4,5],[1,8,27,64,125])
plt.plot([1,2,3,4,5],[1,8,27,64,125])
plt.xlabel("sayilar")
plt.ylabel("sayilarin kuvveti")
# plt.show()

data = pd.read_csv("MakinaOgrenmesi/maas.csv")
# print(data.head())
# print(data.tail())

x  = data.iloc[:,0].values.reshape(-1,1) #numpy dizisi oldu values ile
y = data.iloc[:,1]
# z = data.iloc[:,[0,1]]
# print(x)
# print(y)
# print(z)

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=23)
# print(xtrain)

xnorm = (x-np.min(x))/(np.max(x)-np.min(x))
# print(xnorm)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)
print(xtrain1)