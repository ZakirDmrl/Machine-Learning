import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/urun.csv")
x = data.iloc[:,0:2].values
y = data["satinalma"].values.reshape(-1,1)
S = data[data["satinalma"] == 0]
B = data[data["satinalma"] == 1]
plt.scatter(S["yaş"],S["maaş"],color="red")
plt.scatter(B["yaş"],B["maaş"],color="blue")
# plt.show()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest  = train_test_split(x,y,test_size=0.3,random_state=33)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain1 = sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(xtrain1,ytrain)
yhead = knn.predict(xtest1)
# print(yhead)
print(knn.score(xtest1,ytest))
scorelist = []
for i in range(1,30):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(xtrain1,ytrain)
    scorelist.append(knn2.score(xtest1,ytest))

plt.plot(range(1,30),scorelist)
plt.xlabel("komşu sayısı")
plt.ylabel("doğruluk değeri")
# plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,yhead)
print(cm)