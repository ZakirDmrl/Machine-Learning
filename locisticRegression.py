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
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=44)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain1 =sc.fit_transform(xtrain)
xtest1 = sc.transform(xtest)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtrain1,ytrain)
yhead = lr.predict(xtest1) 
lr.score(xtest1,ytest)
print(lr.score(xtest1,ytest))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,yhead)
print(cm)
# plt.show()