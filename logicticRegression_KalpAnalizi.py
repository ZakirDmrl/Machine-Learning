import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("heart.csv")
# print(data.head())
# print(data.describe())
# print(data.info())
# print(data.isnull().sum())
# print(data["sex"].value_counts())
for i in list(data.columns):
    print("{} -- {}".format(i,data[i].value_counts().shape[0]))

karegorik = ["sex","cp","fbs","restecg","exng","slp","caa","thall","output"]
# print(karegorik)
data_kat = data.loc[:,karegorik]
print(data_kat)
for i in data_kat:
    plt.figure()
    sns.countplot(x=i,data = data_kat,hue="output")
    plt.title(i)
    # plt.show()
    
sayisal = ["age","trtbps","chol","thalachh","oldpeak","output"]
data_say = data.loc[:,sayisal]
sns.pairplot(data_say,hue="output",diag_kind="kde")
# plt.show()

scaler = StandardScaler()
scled_array = scaler.fit_transform(data[sayisal[:-1]])
print(scled_array)
data1 = data.copy()
data1 = pd.get_dummies(data1,columns = karegorik[:-1],drop_first = True)
x  =data1.drop(["output"],axis=1)
y = data1["output"]
x[sayisal[:-1]]=scaler.fit_transform(x[sayisal[:-1]])
print(x)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.15,random_state=5)
logreg = LogisticRegression()
logreg.fit(xtrain,ytrain)
ypred_prob = logreg.predict_proba(xtest)
print(ypred_prob)
ypred =np.argmax(ypred_prob,axis=1)
print(ypred)
dummy = pd.DataFrame(ypred_prob)
dummy["ypred"]= ypred
print(dummy)
print("Test Accuracy",accuracy_score(ypred,ytest))