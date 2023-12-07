import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno  as msno
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,confusion_matrix
from sklearn import tree
data = pd.read_csv("water_potability.csv")
print(data.head())
print(data.describe())
print(data.info())
yeni =pd.DataFrame(data["Potability"].value_counts())
print(yeni)
fig = px.pie(yeni,values="Potability",names=["içilemez","içilebilir"],hole=0.4,opacity=0.8,labels={"label":"Potability","Potability":"numune sayısı"})
fig.update_layout(title=dict(text="içilebilirlik"))
# fig.show()
data.corr()
sns.clustermap(data.corr(),cmap="vlag",dendrogram_ratio=(0.1,0.2),annot = True,linewidth = 0.8,figsize=(9,10))
plt.plot()
# plt.show()
non_potable = data.query("Potability==0")
potable = data.query("Potability==1")
plt.figure(figsize=(15,15))
for ax,col in enumerate(data.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="içilmez")
    sns.kdeplot(x=potable[col],label="içilebilir")
    plt.legend()

plt.tight_layout()
# plt.show()
msno.matrix(data,color=(0,0,1))

print(data.isnull().sum())
data["ph"].fillna(value=data["ph"].mean(),inplace=True)
data["Sulfate"].fillna(value=data["Sulfate"].mean(),inplace=True)
data["Trihalomethanes"].fillna(value=data["Trihalomethanes"].mean(),inplace=True)
# plt.show()
x = data.drop("Potability",axis=1).values
y = data["Potability"].values
print(x)
print(y)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=33)
xtrainmax = np.max(xtrain)
xtrainmin = np.min(xtrain)
xtrain = (xtrain - xtrainmin) / (xtrainmax - xtrainmin)
xtest = (xtest - xtrainmin) / (xtrainmax - xtrainmin)  # Use xtrainmin and xtrainmax to normalize xtest

print(xtrain)
models = [("DTC",DecisionTreeClassifier(max_depth=3)),("RFC",RandomForestClassifier())]
sonuc = []
cmlist = []
for name,model in models:
    model.fit(xtrain,ytrain)
    model_result = model.predict(xtest)
    score = precision_score(ytest,model_result)
    cm = confusion_matrix(ytest,model_result)
    sonuc.append((name,score))
    cmlist.append(cm)

print(sonuc)

for name, i in zip(models, cmlist):
    plt.figure()
    sns.heatmap(i, annot=True, linewidths=0.8, fmt=".1f")
    plt.title(name[0] + " CM")  # Access the model name from the tuple
    plt.show()

yeni1 = models[0][1]
data.columns.tolist()[:-1]
plt.figure(figsize=(15,10))
tree.plot_tree(yeni1,feature_names=data.columns.tolist()[:-1],class_names=["0","1"],filled=True)
plt.show()