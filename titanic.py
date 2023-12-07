import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = pd.read_csv("MakinaOgrenmesi/test.csv")
train_data = pd.read_csv("MakinaOgrenmesi/train.csv")

print(train_data.describe())
print(train_data.columns)
print(train_data.info())

def degiskenler(variable):
    cat = train_data[variable]
    sayi = cat.value_counts()
    plt.figure(figsize=(9.3, 6.0))  
    plt.bar(sayi.index, sayi)
    plt.xticks(sayi.index, sayi.index.values)
    plt.ylabel("frekeans")
    plt.title(variable)
    plt.show()
    print(variable, sayi)

# cat1 = ["Survived", "Sex", "Pclass", "SibSP", "Parch"]

# for i in cat1:
#     degiskenler(i)

def hist(variable):
    plt.figure(figsize=(9.3, 6.0))  
    plt.hist(train_data[variable])
    plt.xlabel(variable)
    plt.ylabel("frekeans")
    plt.title(variable)
    plt.show()


# sayisal = ["Fare","Age","PassengerId"]

# for i in sayisal:
#     hist(i)


print(train_data[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False))
print(train_data[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False))


train_data["Embarked"]=train_data["Embarked"].fillna("C")
print(train_data[train_data["Embarked"].isnull()])
train_data.boxplot(column="Fare",by="Embarked")


plt.show()
