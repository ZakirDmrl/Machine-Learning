import numpy as np
import pandas as pd
import tensorflow as tf
data = pd.read_csv("MakinaOgrenmesi/Churn_Modelling.csv")
print(data.head())

x = data.iloc[:,3:-1].values
y = data.iloc[:,-1].values
print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[1])],remainder="passthrough")
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=22)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6,activation="relu")) # Gizli 1.Katman # Relu Metotunu kullandım 0 ve 1 arasında da değer için
ann.add(tf.keras.layers.Dense(units=6,activation="relu")) # Gizli 2.Katman
ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid")) # Çıkış Katmanı
ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"]) # Burada Adam modeli kullanım ayrıca binary(doğruluğunu 0  1) çıktı istiyorum
ann.fit(xtrain,ytrain,epochs=100) # Kendi kendine öğreneceği için 100 gibi yüksek bir sayı verdim
#Epoch 100/100
# 250/250 [==============================] - 0s 870us/step - loss: 0.3873 - accuracy: 0.8420

ypred = ann.predict(xtest) #değerleri predict ettim
ypred = (ypred>0.5) # 0.5 den büyük olanları aldım

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest,ypred)
print(cm)
accuracy_score(ytest,ypred)