import numpy as np
import pandas as pd
data = pd.read_csv("MakinaOgrenmesi/tweet_data.csv",encoding="latin1")
print(data.head())
# %%
data1 = pd.concat([data["gender"],data["description"]],axis=1) #gender ve description bilgilerini data1 e ekle
print(data1.head())
# %%
data1.dropna(inplace=True) # NA olanları sil
print(data1.head())
data1.reset_index(drop=True,inplace=True) # yeniden indexle
# %%
data1["gender"] = [1 if i =="female" else 0 for i in data1["gender"]] # female => 1, male => 0 yap
print(data1.head())
# %%
import re
metin = re.sub("[^a-zA-Z]"," ",data1["description"][9]) # a-z ve A-Z dışındakileri boşluğa çevir
print(metin)
harfler = metin.lower() # küçük harfe çevir
bol = harfler.split() # boşlulara göre ayır
print(bol)
#%%
from nltk.stem.porter import PorterStemmer # Köklere ayırmak için
ps = PorterStemmer()
#%%
import nltk
stop = nltk.download("stopwords")
#%%
from nltk.corpus import stopwords
#%%
metin = [ps.stem(i) for i in bol if not i in set(stopwords.words("english"))] # stopword olanları ayır
metinson = " ".join(metin) # boşluğa göre birleştir
print(metinson)
#%%
liste = []
for j in range(1000):
    metin = re.sub("[^a-zA-Z]"," ",data1["description"][j])
    metin = metin.lower()
    metin = metin.split()
    metin =[ps.stem(i) for i in metin if not i in set(stopwords.words("english"))]
    metinson = " ".join(metin)
    liste.append(metinson)

# %%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=10000)
x = cv.fit_transform(liste).toarray()
y = data1["gender"].values
print(x)
print(y)
# %%
from sklearn.model_selection import train_test_split # makinayı eğit
x = x[:1000] 
y = y[:1000]  
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=33)

# %%
from sklearn.naive_bayes import GaussianNB
gn = GaussianNB()
gn.fit(xtrain,ytrain)
yhead = gn.predict(xtest)
# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,yhead)
print(cm)
print(gn.score(yhead.reshape(-1,1),ytest))
