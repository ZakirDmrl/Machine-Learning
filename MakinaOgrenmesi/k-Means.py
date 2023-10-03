import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/musteri.csv")
# print(data.head())

x = data.iloc[:,[3,4]].values
from sklearn.cluster import KMeans

wcss = []
for i in range(1,14):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,14),wcss)
plt.show() # Grafiğe göre n_clusters değerini 5 almak daha verimli

kmeans = KMeans(n_clusters=5)
yhead = kmeans.fit_predict(x)
plt.scatter(x[yhead==0,0],x[yhead==0,1],color="red")
plt.scatter(x[yhead==1,0],x[yhead==1,1],color="blue")
plt.scatter(x[yhead==2,0],x[yhead==2,1],color="green")
plt.scatter(x[yhead==3,0],x[yhead==3,1],color="yellow")
plt.scatter(x[yhead==4,0],x[yhead==4,1],color="black")
plt.show()
