import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv("MakinaOgrenmesi/musteri.csv")
print(data.head())
x = data.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch

den = sch.dendrogram(sch.linkage(x,method="ward"))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5)

yhead = ac.fit_predict(x)

plt.scatter(x[yhead==0,0],x[yhead==0,1],color = "red")
plt.scatter(x[yhead==1,0],x[yhead==1,1],color = "green")
plt.scatter(x[yhead==2,0],x[yhead==2,1],color = "blue")
plt.scatter(x[yhead==3,0],x[yhead==3,1],color = "yellow")
plt.scatter(x[yhead==4,0],x[yhead==4,1],color = "cyan")
plt.show()