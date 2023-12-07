import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("MakinaOgrenmesi/bilet.csv")

print(data)

x = data["sıra"].values.reshape(-1,1)
y = data["fiyat"].values.reshape(-1,1)



from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(x,y)
x1 = np.arange(min(x),max(x),0.01).reshape(-1,1)
yhead = dt.predict(x1)
plt.scatter(x,y)
plt.plot(x1,yhead)
plt.show()