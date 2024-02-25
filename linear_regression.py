import inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

reg = linear_model.LinearRegression()

reg.fit(df[['area']],df.price)

plt.scatter(df.area,df.price , color='red', marker='+')
plt.xlabel('area')
plt.ylabel('price')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()


print(reg.predict([[3300]]))

d = pd.read_csv("areas.csv")

p = reg.predict(d)

d['prices'] = p

d.to_csv("prediction.csv",index=False)






