import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("canada_per_capita_income.csv")



reg = linear_model.LinearRegression()
reg.fit(df[["year"]] , df['per_capital_income'])

plt.scatter(df.year , df.per_capital_income , color  = "red" , marker = "+")
plt.plot(df.year , reg.predict(df[["year"]]) , color = "blue")
plt.show()

print(reg.predict([[2020]]))



