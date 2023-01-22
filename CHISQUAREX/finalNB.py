from sklearn.naive_bayes import GaussianNB
import pandas as pd
import math


df = pd.read_csv("train.csv")
print(df.isnull().sum())
median_type = math.floor(df.type.median())
median_params = math.floor(df.params.median())
df.type = df.type.fillna(median_type)
df.params= df.params.fillna(median_params)

print(df.isnull().sum())
x_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

model = GaussianNB()

model.fit(x_train,y_train)

print(model.score(x_train,y_train)*100)

