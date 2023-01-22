import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR

df = pd.read_csv("train.csv")
print(df.isnull().sum())
median_type = math.floor(df.type.median())
median_params = math.floor(df.params.median())
df.type = df.type.fillna(median_type)
df.params= df.params.fillna(median_params)

print(df.isnull().sum())
x_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]
sc = StandardScaler()
X_train = sc.fit_transform(x_train)


model = SVC()
model.fit(X_train,y_train)


print(model.score(x_train,y_train)*100)

