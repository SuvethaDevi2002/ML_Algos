import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")
print(df.isnull().sum())
median_type = math.floor(df.type.median())
median_params = math.floor(df.params.median())
df.type = df.type.fillna(median_type)
df.params= df.params.fillna(median_params)

print(df.isnull().sum())
x = df.iloc[:,:-1]
print(x)
#x = x.drop(['params'],axis=1)
y = df.iloc[:,-1]

X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

model = LinearRegression()

model.fit(X_train,y_train)
y_pred  =model.predict(X_test)
print(model.score(X_train,y_train))
print(classification_report(y_pred,y_test))

