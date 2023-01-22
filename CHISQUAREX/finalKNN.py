import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import DistanceMetric
import math
df = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
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
X_test = sc.transform(X_test)

error = []

for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    #pred = model.predict(X_test)
   # error.append(np.mean(pred!=y_test))

plt.figure(figsize = (15,15))
plt.plot(range(1,40),error,color ='blue' ,markersize = 10)
plt.title("Error rate in K value")
plt.xlabel("K values")
plt.ylabel("Mean error")
plt.show()

model = KNeighborsClassifier(n_neighbors=17, metric= 'minkowski',p=2)
model.fit(X_train,y_train)


print(model.score(x_train,y_train)*100)
'''''
a = [[19.69,21.25,130,1203,0.1096,0.1599,0.1974,0.1279,0.2069,0.05999,0.7456,0.7869,4.585,94.03,0.00615,0.04006,0.03832,0.02058,0.0225,0.004571,23.57,25.53,152.5,1709,0.1444,0.4245,0.4504,0.243,0.3613,0.08758]]
res = model.predict(sc.transform(a))

if res ==0:
    print("Yes its 'M'")
else:
    print("Yes its 'N")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

print("Confusion matrix:", cm)
print(accuracy_score(y_test,y_pred))

sn.heatmap(cm,annot=True,fmt='d')
plt.show()
'''