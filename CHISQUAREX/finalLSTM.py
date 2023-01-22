import pandas as pd
import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

df = pd.read_csv("train.csv")
print(df.isnull().sum())
median_type = math.floor(df.type.median())
median_params = math.floor(df.params.median())
df.type = df.type.fillna(median_type)
df.params= df.params.fillna(median_params)

print(df.isnull().sum())
x = df.iloc[:,:-1]
print(x)
x = x.drop(['params'],axis=1)
y = df.iloc[:,-1]

X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
train = numpy.array(list(zip(X_train,y_train)))
test = numpy.array(list(zip(X_test,y_test)))

def create_dataset(n_X, look_back):
   dataX, dataY = [], []
   for i in range(len(n_X)-look_back):
      a = n_X[i:(i+look_back), ]
      dataX.append(a)
      dataY.append(n_X[i + look_back, ])
   return numpy.array(dataX), numpy.array(dataY)

look_back = 1
trainx,trainy = create_dataset(train, look_back)
testx,testy = create_dataset(test, look_back)

#X_train = numpy.reshape(X_train, (X_train.shape[0], 1, 2))
#X_test = numpy.reshape(X_test, (X_test.shape[0], 1, 2))

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(256, return_sequences = True, input_shape = (X_train.shape[1], 2)))
model.add(LSTM(128,input_shape = (X_train.shape[1], 2)))
model.add(Dense(2))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train, y_train, epochs = 200, batch_size = 10, verbose = 2, shuffle = False)
model.save_weights('LSTMBasic1.h5')

model.load_weights('LSTMBasic1.h5')
predict = model.predict(X_test)

y_pred  =model.predict(X_test)
print(model.score(X_train,y_train))
print(classification_report(y_pred,y_test))