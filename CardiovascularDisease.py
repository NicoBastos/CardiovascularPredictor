import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import array, argmax
import keras
import tensorflow
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt

# names = ["age ", "gender ", "height ", "weight ", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio", ]
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("newData2.txt", names=["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"], low_memory=False)

X = data[["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]]
Y = data["cardio"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = scaler.inverse_transform(X)
# print(X["age"])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(44, activation='relu', input_dim=11))
model.add(Dense(22))
# model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='sigmoid'))

# model.fit(X_train,y_train, batch_size=10, epochs= 100)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


history = model.fit(X_train,y_train, batch_size=20, epochs= 100)
print('\nhistory dict:', history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.show()



