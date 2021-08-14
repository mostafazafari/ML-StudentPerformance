# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 18:47:01 2021

@author: M_Zafari
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#tf 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax, Adam, SGD
from tensorflow.keras.losses import Hinge, MeanSquaredError
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix , classification_report,f1_score, accuracy_score, plot_confusion_matrix
    
      
def Encoder(d):
    q = [] 
    columnsToEncode = list(d.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for feature in columnsToEncode:
        try:
            d[feature] = le.fit_transform(d[feature])
        except:
            print('Error encoding '+feature)
            q.append(feature)
    return d


#---------------------------------------------------------

# import dataset
d = pd.read_excel('final.xlsx')
d = Encoder(d) #category feature and object feature converted to int
d = shuffle(d)

#determine target and feature
x = d.drop(['B5'], axis=1)
y = d['B5']

#normalization data between 1 and 2
SD = MinMaxScaler(feature_range=(1, 2))
x = SD.fit_transform(x)

# Split the data for training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=42, stratify=y, test_size=0.2)


ytrain = tf.keras.utils.to_categorical(ytrain, 4)
ytest = tf.keras.utils.to_categorical(ytest, 4)

#-----------------------------------------------


#Build the model

model = Sequential()
# model.add(Dense(105,input_dim=xtrain.shape[1],activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(280,activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(140, activation='relu'))
# model.add(Dense(4, activation='softmax'))

model.add(Dense(35, input_dim=xtrain.shape[1],activation = 'linear'))
model.add(Dense(105,activation = 'linear'))
#model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(70,activation = 'linear'))
#model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

#opt = Adamax(learning_rate=0.0005, decay=0.001, clipvalue=0.5, epsilon=1e-07, name="Adamax")
#opt = SGD( learning_rate=0.001, momentum=0.0, nesterov=False, name="SGD")
opt = Adam(lr=0.0005, clipnorm=1)

#loss = Hinge(reduction="auto", name="hinge")
loss = MeanSquaredError(reduction="auto", name="mean_squared_error")

model.compile(optimizer= opt,
             loss= loss,
             metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
ANN = model.fit(xtrain,ytrain, epochs=200, batch_size=47, validation_split=.1)

#evaluate
results = model.evaluate(xtest, ytest)
results_train = model.evaluate(xtrain, ytrain)

print('\nFinal train set loss: {:4f}'.format(results_train[0]))
print('Final train set accuracy: {:4f}'.format(results_train[1]))
print('\n-------\n')
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

#------------------------------------------------------------


plt.plot(ANN.history['acc'])
plt.plot(ANN.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

plt.plot(ANN.history['loss'])
plt.plot(ANN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.figure(figsize=(6, 5))
plt.plot(ANN.history['acc'], color='red', )
plt.plot(ANN.history['val_acc'], color='green')
plt.plot(ANN.history['loss'], color='blue')
plt.plot(ANN.history['val_loss'], color='magenta')
plt.title('ANN accuracy and loss')
#plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train accuracy', 'validation accuracy', 'train loss', 'validation loss'])
plt.show()

#model.save('ANN.h5')

ypred = model.predict_classes(xtest)

 confMatx = confusion_matrix(ytest.argmax(axis=1), ypred)



# print('\n\n\n\n')

# tr = []
# ts = []

# for i in range(10):
#     x, y = asd()
#     tr.append(x[1])
#     print('\n\n', tr)
#     print(np.mean(tr))
#     ts.append(y[1])
#     print(ts)
#     print(np.mean(ts))
