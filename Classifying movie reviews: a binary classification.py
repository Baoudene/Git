# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 21:11:33 2018

@author: grerger
"""

from keras.datasets import imdb
(xtrain,ytrain),(xtest,ytest)=imdb.load_data(num_words=10000)

print(len(xtrain[1]),ytrain[0])

word_index = imdb.get_word_index()
#print(word_index)
rvrs_word_index=dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join(
[rvrs_word_index.get(i, '?') for i in xtrain[0]])
print(decoded_review)
      
      
      
import numpy as np

def vrtr(sqncs,dim=10000):
    result=np.zeros((len(sqncs),dim))
    for i,seq in enumerate(sqncs):
        result[i,seq]=1
    return result

xtrain=vrtr(xtrain)
xtest=vrtr(xtest)
ytrain=np.asarray(ytrain,dtype='float32')
ytest=np.asarray(ytest,dtype='float32')
          

from keras import layers
from keras import models
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())
x_val=xtrain[:10000]
xtrain_part=xtrain[10000:]
y_val=ytrain[:10000]
ytrain_part=ytrain[10000:]

history=model.fit(xtrain_part,ytrain_part,batch_size=512,epochs=20,validation_data=(x_val,y_val))
history_dict=history.history

los_val=history_dict['loss']
val_los_val=history_dict['val_loss']
accc=history_dict['acc']
val_accc=history_dict['val_acc']

import matplotlib.pyplot as plt
epoche=range(1,len(los_val)+1)
plt.subplot(2,1,1)
plt.plot(epoche,los_val,'bo',label='train_loss')
plt.plot(epoche,val_los_val,'b',label='validat_loss')
plt.title('val and train loss')
plt.xlabel('epoche')
plt.ylabel('loss')
plt.legend()
plt.show



plt.subplot(2,1,2)
plt.plot(epoche,accc,'bo',label='train_acc')
plt.plot(epoche,val_accc,'b',label='valid_acc')
plt.title('val and train acc')
plt.xlabel('epoche')
plt.ylabel('acc')
plt.legend()
plt.show

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,batch_size=512,epochs=3,)


print(model.evaluate(xtest,ytest))
