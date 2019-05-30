#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:29:00 2019

@author: walter
"""
import cv2
import numpy as np
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

random_seed=21

def take_imgs(paths, clahe=False, threshold=False):
    images = []
	
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	
    for index, path in enumerate(paths):
        img = cv2.imread((path), cv2.IMREAD_GRAYSCALE)
		
        if clahe:
            img = clahe.apply(img)
        if threshold:
            _, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY);
			
        if img is not None:
            img = img.reshape((400,400,1))
            #img = img/255
            images.append(img)
            
    print(" It's OK!!")
    return np.array(images)

names, y = utils.takeNamesAndFloats('/home/walter/Documents/Projeto_Asas/dots_augmentation.txt','/home/walter/Documents/Projeto_Asas/train')

y = np.reshape(y,(-1,38))

X =   take_imgs(names)

#normalize
scaler = MinMaxScaler()
scaler.fit(y)
y = scaler.transform(y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=random_seed)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD 
from keras.utils import multi_gpu_model

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', activation='tanh', input_shape=(400, 400, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(38, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model = multi_gpu_model(model, gpus=2)

model.compile(loss='mean_squared_error', optimizer=sgd)

modelName = 'CNN.h5'
earlystopper = EarlyStopping(patience=3, verbose=1)
checkpointer = ModelCheckpoint(modelName, verbose=1, save_best_only=True)
plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                            verbose=1, mode='auto', min_delta=0.0001, 
                                            cooldown=0, min_lr=0.000001)

model.fit(np.array(Xtrain), Ytrain, batch_size=64, epochs=20, validation_split=0.2, verbose = 1, callbacks=[earlystopper, checkpointer, plateau])

Ytrain_pred = model.predict(Xtrain)
Ytest_pred = model.predict(Xtest)

























































