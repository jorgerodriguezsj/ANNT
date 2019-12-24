# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:50:06 2019

@author: Jorge
"""

from keras.callbacks import  EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers import InputLayer, BatchNormalization
import numpy as np
from sklearn.metrics import classification_report

def Perceptron_multicapa(ventana_perceptron,ventana_display,X_train,Y_train,X_test,Y_test,
                         tamlot,opt,pv,nco,fl,mp,nui,con,delt,pat,NO,AC,BA,DR):
  
                      
    clases = int(max(Y_train)+1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    print(X_train.shape[0],'Muestras de entrenamiento')
    print(X_test.shape[0],'Muestras de test')
    print("")
    Y_train = np_utils.to_categorical(Y_train, clases)
    Y_test = np_utils.to_categorical(Y_test, clases)

#-----------------Preparamos el modelo-----------------------------------
    model = Sequential()
    
    #Capa entrada
    model.add(InputLayer(input_shape=(X_train.shape[1],)))
    
    #capas ocultas
    for i in range(nco):
        model.add(Dense(NO[i].get(),activation = AC[i].get()))
        if BA[i].get() == 0 :
            model.add(BatchNormalization())
        if (DR[i].get()!=0):
            model.add(Dropout(DR[i].get()))
    
    #capa salida
    model.add(Dense(Y_train.shape[1],  activation = 'softmax'))
    
    

    model.summary()
    

    #compilar modelo
    model.compile(loss = fl, optimizer = opt, metrics = ['accuracy'])
    
    #entrenamiento
    if mp == 1:
        nui=10000
        callbacks = [EarlyStopping(monitor=con, min_delta= delt,patience=pat, verbose=2),]
        entrenamiento = model.fit(X_train, Y_train, batch_size=tamlot, epochs = nui,verbose = 2,
                              validation_split=pv,callbacks = callbacks)
    else:
        entrenamiento = model.fit(X_train, Y_train, batch_size=tamlot, epochs = nui,verbose = 2,
                          validation_split=pv)  
        
    #evaluación  
    score = model.evaluate(X_test, Y_test, verbose = 2)
    
    Y_test = np.argmax(Y_test, axis=1)
    Y_pred = model.predict_classes(X_test)
    
    
    print("")
    print("Test score:", score[0])
    print("Precisión del test:", score[1])
    print("")
    #Evaluación por clases
    print(classification_report(Y_test, Y_pred))
    return model, entrenamiento
    
        
        
    
    