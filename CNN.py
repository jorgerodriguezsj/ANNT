# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:09:19 2019

@author: Jorge
"""


from keras.callbacks import  EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout,Flatten
from keras.optimizers import SGD, RMSprop,Adam,Adagrad
from keras.utils import np_utils
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D,Conv3D, MaxPooling3D
import numpy as np


def cnn(ventana_convolucion,ventana_display,X_train,Y_train,X_test,Y_test,
                             tamlot,opt,pv,numero_capas_convolucionales,numero_capas_fullcon,
                             fl,mp,nui,con,delt,pat, NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC,NO,AC,BA,DR):
        
                             
                               

    X_train = X_train[:,:,:,np.newaxis]
    X_test = X_test[:,:,:,np.newaxis]                 
    clases = int(max(Y_train)+1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    print(X_train.shape[0],'Muestras de entrenamiento')
    print(X_test.shape[0],'Muestras de test')
    print("")
    Y_train = np_utils.to_categorical(Y_train, clases)
    Y_test = np_utils.to_categorical(Y_test, clases)

    if (len(X_train.shape)==3):
        conv=Conv1D
        pooling=MaxPooling1D
    if (len(X_train.shape)==4):
        conv=Conv2D
        pooling=MaxPooling2D
    if (len(X_train.shape)==5):
        conv=Conv3D
        pooling=MaxPooling3D
        
#-----------------Preparamos el modelo-----------------------------------
   
    model = Sequential()
    for i in range(numero_capas_convolucionales):
        model.add(conv(filters=NF[i].get(),kernel_size=TF[i].get(),
                       strides=PAS[i].get(),padding=RE[i].get(),
                       data_format="channels_last",activation=ACON[i].get(),
                       input_shape = X_train.shape[1:]))
        
        if (PO[i].get()==0):
            model.add(pooling(pool_size=TAMPOL[i].get(),strides=PASPOL[i].get()))
        
        if (DRC[i].get()!= 0.0):
            model.add(Dropout(DRC[i].get()))

    
    model.add(Flatten())
    
    #capas ocultas
    for i in range(numero_capas_fullcon):
        model.add(Dense(NO[i].get(),activation = AC[i].get()))
        if BA[i].get() == 0 :
            model.add(BatchNormalization())
        if (DR[i].get()!=0):
            model.add(Dropout(DR[i].get()))
    
    #capa salida
    model.add(Dense(clases,  activation = 'softmax'))
    
    

    model.summary()
    

    #compilar modelo
    
    model.compile(loss = fl, optimizer = opt, metrics = ['accuracy'])
    if mp == 1:
        nui=10000
        callbacks = [EarlyStopping(monitor=con, min_delta= delt,patience=pat, verbose=2),]
        entrenamiento = model.fit(X_train, Y_train, batch_size=tamlot, epochs = nui,verbose = 2,
                              validation_split=pv,callbacks = callbacks)
    else:
        entrenamiento = model.fit(X_train, Y_train, batch_size=tamlot, epochs = nui,verbose = 2,
                          validation_split=pv)  
        
       
    score = model.evaluate(X_test, Y_test, verbose = 2)
    from sklearn.metrics import classification_report
    Y_test = np.argmax(Y_test, axis=1)
    Y_pred = model.predict_classes(X_test)
    
    print("")
    print("Test score:", score[0])
    print("Precisión del test:", score[1])
    print("")
    print(classification_report(Y_test, Y_pred))
   
    return model, entrenamiento
    
        
        
    
    