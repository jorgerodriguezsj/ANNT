# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:48:31 2019

@author: jrodriguez119

Extreme Learning Machine

Función que realiza este tipo de red


"""

import numpy as np
import cupy as cp
from time import time
from keras.utils import to_categorical

#ELM básica que devuelve el modelo
def ELM(X_train,X_test,y_train,y_test,neuronas_ocultas,proporcion_datos,xp):
    
    #obtiene de los datos automáticamente el numero de clases 
    clases = int(max(y_train)+1)
    
    #Opción de no emplear el 100% de los datos disponibles para agilizar el cálculo
    n_train = int(X_train.shape[0] *proporcion_datos)
    n_test = int(X_test.shape[0] *proporcion_datos)
    ind_train = np.random.choice(X_train.shape[0], size=n_train, replace = False)
    ind_test = np.random.choice(X_test.shape[0], size=n_test, replace = False)
    
    #Definimos los conjuntos de datos que emplearemos en el entrenamiento
    global Y_test1,Y_pred_test
    X_train1 = X_train[ind_train, :]
    y_train1 = y_train[ind_train].astype('int')
    X_test1 = X_test[ind_test, :]
    y_test1 = y_test[ind_test].astype('int')
    
    #Formato de entrada es recogido automaticamente según el número de columnas de X_train1
    neuronas_entrada = X_train1.shape[1] 

    #---------------------GPU---------------------------------------------    
    if (xp == 0):
        
        #Generamos aleatoriamente los pesos de entrada y las bias
        cp.cuda.runtime.deviceSynchronize()
        tiempo_inicial = time()
        
        #Pesos de entrada y Bias
        Win = cp.asarray((np.random.random([neuronas_entrada, neuronas_ocultas])*2.-1.).astype('float32'))
        Bias = cp.asarray(np.random.random([1,neuronas_ocultas])*2-1)
        
        #Conversiones a formato requerido por CUDA (CUPY)
        X_train1 = cp.asarray(X_train1)
        
        #Calculamos una H previa
        temp_H = cp.dot(X_train1,Win)
        
        #Hay que extender la matriz de Bias para que coincida con la dimension de H
        BiasMatrix = cp.repeat(Bias,temp_H.shape[0],axis = 0)
        
        #Añadimos los Bias a la H previa
        temp_H = temp_H + BiasMatrix
        
        #Función ReLU.
        H = cp.maximum(temp_H,0,temp_H)
        
        #Calculamos los pesos de salida haciendo uso de la pseudoinversa de Moore Penrose
        #Comprobamos que el determinante sea distinto de cero
        if (cp.linalg.det(cp.dot(cp.transpose(H),H)))==0 :
            Y_train1 = to_categorical(y_train1,clases)
            H = cp.asnumpy(H)
            Y_train1 = cp.asnumpy(Y_train1)
            Wout = np.dot(np.linalg.pinv(H), Y_train1)
            del Y_train1
            tiempo_final = time()
            del H
            tiempo = tiempo_final - tiempo_inicial    
            

            #Conjunto de test
            X_test1 = cp.asnumpy(X_test1)
            Win = cp.asnumpy(Win)
            temp_H_test = np.dot(X_test1,Win)
            del X_test1
            
            #Extendemos la matriz de los bias para que cuadre
            BiasMatrix = np.repeat(Bias,temp_H_test.shape[0],axis = 0)
            BiasMatrix = cp.asnumpy(BiasMatrix)
            temp_H_test = temp_H_test + BiasMatrix
            del BiasMatrix
            
            H_test = np.maximum(temp_H_test,0,temp_H_test)
            del temp_H_test
            
            #Prediccion de test
            Y_pred_test = np.dot(H_test,Wout)
            Y_pred_test = np.argmax(Y_pred_test,axis=1)
            del H_test
            
            aciertos = np.sum(Y_pred_test==y_test1)
            
            precision_test= aciertos/y_test1.size
            
        else:
            X_test1 = cp.asarray(X_test1)
            Y_train1 = cp.asarray(to_categorical(y_train1,clases))
            
            Wout = cp.dot(cp.dot(cp.linalg.inv(cp.dot(cp.transpose(H),H)),
                                 cp.transpose(H)),Y_train1)

        
            cp.cuda.runtime.deviceSynchronize()
            tiempo_final = time()
            
            tiempo = tiempo_final - tiempo_inicial    
            
            
            #Conjunto de test
            temp_H_test = cp.dot(X_test1,Win)
            
            
            #Extendemos la matriz de los bias para que cuadre
            BiasMatrix = cp.repeat(Bias,temp_H_test.shape[0],axis = 0)
            
            #Añadimos los bias
            temp_H_test = temp_H_test + BiasMatrix
            
            #Función ReLu
            H_test = cp.maximum(temp_H_test,0,temp_H_test)
            
            #Prediccion de test
            Y_pred_test = cp.dot(H_test,Wout)
            Y_pred_test = cp.asnumpy(Y_pred_test)
            Y_pred_test = np.argmax(Y_pred_test,axis=1)
            aciertos = np.sum(Y_pred_test==y_test1)
            precision_test= aciertos/y_test1.size
  
            
    #---------------------CPU---------------------------------------------  
            
    if (xp == 1):
                
        Y_train1 = (to_categorical(y_train1,clases))
        
        #Generamos aleatoriamente los pesos de entrada y las bias
        tiempo_inicial = time()
        
        #Pesos de entrada y Bias
        Win = ((np.random.random([neuronas_entrada, neuronas_ocultas])*2.-1.).astype('float32'))
        Bias = (np.random.random([1,neuronas_ocultas])*2-1).astype('float32')
        temp_H = np.dot(X_train1,Win).astype('float32')
        del X_train1
        
        #Hay que extender la matriz de Bias para que coincida con la dimension de H
        BiasMatrix = np.repeat(Bias,temp_H.shape[0],axis = 0)
        
        #Añadimos los Bias
        temp_H = temp_H + BiasMatrix
        
        #Función ReLU.
        H = np.maximum(temp_H,0,temp_H)
        del temp_H
        
        #Calculamos los pesos de salida haciendo uso de la pseudoinversa de Moore Penrose
        #Comprobamos que el determinante sea distinto de cero
        
        if (np.linalg.det(np.dot(np.transpose(H),H)))==0 :
            Wout = np.dot(np.linalg.pinv(H), Y_train1)
            
        else:
            Wout = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H),H)),
                                 np.transpose(H)),Y_train1)
            
        del Y_train1
        tiempo_final = time()
        del H
        
        tiempo = tiempo_final - tiempo_inicial    
        
        #Conjunto de test
        
        temp_H_test = np.dot(X_test1,Win)
        del X_test1
        
        #Extendemos la matriz de los bias para que cuadre
        BiasMatrix = np.repeat(Bias,temp_H_test.shape[0],axis = 0)
        temp_H_test = temp_H_test + BiasMatrix
        del BiasMatrix
        
        H_test = np.maximum(temp_H_test,0,temp_H_test)
        del temp_H_test
        
        #Prediccion de test
        Y_pred_test = np.dot(H_test,Wout)
        Y_pred_test = np.argmax(Y_pred_test,axis=1)
        del H_test
        
        aciertos = np.sum(Y_pred_test==y_test1)
        
        precision_test= aciertos/y_test1.size
        
    
    return neuronas_ocultas, precision_test, tiempo , Win , Wout, Bias,y_test1,Y_pred_test
    

#ELM con barrido de varias neuronas    
def ELM1(X_train,X_test,y_train,y_test,ini,fin,inter,
         xp,repeticiones):

    #Fijamos la proporción de datos al 100%    
    proporcion_datos = 1
    
    clases = int(max(y_train)+1)
    n_train = int(X_train.shape[0] *proporcion_datos)
    n_test = int(X_test.shape[0] *proporcion_datos)
    neuronas = []
    acierto_medio = []
    tiempo_medio = []
    
    #Robustez
    if ini != fin:
        numero_neuronas = np.arange(ini,fin+1,inter).astype('int')
    
    if ini == fin and inter==0:
        numero_neuronas =  []
        numero_neuronas.append(ini)
        
    for i in numero_neuronas:
        
        acierto = 0
        tiempo = 0
        neuronas.append(i)
        
        for j in range(repeticiones):
            ind_train = np.random.choice(X_train.shape[0], size=n_train, replace = False)
            ind_test = np.random.choice(X_test.shape[0], size=n_test, replace = False)
           
            X_train1 = X_train[ind_train, :]
            y_train1 = y_train[ind_train].astype('int')
            X_test1 = X_test[ind_test, :]
            y_test1 = y_test[ind_test].astype('int')
            
            neuronas_entrada = X_train1.shape[1] # 784
        
        #---------------------GPU---------------------------------------------    
            if (xp == 0):
                X_train1 = cp.asarray(X_train1)
                X_test1 = cp.asarray(X_test1)
                Y_train1 = cp.asarray(to_categorical(y_train1,clases))
                
                #Generamos aleatoriamente los pesos de entrada y las bias
                cp.cuda.runtime.deviceSynchronize()
                tiempo_inicial = time()
                #Pesos de entrada
                Win = cp.asarray((np.random.random([neuronas_entrada, i])*2.-1.).astype('float32'))
                Bias = cp.asarray(np.random.random([1,i])*2.-1.)
                temp_H = cp.dot(X_train1,Win)
                del X_train1
                #Hay que extender la matriz de Bias para que coincida con la dimension de H
                BiasMatrix = cp.repeat(Bias,temp_H.shape[0],axis = 0)
                
                #Añadimos los Bias
                temp_H = temp_H + BiasMatrix
                
                #Función ReLU.
                H = cp.maximum(temp_H,0,temp_H)
                del temp_H
                
                #Calculamos los pesos de salida haciendo uso de la pseudoinversa de Moore Penrose                         
                if (cp.linalg.det(cp.dot(cp.transpose(H),H)))==0 :
                    Y_train1 = to_categorical(y_train1,clases)
                    H = cp.asnumpy(H)
                    Y_train1 = cp.asnumpy(Y_train1)
                    
                    Wout = np.dot(np.linalg.pinv(H), Y_train1)
                    del Y_train1
                    tiempo_final = time()
                    del H
                    tiempo = tiempo + (tiempo_final - tiempo_inicial)    
                    
        
                    #Conjunto de test
                    X_test1 = cp.asnumpy(X_test1)
                    Win = cp.asnumpy(Win)
                    temp_H_test = np.dot(X_test1,Win)
                    del Win
                    del X_test1
                    
                    #Extendemos la matriz de los bias para que cuadre
                    BiasMatrix = np.repeat(Bias,temp_H_test.shape[0],axis = 0)
                    del Bias
                    BiasMatrix = cp.asnumpy(BiasMatrix)
                    temp_H_test = temp_H_test + BiasMatrix
                    del BiasMatrix
                    
                    H_test = np.maximum(temp_H_test,0,temp_H_test)
                    del temp_H_test
                    
                    #Prediccion de test
                    Y_pred_test = np.dot(H_test,Wout)
                    Y_pred_test = np.argmax(Y_pred_test,axis=1)
                    del H_test
                    del Wout
                    
                    aciertos = np.sum(Y_pred_test==y_test1)
                    del Y_pred_test
                    acierto= acierto + aciertos/y_test1.size
                
                
                else:
                    X_test1 = cp.asarray(X_test1)
                    Y_train1 = cp.asarray(to_categorical(y_train1,clases))
                    
                    Wout = cp.dot(cp.dot(cp.linalg.inv(cp.dot(cp.transpose(H),H)),
                                         cp.transpose(H)),Y_train1)
                    del Y_train1
                
                    cp.cuda.runtime.deviceSynchronize()
                    tiempo_final = time()
                    del H
                    tiempo = tiempo_final - tiempo_inicial    
                    
                    
                    #Conjunto de test
                    temp_H_test = cp.dot(X_test1,Win)
                    del Win
                    del X_test1
                    
                    #Extendemos la matriz de los bias para que cuadre
                    BiasMatrix = cp.repeat(Bias,temp_H_test.shape[0],axis = 0)
                    del Bias
                    temp_H_test = temp_H_test + BiasMatrix
                    del BiasMatrix
                    H_test = cp.maximum(temp_H_test,0,temp_H_test)
                    del temp_H_test
                    
                    #Prediccion de test
                    Y_pred_test = cp.dot(H_test,Wout)
                    Y_pred_test = cp.asnumpy(Y_pred_test)
                    Y_pred_test = np.argmax(Y_pred_test,axis=1)
                    del H_test
                    del Wout
                    aciertos = np.sum(Y_pred_test==y_test1)
                    del Y_pred_test
                    acierto= acierto + aciertos/y_test1.size
                    
             
                    
        #---------------------CPU---------------------------------------------  
                    
            if (xp == 1):
                        
                Y_train1 = (to_categorical(y_train1,clases))
                
                #Generamos aleatoriamente los pesos de entrada y las bias
                tiempo_inicial = time()
                #Pesos de entrada
                Win = ((np.random.random([neuronas_entrada, i])*2.-1.).astype('float32'))
                Bias = (np.random.random([1,i]))
                temp_H = np.dot(X_train1,Win)
                
                #Hay que extender la matriz de Bias para que coincida con la dimension de H
                BiasMatrix = np.repeat(Bias,temp_H.shape[0],axis = 0)
                
                #Añadimos los Bias
                temp_H = temp_H + BiasMatrix
                
                #Función ReLU.
                H = np.maximum(temp_H,0,temp_H)
                
                #Calculamos los pesos de salida haciendo uso de la pseudoinversa de Moore Penrose
                if (np.linalg.det(np.dot(np.transpose(H),H)))==0 :
                    Wout = np.dot(np.linalg.pinv(H), Y_train1)
            
                else:
                    Wout = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(H),H)),
                                 np.transpose(H)),Y_train1)

                del H
                tiempo_final = time()
                
                tiempo = tiempo + (tiempo_final - tiempo_inicial)    
                              
                
                #Conjunto de test
                temp_H_test = np.dot(X_test1,Win)
                
                #Extendemos la matriz de los bias para que cuadre
                BiasMatrix = np.repeat(Bias,temp_H_test.shape[0],axis = 0)
                
                temp_H_test = temp_H_test + BiasMatrix
                
                H_test = np.maximum(temp_H_test,0,temp_H_test)
                
                #Prediccion de test
                Y_pred_test = np.dot(H_test,Wout)
                Y_pred_test = np.argmax(Y_pred_test,axis=1)
                aciertos = np.sum(Y_pred_test==y_test1)
                acierto= acierto + aciertos/y_test1.size

        
        
        
        
        acierto_medio.append(acierto/repeticiones)  
        tiempo_medio.append(tiempo/repeticiones)
          
    return neuronas, acierto_medio, tiempo_medio 