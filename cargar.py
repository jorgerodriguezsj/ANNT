# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:09:12 2019

@author: jrodriguez119
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
from tkinter import filedialog as fd
from tkinter import messagebox as mb

#Para el menu
import menu
import ventana_seleccion

#Esta funcion nos genera la ventana de introducción de datos
def CargaDatos(ventana_inicio):
    global ventana_datos
    
    ventana_datos = tk.Toplevel(ventana_inicio)
    ventana_datos.geometry('725x600+500+200')
    ventana_datos.resizable(width=False, height=False)
    
    #Imagenes a usar
    cross = tk.PhotoImage(file="cross1.png")
    tick = tk.PhotoImage(file="tick1.png")
    
    #Definimos el menú superior que nos permite cerrar el programa
    menu.menu(ventana_datos,ventana_inicio)
    
    #Título de la ventana
    labeltitulo = ttk.Label(ventana_datos,text = "Introducción de los datos",foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo.pack(pady=5)

#-----------------Entrada datos-----------------------------------               
    
    lframe = ttk.Labelframe(ventana_datos,text ="")
    lframe.pack()
    global X_train,Y_train,X_test,Y_test

#-----------------X_train-----------------------------------     
    
    def cargarXtrain():
        global X_train
        nombrearch=fd.askopenfilename(parent = lframe,initialdir = "/",title = "Seleccione archivo Xtrain",filetypes = ((".txt","*.txt"),("todos los archivos","*.*")))
        X_train = np.loadtxt(nombrearch)
        X_train.astype("float32")        
        if X_train.shape[0]!=0:
            mb.showinfo("Información", "Los datos de entrenamiento han sido cargados correctamente.")
            cross1 = ttk.Label(lframe,image=tick)
            cross1.image = tick 
            cross1.grid(row=1,column = 3)
            labelresultado1 = ttk.Label(ventana_datos,text = 'Datos de entrenamiento '
                                       +"(" + str(X_train.shape[0]) + "," + str(X_train.shape[1]) +")",
                                       font=("Arial Bold", 12),justify="left",foreground = "#054FAA")
            labelresultado1.pack(pady=15)
            
        return 
    
    #-----------------Estilo botón-----------------------------------
    s = ttk.Style()
    s.configure('my.TButton', font=('Arial Bold', 12),foreground = "#054FAA")
    #Botón carga     
    btnloadXtrain = ttk.Button(lframe, text = "Cargar X_train",style='my.TButton', command= cargarXtrain)
    btnloadXtrain.grid(row=1,column =1, pady=10 ,padx=10)
   
#-----------------Y_train-----------------------------------          
    def cargarYtrain():
        global Y_train
        nombrearch=fd.askopenfilename(initialdir = "/",title = "Seleccione archivo Ytrain",filetypes = ((".txt","*.txt"),("todos los archivos","*.*")))
        Y_train = np.loadtxt(nombrearch)
        Y_train.astype("int")
        if Y_train.shape[0]!=0:
            mb.showinfo("Información", "Las etiquetas de entrenamiento han sido cargadas correctamente.")
            cross2 = ttk.Label(lframe,image=tick)
            cross2.image = tick 
            cross2.grid(row=2,column = 3)
            labelresultado2 = ttk.Label(ventana_datos,text = 'Etiquetas de entrenamiento ' 
                                       + "(" + str(Y_train.shape[0])+")",
                                       font=("Arial Bold", 12),justify="left",foreground = "#054FAA")
            labelresultado2.pack(pady=15)
        return Y_train
    #Botón carga 
    btnloadYtrain = ttk.Button(lframe, text = "Cargar Y_train",style='my.TButton', command=cargarYtrain)
    btnloadYtrain.grid(row=2,column=1, pady=10)

#-----------------X_test-----------------------------------    
    def cargarXtest():
        global X_test,nombrearch
        nombrearch=fd.askopenfilename(initialdir = "/",title = "Seleccione archivo Xtest",filetypes = ((".txt","*.txt"),("todos los archivos","*.*")))
        X_test = np.loadtxt(nombrearch)
        X_test.astype("float32")
        if X_test.shape[0]!=0:
            mb.showinfo("Información", "Los datos de test han sido cargados correctamente.")
            cross3 = ttk.Label(lframe,image=tick)
            cross3.image = tick 
            cross3.grid(row=3,column = 3)
            labelresultado3 = ttk.Label(ventana_datos,text = 'Datos de test ' 
                                        + "(" + str(X_test.shape[0]) + "," + str(X_test.shape[1]) +")",
                                       font=("Arial Bold", 12),justify="left",foreground = "#054FAA")
            labelresultado3.pack(pady=15)
        return X_test
    #Botón carga 
    btnloadXtest = ttk.Button(lframe, text = "Cargar X_test",style='my.TButton', command=cargarXtest)
    btnloadXtest.grid(row=3,column = 1, pady=10)
#-----------------Y_test-----------------------------------    
    def cargarYtest():
        global Y_test
        nombrearch=fd.askopenfilename(initialdir = "/",title = "Seleccione archivo Ytest",filetypes = ((".txt","*.txt"),("todos los archivos","*.*")))
        Y_test = np.loadtxt(nombrearch)
        Y_test.astype("int")
        if Y_test.shape[0]!=0:
            mb.showinfo("Información", "Las etiquetas de test han sido cargadas correctamente.")
            cross4 = ttk.Label(lframe,image=tick)
            cross4.image = tick 
            cross4.grid(row=4,column = 3)
            labelresultado4 = ttk.Label(ventana_datos,text = 'Etiquetas de test ' + "(" + str(Y_test.shape[0]) +")",
                                       font=("Arial Bold", 12),justify="left",foreground = "#054FAA")
            labelresultado4.pack(pady=15)
        return Y_test
    #Botón carga 
    btnloadYtest = ttk.Button(lframe, text = "Cargar Y_test",style='my.TButton', command=cargarYtest)
    btnloadYtest.grid(row=4,column = 1, pady=10)
    

#-----------------Etiquetas nombre-----------------------------------
    
    label1 = ttk.Label(lframe,text = "Conjunto de entrenamiento: ",foreground = "#054FAA",
                              font=("Arial Bold", 12),justify = "left")
    label1.grid(row=1,column=0,padx=10)
    
    label2 = ttk.Label(lframe,text = "Etiquetas de entrenamiento: ",foreground = "#054FAA",
                              font=("Arial Bold", 12),justify = "left")
    label2.grid(row=2,column=0,padx=10)
    
    label3 = ttk.Label(lframe,text = "Conjunto de test: ",foreground = "#054FAA",
                              font=("Arial Bold", 12),justify = "left")
    label3.grid(row=3,column=0,padx=10)
    
    label4 = ttk.Label(lframe,text = "Etiquetas de test: ",foreground = "#054FAA",
                              font=("Arial Bold", 12),justify = "left")
    label4.grid(row=4,column=0,padx=10)
    
    
#-----------------Cross y tick-----------------------------------
    
    cross1 = ttk.Label(lframe,image=cross)
    cross1.image = cross 
    cross1.grid(row=1,column = 3)
    
    cross2 = ttk.Label(lframe,image=cross)
    cross2.image = cross 
    cross2.grid(row=2,column = 3)
    
    cross3 = ttk.Label(lframe,image=cross)
    cross3.image = cross 
    cross3.grid(row=3,column = 3)
    
    cross4 = ttk.Label(lframe,image=cross)
    cross4.image = cross 
    cross4.grid(row=4,column = 3)
    
#-----------------boton atrás-----------------------------------
    def atras():
        ventana_datos.withdraw()
        ventana_inicio.deiconify()
    
#-----------------Boton continuar----------------------------------- TODO   
    def continuar():
        global X_train,Y_train,X_test,Y_test

        if "X_train" in globals():
            pass
        else:
            mb.showerror("Error", "Variable X_train no definida")
            return
        
        if "X_test" in globals():
            pass
        else:
            mb.showerror("Error", "Variable X_test no definida")
            return
        
        if "Y_train" in globals():
            pass
        else:
            mb.showerror("Error", "Variable Y_train no definida")
            return

        if "Y_test" in globals():
            pass
        else:
            mb.showerror("Error", "Variable Y_test no definida")
            return
        
        ventana_seleccion.seleccion_modo(ventana_datos,X_train,Y_train,X_test,Y_test,ventana_inicio) 
        
        return 
    
    #Botón atras
    btnatras = ttk.Button(ventana_datos, text = "Atras",style='my.TButton', command=atras)
    btnatras.pack(side= "bottom",pady=5)
    
    #Botón continuar
    btncontinuar = ttk.Button(ventana_datos, text = "Continuar",style='my.TButton', command=continuar)
    btncontinuar.pack(side= "bottom",pady=5)
    
    #Escondemos la ventana anterior
    ventana_inicio.withdraw()
    
      
    
    
    
    
    