# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:27:49 2019

@author: jrodriguez119
"""

import tkinter as tk
from tkinter import ttk
import webbrowser
import ventana_elm
import ventana_perceptron
import ventana_convolucional

import menu

#Ventana que nos permite seleccionar el tipo de red a entrenar
   
def seleccion_modo(ventana_datos,X_train,Y_train,X_test,Y_test,ventana_inicio):
    #Escondemos la ventana anterior
    ventana_datos.withdraw()
    
    #Creamos la nueva ventana
    ventana_seleccion = tk.Toplevel(ventana_datos)
    ventana_seleccion.geometry('725x600+500+200')
    ventana_seleccion.resizable(width=False, height=False)
    
    #Titulo
    labeltitulo = ttk.Label(ventana_seleccion,text = "Selecciona el tipo de red a entrenar",
                            foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo.pack(pady=5)
    
    #Frame para introducir los botones de selección
    lframe = ttk.Frame(ventana_seleccion)
    lframe.pack()
    
    #Imagen para los botones info
    info = tk.PhotoImage(file="info1.png")
    
    #Funciones de información         
    def infoperceptron():
        webbrowser.open("https://en.wikipedia.org/wiki/Multilayer_perceptron", new=2, autoraise=True)
    def infoelm():
        webbrowser.open("https://en.wikipedia.org/wiki/Extreme_learning_machine", new=2, autoraise=True)
    def infocnn():
        webbrowser.open("https://en.wikipedia.org/wiki/Convolutional_neural_network", new=2, autoraise=True)
    
    #Llamadas a cada una de las diferentes funciones para generar cada red
    def elm():
        ventana_elm.parametroselm(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio)
        
    def percept():
        ventana_perceptron.Ventana_perceptron(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio)
    
    def cnn():
        ventana_convolucional.Ventana_convolucional(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio)
        
    #Conjunto de botones de selección    
    btnperceptron = ttk.Button(lframe, text = "Perceptrón Multicapa",style='my.TButton', command= percept, width = 25)
    btnperceptron.grid(row=0,column =0, pady=10 ,padx=10)
    
    btncnn = ttk.Button(lframe, text = "Red Neuronal Convolucional",style='my.TButton', command= cnn,width = 25)
    btncnn.grid(row=1,column =0, pady=10 ,padx=10)
    
    btnelm = ttk.Button(lframe, text = "Extreme Learning Machine",style='my.TButton', command= elm ,width = 25)
    btnelm.grid(row=2,column =0, pady=10 ,padx=10)
    
    #Botones de información
    botoninfo = ttk.Button(lframe, image=info,command = infoperceptron)
    botoninfo.grid(row = 0 , column = 1,padx=3)
    
    botoninfo1 = ttk.Button(lframe, image=info,command = infocnn)
    botoninfo1.grid(row = 1 , column = 1,padx=3)
    
    botoninfo2 = ttk.Button(lframe, image=info,command = infoelm)
    botoninfo2.grid(row = 2 , column = 1,padx=3)


    #función atrás
    def atras():
        ventana_seleccion.withdraw()
        ventana_datos.deiconify()
    #Botón atrás    
    btnatras = ttk.Button(ventana_seleccion, text = "Atras",style='my.TButton', command=atras)
    btnatras.pack(side= "bottom",pady=5)

    #Imagen decorativa
    imagen_principal = tk.PhotoImage(file="nn.png")
    logo = ttk.Label(ventana_seleccion,image=imagen_principal)
    logo.image = imagen_principal 
    logo.pack(pady=40)


    #Insertamos el menú en la ventana
    menu.menu(ventana_seleccion,ventana_inicio)
    
    ventana_seleccion.mainloop()