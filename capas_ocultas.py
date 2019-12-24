# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:35:32 2019

@author: Jorge

Creacion del modelo - Capas ocultas
"""

from tkinter import *
from tkinter import ttk


numero_capas = 3

NO = {}
AC = {}
BA = {} #Aquí almaceno los valores 0-1 para batch Normalization
DR = {}


ventana_perceptron = Tk()

for i in range(numero_capas) :
#-----------------Capa----------------------------------    
    lframe = ttk.Labelframe(ventana_perceptron,text ="Capa {}".format(i+1))
    lframe.grid(column=0,row=i) 
#-----------------Linea intro neuronas----------------------------------     
    neu=IntVar()
    l1 = ttk.Label(lframe, text="Neuronas capa oculta: ")
    l1.grid(column=0, row=0)
    en = ttk.Entry(lframe,width=5,textvariable=neu)
    en.grid(column=1, row=0, pady=5,columnspan=2)
    NO[i]=neu
#-----------------Linea intro f.activacion----------------------------------      
    act = StringVar()
    l2 = ttk.Label(lframe, text="Función de activación: ")
    l2.grid(column=0, row=1)
    cob=ttk.Combobox(lframe,width=9,state="readonly",textvariable = act)
    cob["values"] = ["None", "softmax","relu","tanh","sigmoid","softplus",
       "softsign","hard_sigmoid","exponential","linear"]
    cob.grid(column = 1 ,row = 1,pady=5,columnspan=2)
    cob.current(0)
    AC[i]=act
#-----------------Linea intro batch normalization------------------------------
    batchn=IntVar()
    l3=ttk.Label(lframe,text = "Normalización del lote: ")
    l3.grid(column=0,row=2) 
    bat1= ttk.Radiobutton(lframe,text='Sí', value=0,variable=batchn)
    bat2 = ttk.Radiobutton(lframe,text='No', value=1,variable=batchn)
    bat1.grid(column=1, row=2)
    bat2.grid(column=2, row=2)
    BA[i]=batchn
#-----------------Linea intro dropout------------------------------
    drp=IntVar()
    l4 = ttk.Label(lframe, text="Dropout: ")
    l4.grid(column=0, row=3)
    en1 = ttk.Entry(lframe,width=5,textvariable=drp)
    en1.grid(column=1, row=3, pady=5,columnspan=2)
    DR[i]=drp
    
    
ventana_perceptron.mainloop()