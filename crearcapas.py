# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:35:32 2019

@author: Jorge
"""

import tkinter as tk
from tkinter import ttk


#Funcion que genera una ventana en la que definir las capas ocultas del modelo
def capas(numero_capas, ventana_perceptron):
    global NO,AC,BA,DR, neu,act,batchn,drp
    #Diccionarios para almacenar las caracteristicas de cada capa
    NO = {}
    AC = {}
    BA = {} #Aquí almaceno los valores 0-1 para batch Normalization
    DR = {}
    
    
    ventana_capas = tk.Toplevel(ventana_perceptron)
    ventana_capas.geometry("400x600")
    s = ttk.Style()
    s.configure('my.TButton', font=('Arial Bold', 12),foreground = "#054FAA")
    
    def atras():
        ventana_capas.destroy()
        ventana_perceptron.deiconify()
        
    btnatras = ttk.Button(ventana_capas, text = "Ok",style='my.TButton', command=atras)
    btnatras.pack(pady=5)
    
    canvas = tk.Canvas(ventana_capas)
    scrollbar = ttk.Scrollbar(ventana_capas, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set) 
    
    for i in range(numero_capas) :
    #-----------------Capa----------------------------------    
        lframe = ttk.Labelframe(scrollable_frame,text ="Capa {}".format(i+1))
        lframe.grid(column=0,row=i, padx=90, pady = 10) 
    #-----------------Linea intro neuronas----------------------------------     
        neu=tk.IntVar()
        l1 = ttk.Label(lframe, text="Neuronas capa oculta: ")
        l1.grid(column=0, row=0)
        en = ttk.Entry(lframe,width=5,textvariable=neu)
        en.grid(column=1, row=0, pady=5,columnspan=2)
        NO[i]=neu
    #-----------------Linea intro f.activacion----------------------------------      
        act = tk.StringVar()
        l2 = ttk.Label(lframe, text="Función de activación: ")
        l2.grid(column=0, row=1)
        cob=ttk.Combobox(lframe,width=9,state="readonly",textvariable = act)
        cob["values"] = [ "softmax","relu","tanh","sigmoid","softplus",
           "softsign","hard_sigmoid","exponential","linear"]
        cob.grid(column = 1 ,row = 1,pady=5,columnspan=2)
        cob.current(0)
        AC[i]=act
    #-----------------Linea intro batch normalization------------------------------
        batchn=tk.IntVar()
        l3=ttk.Label(lframe,text = "Normalización del lote: ")
        l3.grid(column=0,row=2) 
        bat1= ttk.Radiobutton(lframe,text='Sí', value=0,variable=batchn)
        bat2 = ttk.Radiobutton(lframe,text='No', value=1,variable=batchn)
        bat1.grid(column=1, row=2)
        bat2.grid(column=2, row=2)
        BA[i]=batchn
    #-----------------Linea intro dropout------------------------------
        drp=tk.DoubleVar()
        l4 = ttk.Label(lframe, text="Dropout: ")
        l4.grid(column=0, row=3)
        en1 = ttk.Entry(lframe,width=5,textvariable=drp)
        en1.grid(column=1, row=3, pady=5,columnspan=2)
        DR[i]=drp
        
        canvas.pack(side="left", fill="both", expand=False)
        scrollbar.pack(side="right", fill="y")    
    
    return NO,AC,BA,DR

