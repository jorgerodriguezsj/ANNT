# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:50:55 2019

@author: Jorge
"""


import tkinter as tk
from tkinter import ttk



def capas(numero_capas_conv,numero_capas_fc, ventana_convolucion):
    #Capas convolucionales
    global NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC
    NF={}
    TF={}
    RE={}
    PAS={}
    PO={}
    TAMPOL={}
    PASPOL={}
    ACON={}
    DRC={}
        
    #Capas conectadas completamente
    global NO,AC,BA,DR, neu,act,batchn,drp
    NO = {}
    AC = {}
    BA = {} #Aquí almaceno los valores 0-1 para batch Normalization
    DR = {}
    
    
    ventana_capas = tk.Toplevel(ventana_convolucion)
    ventana_capas.geometry("400x600")
    s = ttk.Style()
    s.configure('my.TButton', font=('Arial Bold', 12),foreground = "#054FAA")
    
    def atras():
        ventana_capas.destroy()
        ventana_convolucion.deiconify()
        
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

    #-----------------Capas convolucionales----------------------------------      
    for i in range(numero_capas_conv):
        lframe0 = ttk.Labelframe(scrollable_frame,text ="Capa convolucional {}".format(i+1))
        lframe0.grid(column=0,row=i, padx=90, pady = 10) 
        
    #-----------------Intro filtros----------------------------------         
        fil=tk.IntVar()
        l1 = ttk.Label(lframe0, text="Número de filtros: ")
        l1.grid(column=0, row=0,sticky=tk.W,padx=10)
        efil = ttk.Entry(lframe0,width=5,textvariable=fil)
        efil.grid(column=1, row=0, pady=5,columnspan=2)
        NF[i]=fil
        
        tamfil=tk.IntVar()
        l2 = ttk.Label(lframe0, text="Tamaño del filtro : ")
        l2.grid(column=0, row=1,sticky=tk.W,padx=10)
        etamfil = ttk.Entry(lframe0,width=5,textvariable=tamfil)
        etamfil.grid(column=1, row=1, pady=5,columnspan=2)
        TF[i]=tamfil     
        
        actcon = tk.StringVar()
        l3 = ttk.Label(lframe0, text="Función de activación: ")
        l3.grid(column=0, row=2,sticky=tk.W,padx=10)
        cobcon=ttk.Combobox(lframe0,width=9,state="readonly",textvariable = actcon)
        cobcon["values"] = [ "softmax","relu","tanh","sigmoid","softplus",
           "softsign","hard_sigmoid","exponential","linear"]
        cobcon.grid(column = 1 ,row = 2,pady=5,columnspan=2)
        cobcon.current(0)
        ACON[i]=actcon
        
        rell = tk.StringVar()
        l4 = ttk.Label(lframe0, text="Padding (Relleno): ")
        l4.grid(column=0, row=3,sticky=tk.W,padx=10)
        cobrell=ttk.Combobox(lframe0,width=9,state="readonly",textvariable = rell)
        cobrell["values"] = [ "same","valid"]
        cobrell.grid(column = 1 ,row = 3,pady=5,columnspan=2)
        cobrell.current(0)
        RE[i]=rell
        
        pasconv=tk.IntVar()
        l9 = ttk.Label(lframe0, text="Pasos: ")
        l9.grid(column=0, row=4,sticky=tk.W,padx=10)
        epasconv = ttk.Entry(lframe0,width=5,textvariable=pasconv)
        epasconv.grid(column=1, row=4, pady=5,columnspan=2)
        PAS[i]=pasconv
        
        pol=tk.IntVar()
        l5=ttk.Label(lframe0,text = "Pooling: ")
        l5.grid(column=0,row=5,sticky=tk.W,padx=10) 
        pol1= ttk.Radiobutton(lframe0,text='Sí', value=0,variable=pol)
        pol2 = ttk.Radiobutton(lframe0,text='No', value=1,variable=pol)
        pol1.grid(column=1, row=5)
        pol2.grid(column=2, row=5)
        PO[i]=pol
        
        
        tampol=tk.IntVar()
        l6 = ttk.Label(lframe0, text="                   Tamaño: ")
        l6.grid(column=0, row=6,sticky=tk.W,padx=10)
        etampol = ttk.Entry(lframe0,width=5,textvariable=tampol)
        etampol.grid(column=1, row=6, pady=5,columnspan=2)
        TAMPOL[i]=tampol
        
        paspol=tk.IntVar()
        l7 = ttk.Label(lframe0, text="                   Pasos: ")
        l7.grid(column=0, row=7,sticky=tk.W,padx=10)
        epaspol = ttk.Entry(lframe0,width=5,textvariable=paspol)
        epaspol.grid(column=1, row=7, pady=5,columnspan=2)
        PASPOL[i]=paspol
        
        drc=tk.DoubleVar()
        l8 = ttk.Label(lframe0, text="Dropout")
        l8.grid(column=0, row=8,sticky=tk.W,padx=10)
        edrc = ttk.Entry(lframe0,width=5,textvariable=drc)
        edrc.grid(column=1, row=8, pady=5,columnspan=2)
        DRC[i]=drc
    
    for i in range(numero_capas_fc) :
    #-----------------Capa----------------------------------    
        lframe = ttk.Labelframe(scrollable_frame,text ="Capa completamente conectada {}".format(i+1))
        lframe.grid(column=0,row=numero_capas_conv+i, padx=90, pady = 10) 
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
    

    
    
    return NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC,NO,AC,BA,DR,

