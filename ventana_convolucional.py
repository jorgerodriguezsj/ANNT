# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:11:25 2019

@author: Jorge
"""

import tkinter as tk
from tkinter import ttk
import crearcapasconv
from threading import Thread
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from matplotlib.figure import Figure
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import sklearn
import CNN
import menu

class Display(tk.Frame):
            def __init__(self,parent=0):
               tk.Frame.__init__(self,parent)
               self.output = tk.Text(self, width=80, height=15)
               self.output.pack(padx = 30, pady = 5,)
               sys.stdout = self
               self.pack()
        
            def flush(self):
                pass
    
            def write(self, txt):
                self.output.insert(tk.END,str(txt))
                self.output.see("end")
                self.update_idletasks()

#Función que crea una ventana para introducir los parametros necesarios para crear una red convolucional
def Ventana_convolucional(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio):

    ventana_convolucion = tk.Toplevel(ventana_seleccion)
    ventana_convolucion.geometry('725x600+500+200')
    #Insertamos menu
    menu.menu(ventana_convolucion,ventana_inicio)
    #Escondemos ventana anterior
    ventana_seleccion.withdraw()
    
    #título 
    labeltitulo = ttk.Label(ventana_convolucion,text = "Parámetros necesarios para la red Convolucional",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo.pack(pady=10)
    
    lframe = ttk.Frame(ventana_convolucion)
    lframe.pack()
    
    #------------------------ entrada de datos ---------------------------------
    #Tamaño del lote
    tamlot = tk.IntVar()
    lbtamlote = ttk.Label(lframe,text = "Tamaño lote: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lbtamlote.grid(column=0, row=0 ,pady=5,sticky=tk.W)
    etamlot = ttk.Entry(lframe,width=5, textvariable = tamlot)
    etamlot.grid(column=1, row=0,pady=5,sticky=tk.E)
    
    #Optimizador
    opt =tk.StringVar()
    lbopt = ttk.Label(lframe, text="Optimizador: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbopt.grid(column=0, row=1,pady=5,sticky=tk.W)
    cbopt=ttk.Combobox(lframe,width=9,state="readonly",textvariable = opt)
    cbopt["values"] = ["SGD", "RMSprop","Adam","Adagrad"]
    cbopt.grid(column = 1 ,row = 1,pady=5,columnspan=2)
    cbopt.current(0)
    
    #Proporción de validación
    pv = tk.DoubleVar()
    pv.set(0.2)
    lbpv = ttk.Label(lframe,text = "Proporción de Validación :",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lbpv.grid(column=0, row=2 ,pady=5,sticky=tk.W)
    epv = ttk.Entry(lframe,width=5, textvariable = pv)
    epv.grid(column=1, row=2,pady=5,sticky=tk.E)
    
    #Número de capas convolucionales
    ncon = tk.IntVar()
    lbncon = ttk.Label(lframe,text = "Número capas Convolucionales :",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lbncon.grid(column=0, row=3 ,pady=5,sticky=tk.W)
    encon = ttk.Entry(lframe,width=5, textvariable = ncon)
    encon.grid(column=1, row=3,pady=5,sticky=tk.E)

    #Número de capas completamente conectadas
    ncfc = tk.IntVar()
    lbncfc = ttk.Label(lframe,text = "Número capas completamente conectadas :",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lbncfc.grid(column=0, row=4 ,pady=5,sticky=tk.W)
    encfc = ttk.Entry(lframe,width=5, textvariable = ncfc)
    encfc.grid(column=1, row=4,pady=5,sticky=tk.E)
    
    #Función Loss
    fl =tk.StringVar()
    lbfl = ttk.Label(lframe, text="Función Loss: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbfl.grid(column=0, row=5,pady=5,sticky=tk.W)
    cbfl=ttk.Combobox(lframe,width=21,state="readonly",textvariable = fl)
    cbfl["values"] = ["kullback_leibler_divergence","mean_squared_error", "categorical_hinge",
        "categorical_crossentropy","binary_crossentropy","poisson","cosine_proximity"]
    cbfl.grid(column = 1 ,row = 5,pady=5,columnspan=2,sticky=tk.E)
    cbfl.current(3)
    
    #Métodos de parada
    labeltitulo1 = ttk.Label(ventana_convolucion,text = "Método de parada",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo1.pack(pady=10)
    
    lframe1 = ttk.Frame(ventana_convolucion)
    lframe1.pack()
    
    mp=tk.IntVar()
    bat1= ttk.Radiobutton(lframe1, value=0,variable=mp)
    bat1.grid(column=0, row=0)
    
    #Número de iteraciones antes de la parada
    nui=tk.IntVar()
    lbnui = ttk.Label(lframe1, text="Número de iteraciones: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbnui.grid(column=1, row=0,pady=5,sticky=tk.W)
    enui = ttk.Entry(lframe1,width=5, textvariable = nui)
    enui.grid(column=2, row=0,pady=5,sticky=tk.E)
    
    
    bat2 = ttk.Radiobutton(lframe1, value=1,variable=mp)
    bat2.grid(column=0, row=1)
    lbparada = ttk.Label(lframe1, text="Parada temprana: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbparada.grid(column = 1, row = 1,sticky=tk.W )
    
    #Parámetro a controlar para la parada
    lbcon = ttk.Label(lframe1, text="       Parámetro a controlar: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbcon.grid(column = 1, row = 2,pady=5,sticky=tk.W )
    con =tk.StringVar()
    cbcon=ttk.Combobox(lframe1,width=9,state="readonly",textvariable = con)
    cbcon["values"] = ["loss","val_loss", "acc","val_acc"]
    cbcon.grid(column = 2 ,row = 2,pady=5,sticky=tk.E)
    cbcon.current(0)
    
    #Delta mínimo
    delt =tk.DoubleVar()
    delt.set(0.001)
    lbdelt = ttk.Label(lframe1, text="       Delta min: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbdelt.grid(column=1, row=3,pady=5,sticky=tk.W)
    edelt = ttk.Entry(lframe1,width=5, textvariable = delt)
    edelt.grid(column=2, row=3,pady=5,sticky=tk.E)
    
    #Paciencia antes de parar
    pat =tk.IntVar()
    pat.set(3)
    lbpat = ttk.Label(lframe1, text="       Paciencia: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbpat.grid(column=1, row=4,pady=5,sticky=tk.W)
    epat = ttk.Entry(lframe1,width=5, textvariable = pat)
    epat.grid(column=2, row=4,pady=5,sticky=tk.E)
    
    #Función que define el modelo
    def crearmodelo():
        global NO,AC,BA,DR,NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC,numero_capas_conv,numero_capas_fc
        numero_capas_conv = int(ncon.get())
        numero_capas_fc = int(ncfc.get())
        NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC,NO,AC,BA,DR = crearcapasconv.capas(numero_capas_conv,
                                                                 numero_capas_fc, ventana_convolucion)
        
    btnmodelo = ttk.Button(ventana_convolucion, text = "Crear modelo",style='my.TButton', command=crearmodelo)
    btnmodelo.pack(pady=40)
    
    lframe2 = ttk.Frame(ventana_convolucion)
    lframe2.pack(side= "bottom")
    
    #Función que permite entrenar la red convolucional
    def entrenar():
        
        lote = tamlot.get()
        optimizador = opt.get()
        prop_val = pv.get()
        numero_capas_convolucionales = int(ncon.get())
        numero_capas_fullcon = int(ncfc.get())
        loss = fl.get()
        parada = mp.get()
        iteraciones = nui.get()
        control = con.get()
        delta = delt.get()
        paciencia = pat.get()
        
        #Excepciones
        if lote == 0:
            mb.showerror("Error", "Variable tamaño del lote = 0 ")
            return
        if prop_val == 0:
            mb.showerror("Error", "El algoritmo necesita una parte del conjunto de entrenamiento para su validación ")
            return
        if prop_val > 1:
            mb.showerror("Error", "Proporción de validación no válida ")
            return
        if numero_capas_convolucionales == 0:
            mb.showerror("Error", "Variable numero de capas convolucionales = 0 ")
            return
        if numero_capas_fullcon == 0:
            mb.showerror("Error", "Variable numero de capas completamente conectadas = 0 ")
            return
        if parada == 0 and iteraciones==0:
            mb.showerror("Error", "No se ha indicado el número de iteraciones requeridas ")
            return        
        if parada == 1 and delta==0.0:
            mb.showerror("Error", "No se ha indicado el mínimo delta para controlar la evolución ")
            return 
        
        while True:
            try:
                NF
                break
            except NameError:
                 mb.showerror("Error", "No se ha creado el modelo, haga click en crear modelo ")
                 return  
        
        for i in range(numero_capas_convolucionales) :
            if NF[i].get()==0:
                mb.showerror("Error", "Número de filtros = 0 ")
                return 
        for i in range(numero_capas_convolucionales) :
            if TF[i].get()==0:
                mb.showerror("Error", "Tamaño de filtro = 0 ")
                return     
        for i in range(numero_capas_convolucionales) :
            if PAS[i].get()==0:
                mb.showerror("Error", "Número de pasos = 0 ")
                return 
        for i in range(numero_capas_convolucionales) :
            if TAMPOL[i].get()==0:
                mb.showerror("Error", "Tamaño de Pooling = 0 ")
                return  
        for i in range(numero_capas_convolucionales) :
            if PASPOL[i].get()==0:
                mb.showerror("Error", "Pasos de pooling = 0 ")
                return 
        for i in range(numero_capas_convolucionales) :
            if DRC[i].get()> 1:
                mb.showerror("Error", "Dropout no válido ")
                return 
            
        for i in range(numero_capas_fullcon) :
            if NO[i].get()==0:
                mb.showerror("Error", "No es posible tener capas con 0 neuronas, asegurese de haber creado el modelo correctamente ")
                return 
        
        for i in range(numero_capas_fullcon) :
            if  DR[i].get() > 1:
                mb.showerror("Error", "Valor Dropout no válido ")
                return 
        
        
        #Ventana donde aparecerá el progreso del entrenamiento
        ventana_display = tk.Toplevel(ventana_convolucion)    
        labeltitulo1 = ttk.Label(ventana_display,text = "Entrenamiento",
                                foreground = "#054FAA",font=("Arial Bold", 15))
        labeltitulo1.pack(pady=5)
        
        #Función que dibuja la evolución del entrenamiento
        def plot():
        
            ventana_plot = tk.Toplevel(ventana_convolucion)
            ventana_plot.geometry('900x600')
            
            f = Figure(figsize = (5,5),dpi = 100)
            a = f.add_subplot(121)
            b = f.add_subplot(122)
            #Resumimos e imprimimos esos datos
            a.plot(entrenamiento.history['acc'])
            a.plot(entrenamiento.history['val_acc'])
            a.set_title('Precisión del modelo')
            a.set_ylabel('Precisión')
            a.set_xlabel('Iteraciones')
            a.legend(['Entrenamiento', 'Validación'], loc='upper left')
        
            # summarize history for loss
            b.plot(entrenamiento.history['loss'])
            b.plot(entrenamiento.history['val_loss'])
            b.set_title('Loss del modelo')
            b.set_ylabel('Loss')
            b.set_xlabel('Iteraciones')
            b.legend(['Entrenamiento', 'Validación'], loc='upper left')
            
            canvas1 = FigureCanvasTkAgg(f,ventana_plot)
            canvas1.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True)
            
            toolbar = NavigationToolbar2Tk(canvas1,ventana_plot)
            toolbar.update()
            canvas1._tkcanvas.pack(side = tk.TOP,fill = tk.BOTH, expand = True)  
        
        def guardarcompl():
            nombrearch=fd.asksaveasfilename(initialdir = "/",title = "Guardar como",defaultextension = 'h5')
            model.save(nombrearch)
            mb.showinfo("Información", "Los datos fueron guardados.")
            
        def guardarpesos():
            nombrearch=fd.asksaveasfilename(initialdir = "/",title = "Guardar como",defaultextension = 'h5')
            model.save_weights(nombrearch)
            mb.showinfo("Información", "Los datos fueron guardados.")
        def atras():
            ventana_display.destroy()
            
        framebotones = ttk.Frame(ventana_display)
        framebotones.pack(side= "bottom")
        
        btnguardarcompl = ttk.Button(framebotones, text="Modelo completo", 
                        command=guardarcompl,style='my.TButton',width = 15)
        btnguardarcompl.grid(row = 0, column = 0, padx = 10, pady = 5,sticky=tk.W)
        
        btnguardarpesos = ttk.Button(framebotones, text="Pesos", 
                        command=guardarpesos,style='my.TButton',width = 15)
        btnguardarpesos.grid(row = 0, column = 1, padx = 10, pady = 5,sticky=tk.W)
        
        btnplot = ttk.Button(framebotones, text="Plot", 
                        command=plot,style='my.TButton',width = 15)
        btnplot.grid(row = 1, column = 0, padx = 10, pady = 5,sticky=tk.W)
    
        btnatras = ttk.Button(framebotones, text="Atrás", 
                        command=atras,style='my.TButton',width = 15)
        btnatras.grid(row = 1, column = 1, padx = 10, pady = 5,sticky=tk.W) 
        
        def pantalla():
            Display(ventana_display)
        def run():
            global model, entrenamiento
            while True:
                try:
                    model, entrenamiento = CNN.cnn(ventana_convolucion,ventana_display,X_train,Y_train,X_test,Y_test,
                             lote,optimizador,prop_val,numero_capas_convolucionales,numero_capas_fullcon,loss,
                             parada,iteraciones,control,delta,paciencia, NF,TF,RE,PAS,PO,TAMPOL,PASPOL,ACON,DRC,NO,AC,BA,DR)
                    break
                except tk.TclError:
                    mb.showerror("Error desconocido", "Por favor vuelva a intentarlo ")
                    ventana_display.destroy()
                    return
                except RuntimeError:
                    mb.showerror("Error desconocido", "Por favor reinicie la aplicación ")
                    ventana_display.destroy()
                    return
                except sklearn.metrics.classification.UndefinedMetricWarning:
                    mb.showerror("Error ", "Algo salió mal con los datos, reinicie la aplicación y vuelva a intentarlo ")
                    ventana_display.destroy()
                    return                 
        
        t1=Thread(target=pantalla)
        t2=Thread(target=run)
        t1.start()
        t2.start()
    
    
    btntrain = ttk.Button(lframe2, text = "Entrenar",style='my.TButton', command=entrenar)
    btntrain.grid(row = 0, column = 1, padx = 20, pady=15)
    
    def atras():
        ventana_convolucion.destroy()
        ventana_seleccion.deiconify()
    
    btnatras = ttk.Button(lframe2, text = "Atras",style='my.TButton', command=atras)
    btnatras.grid(row=0,column=0, padx = 20, pady=15)

   
