# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:05:41 2019

@author: jrodriguez119
"""

import tkinter as tk
from tkinter import ttk
import crearcapas
import perceptron_multicapa
from threading import Thread
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from matplotlib.figure import Figure
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import menu
import sklearn

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
        
       
        
#Función que genera la ventana de parámetros del Perceptron multicapa
def Ventana_perceptron(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio):
    #Crear ventana
    ventana_perceptron = tk.Toplevel(ventana_seleccion)
    ventana_perceptron.geometry('725x600+500+200')
    #Insertar menu
    menu.menu(ventana_perceptron,ventana_inicio)
    #Esconder ventana previa
    ventana_seleccion.withdraw()
    #Título
    labeltitulo = ttk.Label(ventana_perceptron,text = "Parámetros necesarios para el Perceptrón",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo.pack(pady=10)
    
    #Frame donde alojar los widget de entrada
    lframe = ttk.Frame(ventana_perceptron)
    lframe.pack()
    
    #------------------------ entrada de datos ---------------------------------
    #Tamaño de lote
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
    cbopt["values"] = ["SGD", "RMSProp","Adam","Adagrad"]
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
    
    #Número de capas ocultas
    nco = tk.IntVar()
    lbnco = ttk.Label(lframe,text = "Número capas ocultas :",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lbnco.grid(column=0, row=3 ,pady=5,sticky=tk.W)
    enco = ttk.Entry(lframe,width=5, textvariable = nco)
    enco.grid(column=1, row=3,pady=5,sticky=tk.E)
    
    #Función Loss
    fl =tk.StringVar()
    lbfl = ttk.Label(lframe, text="Función Loss: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbfl.grid(column=0, row=4,pady=5,sticky=tk.W)
    cbfl=ttk.Combobox(lframe,width=21,state="readonly",textvariable = fl)
    cbfl["values"] = ["kullback_leibler_divergence","mean_squared_error", "categorical_hinge",
        "categorical_crossentropy","binary_crossentropy","poisson","cosine_proximity"]
    cbfl.grid(column = 1 ,row = 4,pady=5,columnspan=2,sticky=tk.E)
    cbfl.current(3)
    
    #Método de parada
    labeltitulo1 = ttk.Label(ventana_perceptron,text = "Método de parada",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo1.pack(pady=10)
    
    lframe1 = ttk.Frame(ventana_perceptron)
    lframe1.pack()
    
    #Tipo de parada
    
    #Parada por número de iteraciones
    mp=tk.IntVar()
    bat1= ttk.Radiobutton(lframe1, value=0,variable=mp)
    bat1.grid(column=0, row=0)
    
    nui=tk.IntVar()
    lbnui = ttk.Label(lframe1, text="Número de iteraciones: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbnui.grid(column=1, row=0,pady=5,sticky=tk.W)
    enui = ttk.Entry(lframe1,width=5, textvariable = nui)
    enui.grid(column=2, row=0,pady=5,sticky=tk.E)
    
    #Parada por control de un parámetro
    bat2 = ttk.Radiobutton(lframe1, value=1,variable=mp)
    bat2.grid(column=0, row=1)
    lbparada = ttk.Label(lframe1, text="Parada temprana: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbparada.grid(column = 1, row = 1,sticky=tk.W )
    
    #Parámetro a controlar
    lbcon = ttk.Label(lframe1, text="       Parámetro a controlar: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbcon.grid(column = 1, row = 2,pady=5,sticky=tk.W )
    con =tk.StringVar()
    cbcon=ttk.Combobox(lframe1,width=9,state="readonly",textvariable = con)
    cbcon["values"] = ["loss","val_loss", "acc","val_acc"]
    cbcon.grid(column = 2 ,row = 2,pady=5,sticky=tk.E)
    cbcon.current(0)
    
    #Delta mínima de evolución
    delt =tk.DoubleVar()
    delt.set(0.001)
    lbdelt = ttk.Label(lframe1, text="       Delta min: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbdelt.grid(column=1, row=3,pady=5,sticky=tk.W)
    edelt = ttk.Entry(lframe1,width=5, textvariable = delt)
    edelt.grid(column=2, row=3,pady=5,sticky=tk.E)
    
    #Paciencia para realizar la parada
    pat =tk.IntVar()
    pat.set(3)
    lbpat = ttk.Label(lframe1, text="       Paciencia: ",
                     foreground = "#054FAA",font=("Arial Bold", 12))
    lbpat.grid(column=1, row=4,pady=5,sticky=tk.W)
    epat = ttk.Entry(lframe1,width=5, textvariable = pat)
    epat.grid(column=2, row=4,pady=5,sticky=tk.E)
    
    #Función que abre una ventana externa y nos permite crear nuestro modelo editando las capas ocultas
    def crearmodelo():
        global NO,AC,BA,DR,numero_capas
        numero_capas = int(nco.get())
        NO,AC,BA,DR = crearcapas.capas(numero_capas, ventana_perceptron)
        
    btnmodelo = ttk.Button(ventana_perceptron, text = "Crear modelo",style='my.TButton', command=crearmodelo)
    btnmodelo.pack(pady=50)
    
    lframe2 = ttk.Frame(ventana_perceptron)
    lframe2.pack(side= "bottom")
    
    
    def entrenar():
        
        lote = tamlot.get()
        optimizador = opt.get()
        prop_val = pv.get()
        numero_capas_ocultas = int(nco.get())
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
        if numero_capas_ocultas == 0:
            mb.showerror("Error", "Variable numero de capas ocultas = 0 ")
            return
        if parada == 0 and iteraciones==0:
            mb.showerror("Error", "No se ha indicado el número de iteraciones requeridas ")
            return        
        if parada == 1 and delta==0.0:
            mb.showerror("Error", "No se ha indicado el mínimo delta para controlar la evolución ")
            return 
        
        while True:
            try:
                NO
                break
            except NameError:
                 mb.showerror("Error", "No se ha creado el modelo, haga click en crear modelo ")
                 return  
             
        for i in range(numero_capas_ocultas) :
            if NO[i].get()==0:
                mb.showerror("Error", "No es posible tener capas con 0 neuronas, asegurese de haber creado el modelo correctamente ")
                return 
        
        for i in range(numero_capas_ocultas) :
            if  DR[i].get() > 1:
                mb.showerror("Error", "Valor Dropout no válido ")
                return 
        
        

        #Ventana donde aparece el proceso y los botones para guardar el modelo
        ventana_display = tk.Toplevel(ventana_perceptron)    
        labeltitulo1 = ttk.Label(ventana_display,text = "Entrenamiento",
                                foreground = "#054FAA",font=("Arial Bold", 15))
        labeltitulo1.pack(pady=5)
        
        #Funcion que representa la evolución del entrenamiento
        def plot():
        
            ventana_plot = tk.Toplevel(ventana_perceptron)
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
            global Display
            Display(ventana_display)
        def run():
            global model, entrenamiento
            while True:
                try:
                    model, entrenamiento = perceptron_multicapa.Perceptron_multicapa(ventana_perceptron,ventana_display,X_train,Y_train,X_test,Y_test,
                             lote,optimizador,prop_val,numero_capas_ocultas,loss,
                             parada,iteraciones,control,delta,paciencia,NO,AC,BA,DR)
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
        ventana_perceptron.destroy()
        ventana_seleccion.deiconify()
    
    btnatras = ttk.Button(lframe2, text = "Atras",style='my.TButton', command=atras)
    btnatras.grid(row=0,column=0, padx = 20, pady=15)
    
