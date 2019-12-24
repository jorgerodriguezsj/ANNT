# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:01:08 2019

@author: jrodriguez119
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb
import elm
import pickle
from threading import Thread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
from matplotlib.figure import Figure
import sys
import menu

#Funcion que crea la ventana para la red ELM
def parametroselm(ventana_seleccion,X_train,Y_train,X_test,Y_test,ventana_inicio):
   
    #Escondemos la ventana anterior
    ventana_seleccion.withdraw()
    
    #Caracteristicas de la ventana
    ventana_elm = tk.Toplevel(ventana_seleccion)
    ventana_elm.geometry('725x600+500+200')
    ventana_elm.resizable(width=False, height=False)
    
    #Insertamos menu
    menu.menu(ventana_elm,ventana_inicio)
    
    #Creamos dos pestañas 
    pestanas = ttk.Notebook(ventana_elm)
    tab1 = ttk.Frame(pestanas)
    tab2 = ttk.Frame(pestanas)
    pestanas.add(tab1, text='ELM')
    pestanas.add(tab2, text='ELM con Barrido')
    pestanas.pack(expand=1, fill='both')
    
    
##############################################################################   
#---------------------------ELM ----------------------------------------------
##############################################################################
    labeltitulo = ttk.Label(tab1,text = "Parámetros necesarios para la ELM",
                            foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo.pack(pady=5)
    
    #Frame para introducir los widget
    lframe = ttk.Frame(tab1)
    lframe.pack()
    
    #Definimos posiciones
    W=tk.W
    E=tk.E
    
    #-----------------Entrada neuronas-----------------------------------    
    neuronas = tk.IntVar()
    lblneuronas = ttk.Label(lframe,text = "Número de neuronas: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lblneuronas.grid(column=0, row=2 ,pady=5,sticky=W)
    eneuronas = ttk.Entry(lframe,width=5, textvariable = neuronas)
    eneuronas.grid(column=1, row=2,pady=5,sticky=E)
    
    
    #-----------------Entrada proporcion datos-----------------------------------   
    proporcion = tk.DoubleVar()
    lblproporcion = ttk.Label(lframe,text = "Proporción de datos (0-1): ",
                            foreground = "#054FAA",font=("Arial Bold", 12)) 
    lblproporcion.grid(column=0, row=3,pady=5,sticky=W)
    eproporcion = ttk.Entry(lframe,width=5,textvariable=proporcion)
    eproporcion.grid(column=1, row=3,pady=5,sticky=E)
    
    lframe1 = ttk.Frame(tab1)
    lframe1.pack()
    
    #-----------------Entrada GPU o CPU-----------------------------------
    warning1 = tk.PhotoImage(file="warning.png")
    s = ttk.Style()
    s.configure('my.TRadiobutton', font=('Arial Bold', 12),foreground = "#054FAA")
    xptemp = tk.IntVar()
    motor1 = ttk.Radiobutton(lframe1,text='GPU', value=0,variable=xptemp,style='my.TRadiobutton')
    motor2 = ttk.Radiobutton(lframe1,text='CPU', value=1,variable=xptemp,style='my.TRadiobutton')
    motor1.grid(row=0, column=0,pady=5,padx=10)
    motor2.grid(row=0, column=1,pady=5,padx=10)
    
    #Función y botón que alertan del uso de la GPU
    def warning():
        mb.showwarning(message="¡Cuidado!\nPara elegir GPU tiene que estar seguro de\nque su tarjeta gráfica admite ese volumen",
                      title="Cuidado")
    botonwarning = ttk.Button(lframe1, image = warning1,command = warning)
    botonwarning.image = warning1
    botonwarning.grid(row = 0 , column = 2, padx=3)


##############################################################################   
#-------------------ELM barrido----------------------------------------------
##############################################################################
    labeltitulo1 = ttk.Label(tab2,text = "Parámetros necesarios para la ELM",
                            foreground = "#054FAA",font=("Arial Bold", 15))
    labeltitulo1.pack(pady=5)
    
    lframe1 = ttk.Frame(tab2)
    lframe1.pack()
    W=tk.W
    E=tk.E
    
    #-----------------Entrada neuronas inicio----------------------------------   
    neuronasini = tk.IntVar()
    lblneuronas = ttk.Label(lframe1,text = "Neuronas de inicio: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lblneuronas.grid(column=0, row=0 ,pady=5,sticky=W)
    eneuronas = ttk.Entry(lframe1,width=5, textvariable = neuronasini)
    eneuronas.grid(column=1, row=0,pady=5,sticky=E)
    
    #-----------------Entrada neuronas fin----------------------------------
    neuronasfin = tk.IntVar()
    lblneuronas = ttk.Label(lframe1,text = "Neuronas de fin: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lblneuronas.grid(column=0, row=1 ,pady=5,sticky=W)
    eneuronas = ttk.Entry(lframe1,width=5, textvariable = neuronasfin)
    eneuronas.grid(column=1, row=1,pady=5,sticky=E)
    
    #-----------------Entrada número de pasos----------------------------------
    intervalo = tk.IntVar()
    lblneuronas = ttk.Label(lframe1,text = "Pasos: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lblneuronas.grid(column=0, row=2 ,pady=5,sticky=W)
    eneuronas = ttk.Entry(lframe1,width=5, textvariable = intervalo)
    eneuronas.grid(column=1, row=2,pady=5,sticky=E)
    
    #-----------------Entrada número repeticiones media------------------------
    repeticiones = tk.IntVar()
    lblneuronas = ttk.Label(lframe1,text = "Repeticiones para la media: ",
                            foreground = "#054FAA",font=("Arial Bold", 12))
    lblneuronas.grid(column=0, row=3 ,pady=5,sticky=W)
    eneuronas = ttk.Entry(lframe1,width=5, textvariable = repeticiones)
    eneuronas.grid(column=1, row=3,pady=5,sticky=E)
    
    
    #Frame para meter los botones
    lframe2 = ttk.Frame(tab2)
    lframe2.pack()
    
    #-----------------Entrada GPU o CPU-----------------------------------
    s = ttk.Style()
    s.configure('my.TRadiobutton', font=('Arial Bold', 12),foreground = "#054FAA")
    xptemp1 = tk.IntVar()
    motor1 = ttk.Radiobutton(lframe2,text='GPU', value=0,variable=xptemp1,style='my.TRadiobutton')
    motor2 = ttk.Radiobutton(lframe2,text='CPU', value=1,variable=xptemp1,style='my.TRadiobutton')
    motor1.grid(row=0, column=0,pady=5,padx=10)
    motor2.grid(row=0, column=1,pady=5,padx=10)
    botonwarning = ttk.Button(lframe2, image = warning1,command = warning)
    botonwarning.image = warning1
    botonwarning.grid(row = 0 , column = 3, padx=3)
  
 
            
    #Función que inicia la progressbar
    def entrenando():
        global progressbar,lb
        progressbar = ttk.Progressbar(progressframe, mode="indeterminate")
        progressbar.pack()
        progressbar.start(30)
    
    #Función que llama a entrenar a la ELM
    def entrenar():
        #fijamos los valores obtenidos en entry
        proporcion_datos = proporcion.get()
        numero_neuronas = neuronas.get()
        xp = xptemp.get()
        
        #Excepciones
        if numero_neuronas == 0:
            mb.showerror("Error", "Variable número de neuronas = 0 ")
            progressbar.destroy()
            return
        if proporcion_datos == 0.0:
            mb.showerror("Error", "Variable proporcion de datos = 0 ")
            progressbar.destroy()
            return
        if proporcion_datos > 1.0:
            mb.showerror("Error", "Variable proporcion de datos no válida ")
            progressbar.destroy()
            return  
        
            
        import cupy
        while True:
            try:
                neuronas_ocultas,precision_test,tiempo, Win , Wout, Bias,Y_test1,Y_pred_test = elm.ELM(X_train,
                                                    X_test,Y_train,Y_test,numero_neuronas,
                                                    proporcion_datos,xp)
                break
            except cupy.cuda.memory.OutOfMemoryError:
                 mb.showerror("Error", "La GPU no tiene memoria suficiente ")
                 progressbar.destroy()
                 return
                
        #Título
        labeltitulo1 = ttk.Label(superframe,text = "Resultados ELM",
                            foreground = "#054FAA",font=("Arial Bold", 15))
        labeltitulo1.grid(row=0, columnspan = 2)
            
           
    #---------------Imprimir los resultados en pantalla-----------------------
    
        resultadoneuronas = ttk.Label(superframe,text = "Número de Neuronas: ",
                                  font=("Arial Bold", 12),foreground = "#054FAA")
        resultadoneuronas.grid(column=0, row=1,sticky=W,pady=5)

        resultadoneuronas1 = ttk.Label(superframe,
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadoneuronas1.grid(column=1, row=1, sticky=W)
        
        resultadoprop = ttk.Label(superframe,text = "Datos empleados: ",
                              font=("Arial Bold", 12), foreground = "#054FAA")
        resultadoprop.grid(column=0, row=2,sticky=W,pady=5)

        resultadoprop1 = ttk.Label(superframe,
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadoprop1.grid(column=1, row=2, sticky=W)    
        
        resultadoprec = ttk.Label(superframe,text = "Aciertos Test: ",
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadoprec.grid(column=0, row=3,sticky=W,pady=5)

        resultadoprec1 = ttk.Label(superframe,
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadoprec1.grid(column=1, row=3, sticky=W)
        
        resultadotiempo = ttk.Label(superframe,text = "Tiempo entrenamiento: ",
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadotiempo.grid(column=0, row=4, sticky=W,pady=5)

        resultadotiempo1 = ttk.Label(superframe,
                              font=("Arial Bold", 12),foreground = "#054FAA")
        resultadotiempo1.grid(column=1, row=4, sticky=W)
            
            
            
    #---------------Resultados neuronas------------------------------    
        resultadoneuronas1['text'] =  str(neuronas_ocultas)
   
    #---------------Resultados lote------------------------------  
        resultadoprop1['text'] = str(proporcion_datos*100)+" " + "%"
                                       
    #---------------Resultados precision------------------------------   
        resultadoprec1['text'] =  '{:.2f}'.format((precision_test*100))+" " + "%"
      
    #---------------Resultados tiempo------------------------------   
        resultadotiempo1['text'] =  '{:.3f}'.format((tiempo))+" " + "s"
                              
        #Función que nos permite guardar el modelo        
        def guardarelm():
            nombrearch=fd.asksaveasfilename(initialdir = "/",title = "Guardar como")
            if nombrearch!='':
                archiWin=open(nombrearch, 'wb')
                pickle.dump((Win,Wout,Bias), archiWin)
                archiWin.close()
                mb.showinfo("Información", "Los datos fueron guardados en el archivo.")
                
        """
        
        Los datos del modelo se descargan en formato binario serializado, para su posterior uso 
        hay que cargarlos mediante
            archivo = open("modelo","rb")
            Win,Wout,Bias = pickle.load(archivo)
            
        """
        
        #Función que saca los detalles del test del modelo por clases
        def detalles():
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
            
            #Creamos el display donde aparecen los datos por clases
            def pantalla():
                ventana_display = tk.Toplevel(ventana_elm)
                labeltitulo1 = ttk.Label(ventana_display,text = "Resultados por clase",
                    foreground = "#054FAA",font=("Arial Bold", 15))
                labeltitulo1.pack(pady=5)
                Display(ventana_display)
            #Llamada a función que nos devuelve info por clases
            def run():
                from sklearn.metrics import classification_report
                print(classification_report(Y_test1, Y_pred_test))
            
        
            t1=Thread(target=pantalla)
            t2=Thread(target=run)
            t1.start()
            t2.start()
        
        #frame que contiene los botones
        miniframe = tk.Frame(superframe)
        miniframe.grid(row = 5, columnspan = 2)
        
        #Botón para invocar los detalles
        btndet = ttk.Button(miniframe, text = "Resultados por Clase",style='my.TButton', command=detalles)
        btndet.grid(row = 0, column = 1, padx = 20, pady=15)
        #Botón guardar datos    
        btnsave = ttk.Button(miniframe, text="Guardar modelo", 
                    command=guardarelm,style='my.TButton')
        btnsave.grid(pady=15, row = 0, column = 0)
        
        
        progressbar.destroy()


###############################################################################

    #Progressbar  
    def entrenando1():
        global progressbar,lb
        progressbar = ttk.Progressbar(progressframe1, mode="indeterminate")
        progressbar.pack()
        progressbar.start(30)

    #Función que llama a entrenar a ELM1
    def entrenar1():
        #Fijamos los datos necesarios obtenidos
        ini = neuronasini.get()
        fin = neuronasfin.get()
        inter = intervalo.get()
        rep = repeticiones.get()
        xp = xptemp1.get()
     
        #Robustez
        if ini == 0:
            mb.showerror("Error", "Variable número de neuronas iniciales = 0 ")
            progressbar.destroy()
            return
        if fin == 0:
            mb.showerror("Error", "Variable número de neuronas finales = 0 ")
            progressbar.destroy()
            return
        if rep == 0:
            mb.showerror("Error", "Variable número de repeticiones = 0 ")
            progressbar.destroy()
            return
        if ini == 0:
            mb.showerror("Error", "Variable número de neuronas iniciales = 0 ")
            progressbar.destroy()
            return
        if ini > fin :
            mb.showerror("Error", "Intervalo no válido ")
            progressbar.destroy()
            return
        if inter == 0 and ini!=fin:
            mb.showerror("Error", "Paso no válido ")
            progressbar.destroy()
            return
        
        import cupy
        while True:
            try:
                neuronas, acierto_medio, tiempo_medio = elm.ELM1(X_train,X_test,Y_train,Y_test,ini,fin,inter,
                                                                 xp,rep)
                break
            except cupy.cuda.memory.OutOfMemoryError:
                 mb.showerror("Error", "La GPU no tiene memoria suficiente ")
                 progressbar.destroy()
                 return
        
        #Limpiamos para la reescritura
        canvas.delete(scrollable_frame)   

        #Título
        labeltitulo1 = ttk.Label(scrollable_frame,text = "Estudio ELM",
                            foreground = "#054FAA",font=("Arial Bold", 15))
        labeltitulo1.grid(pady=5,row=0, columnspan = 3)
      
   
        #---------------Labels resultados-------------------------------------- 
        
        lblneuronas = ttk.Label(scrollable_frame,text = "Neuronas",
                                  font=("Arial Bold", 12),foreground = "#054FAA")
        lblneuronas.grid(column=0, row=1, sticky=W,padx=10)
        
        lblAcierto = ttk.Label(scrollable_frame,text = "Acierto",
                                  font=("Arial Bold", 12),foreground = "#054FAA")
        lblAcierto.grid(column=1, row=1, sticky=W,padx=10)
        
        lblTiempo = ttk.Label(scrollable_frame,text = "Tiempo",
                                  font=("Arial Bold", 12),foreground = "#054FAA")
        lblTiempo.grid(column=2, row=1, sticky=W,padx=10)
        
        lblinea = ttk.Label(scrollable_frame,text = "________________________________",
                                  font=("Arial Bold", 12),foreground = "#054FAA")
        lblinea.grid(column = 0 , columnspan=3,row=2)
        
        for i in range(len(neuronas)):
            resultadoneuronas = ttk.Label(scrollable_frame,
                                  font=("Arial Bold", 12),foreground = "#054FAA")
            resultadoneuronas.grid(column=0, row=i+3, sticky=W,padx=10)
            
            resultadoacierto = ttk.Label(scrollable_frame,
                                  font=("Arial Bold", 12),foreground = "#054FAA")
            resultadoacierto.grid(column=1, row=i+3, sticky=W,padx=10)
            
            resultadotiempo = ttk.Label(scrollable_frame,
                                  font=("Arial Bold", 12),foreground = "#054FAA")
            resultadotiempo.grid(column=2, row=i+3, sticky=W,padx=10)
               
                
                
        #---------------Resultados neuronas------------------------------    
            resultadoneuronas['text'] =  str(neuronas[i])
                                         
        #---------------Resultados precision------------------------------   
            resultadoacierto['text'] =  '{:.2f}'.format((acierto_medio[i]*100))+" " + "%"
          
        #---------------Resultados tiempo------------------------------       
            resultadotiempo['text'] =  '{:.3f}'.format((tiempo_medio[i]))+" " + "s"
        
        #Fijamos el frame y el Canvas                      
        superframe1.pack()  
        canvas.pack(side="left", fill="x", expand=False)
        scrollbar.pack(side="right", fill="y") 
        
        #Eliminar progressbar
        progressbar.destroy()

        #Plot
        if  ini!=fin:
            ventana_plot = tk.Toplevel(tab2)
            ventana_plot.geometry('900x600')
            
            f = Figure(figsize = (5,5),dpi = 100)
            a = f.add_subplot(121)
            b = f.add_subplot(122)
            
            a.set_title('Aciertos')
            a.set_ylabel('Acierto')
            a.set_xlabel('Número de neuronas')
            a.plot(neuronas,acierto_medio)
            
            b.set_title('Tiempo')
            b.set_ylabel('Tiempo (s)')
            b.set_xlabel('Número de neuronas')
            b.plot(neuronas,tiempo_medio)
            
            canvas1 = FigureCanvasTkAgg(f,ventana_plot)
            canvas1.get_tk_widget().pack(side = tk.TOP,fill = tk.BOTH, expand = True)
            
            toolbar = NavigationToolbar2Tk(canvas1,ventana_plot)
            toolbar.update()
            canvas1._tkcanvas.pack(side = tk.TOP,fill = tk.BOTH, expand = True)
        
    #Funciones para la ejecución simultanea del progressbar y el entrenamiento
    def run():
        t1=Thread(target=entrenando)
        t2=Thread(target=entrenar)
        t1.start()
        t2.start()
        
    def run1():
        t1=Thread(target=entrenando1)
        t2=Thread(target=entrenar1)
        t1.start()
        t2.start()
        
    #Botón entrenar
    btnelm = ttk.Button(tab1, text="Entrenar", 
                    command=run,style='my.TButton')
    btnelm.pack(pady=5)
    
    #Frame para alojar progressbar
    progressframe = ttk.Frame(tab1)
    progressframe.pack()
    
    #Botón entrenar barrido
    btnelm1 = ttk.Button(tab2, text="Entrenar", 
                    command=run1,style='my.TButton')
    btnelm1.pack(pady=5)
    
    #Frame para alojar progressbar 
    progressframe1 = ttk.Frame(tab2)
    progressframe1.pack()
    
    #Líneas estéticas 
    linea = ttk.Label(tab1,text = "----------------------------------------------------------------------------",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    linea.pack(pady=5)
    linea1 = ttk.Label(tab2,text = "----------------------------------------------------------------------------",
                                foreground = "#054FAA",font=("Arial Bold", 15))
    linea1.pack(pady=5)
    
    #Frames para alojar el canvas
    superframe= ttk.Frame(tab1)
    superframe.pack()
    superframe1= ttk.Frame(tab2)
    canvas = tk.Canvas(superframe1)
    
    #Frame con scroll
    scrollbar = ttk.Scrollbar(superframe1, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)  
    
    #Función atrás
    def atras():
        ventana_elm.withdraw()
        ventana_seleccion.deiconify()
        
    #Botón atrás    
    btnatras = ttk.Button(tab1, text = "Atras",style='my.TButton', command=atras)
    btnatras.pack(side= "bottom",pady=5)
    btnatras = ttk.Button(tab2, text = "Atras",style='my.TButton', command=atras)
    btnatras.pack(side= "bottom",pady=5)