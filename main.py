# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:27:26 2019

@author: jrodriguez119

"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

from tkinter import Tk, PhotoImage
from tkinter import ttk
import cargar



#Creamos raíz como ventana_inicio
ventana_inicio = Tk()

ventana_inicio.title("ANNT")
ventana_inicio.geometry('725x600+500+200')
ventana_inicio.resizable(width=False, height=False)

#Imagen principal
imagen_principal = PhotoImage(file="principal1.png")
logo = ttk.Label(ventana_inicio,image=imagen_principal)
logo.image = imagen_principal # keep a reference!
logo.pack()

#Título1
titulo = ttk.Label(ventana_inicio,text = "ANNT ",foreground = "#054FAA",
                              font=("Arial Bold", 25))
titulo.pack()

#Título2
titulo = ttk.Label(ventana_inicio,
                   text = "Aplicación para el entrenamiento de Redes Neuronales Artificiales ",
                   foreground = "#054FAA",
                   font=("Arial Bold", 15))
titulo.pack()

#Llamamos a la siguiente ventana, la de carga de datos
def carga():
    cargar.CargaDatos(ventana_inicio)
    

#definimos un estilo para los botones    
s = ttk.Style()
s.configure('my.TButton', font=('Arial Bold', 15),foreground = "#054FAA")
btnadelante = ttk.Button(ventana_inicio, text = "Inicio",style='my.TButton', command=carga)
btnadelante.pack(pady=75)

#Ultima línea
titulo = ttk.Label(ventana_inicio,
                   text = "Aplicación desarrollada por Jorge Rodríguez San José ",
                   foreground = "#054FAA",
                   font=("Arial Bold", 10))
titulo.pack(side = "bottom" ,pady=5)



ventana_inicio.mainloop()