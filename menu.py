# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:27:14 2019

@author: Jorge
"""

import tkinter as tk

def menu(ventana,root):
    def salir():
        #añadir una ventana de aviso
        
        root.destroy()


        
    menubar = tk.Menu(ventana)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Salir", command=salir)
    menubar.add_cascade(label="Salir", menu=filemenu)

    ventana.config(menu=menubar)
