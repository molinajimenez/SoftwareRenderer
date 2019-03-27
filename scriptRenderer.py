from tareaArte import *
from objReader import * 
import sys
from collections import namedtuple
V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

#pasamos x,y,nombre,t para render triangulizado

def menu():
    opc= True
    while opc:
        print("Menu para renderizar 3 figuras..")

        print("Escoge figura a renderizar :)")
        print("1. Mono")
        print("------------------------")
        op=input("Ingresa numero: ")

        if op=="1":
            mono()
        else:
            opc=False

def mono():
    utils=Render(800,600)
    try:
        utils.lookAt(V3(0,1,-1),V3(0,2,0),V3(0,0,0))
        utils.load(filename='monkey.obj', mtlFilename='monkey.mtl', scale=(0.25,0.25, 0.25))
        utils.glFinish("mono")
    except IndexError:
        print("La estas regando con la escala, se salio del framebuffer...")
        utils.glClearColor(1,0,0)
        utils.glFinish("tryMeError") 


menu()