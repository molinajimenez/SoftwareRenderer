from tareaArte import *
from objReader import * 
import sys
from collections import namedtuple
V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
from math import pi

#pasamos x,y,nombre,t para render triangulizado

def menu():
    opc= True
    while opc:
        print("Menu para renderizar 3 figuras..")

        print("Escoge figura a renderizar :)")
        print("1. Low Angle")
        print("2. High Angle")
        print("3. Dutch Angle")
        print("4. Medium Angle")
        print("------------------------")
        op=input("Ingresa numero: ")

        if op=="1":
            Low()
        if op=="2":
            High()
        if op=="3":
            Dutch()
        if op=="4":
            medium()
        else:
            opc=False

def Low():
    utils=Render(1200,1200)
    #def load(self, filename, mtlFilename=None, translate = (1,0,0), scale = (0.5, 0.5, 0.5), rotate=(0,0,0), eye, up, center, texture=None):
    utils.load('cat.obj','cat.mtl', V3(0.27,-0.1,0), V3(0.1,0.1,0.1),V3(0,-pi/2,0),eye=V3(0,-0.2,1),up=V3(0,1,0),center=V3(0,0,0),texture=None)
    utils.glFinish("catLow")

    #utils.load('cat.obj','cat.mtl', V3(0.355,-0.259,0), V3(0.155,0.155,0.155),V3(pi/16,-pi/1.98,0),eye=V3(0,-0.2,1),up=V3(0,1,0),center=V3(0,0,0),texture=None)
    #utils.glFinish("monoMedium")
def High():
    utils=Render(1200,1200)
    utils.load('cat.obj','cat.mtl', V3(0.27,-0.1,0), V3(0.1,0.1,0.1),V3(0,-pi/2,0),eye=V3(0,0.4,1),up=V3(0,1,00),center=V3(0,0,0),texture=None)
    utils.glFinish("catHigh")

def Dutch():
    utils=Render(1200,1200)
    utils.load('cat.obj','cat.mtl', V3(-0.20,-0.1,0), V3(0.12,0.12,0.12),V3(0,-pi/4,pi/10),eye=V3(0,0.2,1),up=V3(0,1,0),center=V3(0,0,0),texture=None)
    utils.glFinish("catDutch")

def medium():
    utils=Render(1200,1200)
    utils.load('cat.obj','cat.mtl', V3(0.355,-0.259,0), V3(0.155,0.155,0.155),V3(pi/16,-pi/1.98,0),eye=V3(0,-0.2,1),up=V3(0,1,0),center=V3(0,0,0),texture=None)
    utils.glFinish("monoMedium")

menu()