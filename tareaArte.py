# Universidad del Valle de Guatemala
# Francisco Molina Jimenez - 17050
# recordar que [Y][X}

import struct
from random import randint
from random import uniform
from math import sqrt
from math import cos
from math import sin
from math import pi
from math import floor
import sys
from objReader import *
from collections import namedtuple

sys.setrecursionlimit(4200)

#structs...
V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])
# 1 bit
def char(c):
    return struct.pack("=c", c.encode('ascii'))

# 2 bit
def word(c):
    return struct.pack("=h", c)

# 4 bit
def dword(c):
    return struct.pack("=l", c)

# color
def colorArt(r, g, b):
    return bytes([b, g, r])

def sum(v0, v1):
    return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
    return V3(v0.x * k, v0.y * k, v0.z * k)

def dot(v0, v1):
    return (v0.x * v1.x + v0.y * v1.y + v0.z * v1.z)

def cross(v0, v1):
    return V3(
        v0.y *  v1.z - v0.z * v1.y,
        v0.z * v1.x - v0.x * v1.z,
        v0.x * v1.y - v0.y * v1.x
    )

def length(v0):
    return (v0.x**2 + v0.y **2 + v0.z **2)**0.5

def norm(v0):
    l = length(v0)

    if not l:
        return V3(0, 0, 0)

    return V3(v0.x / l, v0.y / l, v0.z / l)

def bbox(A, B, C):
    xs = sorted([int(A.x), int(B.x), int(C.x)])
    ys = sorted([int(A.y), int(B.y), int(C.y)])

    return V2(int(xs[0]), int(ys[0])), V2(int(xs[2]), int(ys[2]))

def barycentric(A,B,C,P):
        cx, cy, cz = cross(
                V3(B.x - A.x, C.x - A.x, A.x - P.x),
                V3(B.y - A.y, C.y - A.y, A.y - P.y)
        )

        #[cx/cz, cy/cz, cz7cz] = [u,v,1]

        if cz <1:    
            return -1,-1,-1
        
        u = cx/cz
        v = cy/cz
        w = 1 - (u + v)

        return w,v,u

class Render(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.framebuffer = []
        #self.glClear()
        self.color = color(255, 255, 255)
        self.vColor = color(255, 255, 255)
        self.glCreateWindow()
        self.glClear()
        self.zbuffer = [
      [-float('inf') for x in range(self.width)]
      for y in range(self.height)
    ]

    def genColor(self,r,g,b):
        return bytes([b,g,r])

    def write(self, filename):
        f = open(filename, 'wb')
        # header 14
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        # image header 40
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))
        f.write(dword(0))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0))

        for x in range(self.height):
            for y in range(self.width):
                f.write(self.framebuffer[x][y])
        f.close()

    # Crea el bmp.
    def glCreateWindow(self):
        self.framebuffer = [[0 for x in range(self.height)] for y in range(self.width)]

    # Llenar el cuadro de un color.
    def glClear(self):
        self.framebuffer = [[color(0, 0, 0) for x in range(self.width)] for y in range(self.height)]

        self.zbuffer=[
            [-float('inf') for x in range(self.width+1)]
            for y in range(self.height+1)
        ]

    # Llenar cuadro del color que desee.
    def glClearColor(self, r, g, b):
        self.framebuffer = [[color(int(r * 255), int(g * 255), int(b * 255)) for x in range(self.width)] for y in
                            range(self.height)]

    # crea un viewport NOTA: NO DEBE EXCEDER RESOLUCION DEL BITMAP.
    def glViewport(self, x, y, widthv, heightv):
        self.x = x
        self.y = y
        self.ancho = widthv
        self.alto = heightv

    def completeViewPort(self):
        self.glViewport(0,0,self.width,self.height)

    # x,y son valores entre 0 y 1
    def glVertex(self, x, y):
        vx = int((x + 1) * (self.ancho / 2)) + self.x
        vy = int((y + 1) * (self.alto / 2)) + self.y
        try:
            if x == 1 and y == 1:
                print("Restando offset para param 1 doble")
                vx=vx-1
                vy=vy-1
                # solucion chafa...
        except IndexError:
            print("Las dimensiones del viewport deben ser menores a las del bmp.")
            print("Saliendo de programa.....")
        return vx,vy
    
    def glVertexInv(self, vx,vy):
        self.glViewport(0,0,self.width,self.height)
        x=(((vx-self.x)*2)/self.ancho) -1
        y=(((vy-self.y)*2)/self.alto) -1
        return x, y


    # cambia color al vertex
    def glColor(self, r, g, b):
        self.vColor = color(r, g, b)

    # FUNCIONALIDAD DE PROGRAMA
    def point(self, x, y, color):
        self.framebuffer[y][x] = color

   #in: valores normalizados
   #out: equivalentes en las coordenadas sin normalizar...  
    def glPoint(self, x,y,color=(0,0,0)):
        self.completeViewPort()
        while x>1 or x<-1 or y>1 or y<-1:
            print("Este metodo solo permite valores entre -1 y 1")
            x=input("Ingrese valor para X")
            y=input("Ingrese valor para Y")
        x,y=self.glVertex(x,y)
        self.framebuffer[y][x]=color


    def glLine(self, x0,x1,y0,y1):
        self.completeViewPort()
        while x0>1 or x1>1 or y0>1 or y0>1:
            print("Este metodo solo permite valores entre -1 y 1")
            x0=input("Ingrese valor para X inicial")
            x1=input("Ingrese valor para X final")
            y0=input("Ingrese valor para y inicial")
            y1=input("Ingrese valor parra y final")

        x0,y0=self.glVertex(x0,y0)
        x1,y1=self.glVertex(x1,y1)
        #calculo para pendiente.
        difY = abs(y1 - y0)
        difX = abs(x1 - x0)


        #si el cambio en y es mayor que en x.
        steep = difY >= difX

        #asignamos un cambio de variables
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            difY = abs(y1 -y0)
            difX = abs(x1 -x0)

        # si el inicial es mayor al final, reasigna.
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        # calcula de nuevo la dif
        m = difY

        # y = y0 - m * (x0 - x)
        offset = 0 * 2 * difX
        #hasta cuando deberia poner otro pixel
        threshold = 0.5 * difX

        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                self.point(y, x, color(255,255,255))
            else:
                self.point(x, y, color(255,255,255))
            offset += m
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += 1 * difX



    def line(self, x0, x1, y0, y1):
       #calculo para pendiente.
        difY = abs(y1 - y0)
        difX = abs(x1 - x0)


        #si el cambio en y es mayor que en x.
        steep = difY >= difX

        #asignamos un cambio de variables
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            difY = abs(y1 -y0)
            difX = abs(x1 -x0)

        # si el inicial es mayor al final, reasigna.
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        # calcula de nuevo la dif
        m = difY

        # y = y0 - m * (x0 - x)
        offset = 0 * 2 * difX
        #hasta cuando deberia poner otro pixel
        threshold = 0.5 * difX

        y = y0
        for x in range(x0, x1 + 1):
            if steep:
                self.point(y, x, color(255,255,255))
            else:
                self.point(x, y, color(255,255,255))
            offset += m
            if offset >= threshold:
                y += 1 if y0 < y1 else -1
                threshold += 1 * difX


    def glFinish(self, txt):
        self.write(txt + ".bmp")

    
    def triangle(self, A, B, C, color=None,texture=None, texture_coords=(), intensity=1):
        #print(A.x,A.y)
        #print(B.x,B.y)
        bbox_min, bbox_max = bbox(A,B,C)
        '''if bbox_min.x<0:
            bbox_min=V2(0,bbox_min.y)
        if bbox_min.y<0:
            bbox_min=V2(bbox_min.x,0)
        if bbox_max.x<0:
            bbox_max=V2(0,bbox_max.y)
        if bbox_max.y<0:
            bbox_max=V2(bbox_max.x,0)
            
        if bbox_min.x>=self.width:
            bbox_min=V2(self.width-1,bbox_min.y)
        if bbox_min.y>=self.height:
            bbox_min=V2(bbox_min.x,self.height-1)
        if bbox_max.x>=self.width:
            bbox_max=V2(self.width-1,bbox_max.y)
        if bbox_max.y>=self.height:
            bbox_max=V2(bbox_max.x,self.height-1)'''
        
        #print(bbox_min,bbox_max)
        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = barycentric(A, B, C, V2(x, y))
                if w < 0 or v < 0 or u < 0:
                    continue
                # si encontramos textura... si no, continua procedimiento normal..
                if texture:
                    #aqui es donde se ponen las texturas sobre elrender..
                    tA, tB, tC = texture_coords
                    tx = tA.x * w + tB.x * v + tC.x * u
                    ty = tA.y * w + tB.y * v + tC.y * u
          
                    color = texture.get_color(tx, ty, intensity)
        
                z = A.z * w + B.z * v + C.z * u
                if x < self.width and y<self.height and x >=0 and y>=0:
                    if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
                        self.point(x, y, color)
                        self.zbuffer[x][y] = z


    #escala y mueve, basicamente...
    def transform(self, vertex, translate=(0,0,0), scale=(1,1,1), rotate=(0,0,0)):
        augmented_vertex = [
            [vertex.x],
            [vertex.y],
            [vertex.z],
            [1]
        ]


        #la multiplicacion de matrices va de afuera para adentro
        #vertices = self.Viewport @ self.Projection @ self.View @ self.Model @ augmented_vertex
        modelViewMul=self.matrix_multiply(self.Viewport,self.Projection)
        viewMul=self.matrix_multiply(modelViewMul, self.View)
        viewPortMul=self.matrix_multiply(viewMul, self.Model)
        vertices = self.matrix_multiply(viewPortMul, augmented_vertex)
        #vertices = vertices.tolist()[0]
        transformed_vertex=[
            round((vertices[0][0])/vertices[3][0]),
            round((vertices[1][0])/vertices[3][0]),
            round((vertices[2][0])/vertices[3][0])
        ]
        return V3(*transformed_vertex)
    #pipeline
        #transformed_vertex = self.Viewport @ selfProjection @ self.View @ self.Model @ augmented_vertex
    
    def debug(self, filename):
        model = objReader(filename)
        for elem in model.textVert:
            print(elem)
    
    
    #Loader de obj, llama a la clase que procesa.... 
    def load(self, filename, mtlFilename=None, translate = (1,0,0), scale = (0.5, 0.5, 0.5), rotate=(0,0,0), eye=V3(0,0,0), up=V3(0,0,0), center=V3(0,0,0), texture=None):
        
        model = objReader(filename)
        self.loadModelMatrix(translate, scale, rotate)
        self.lookAt(eye, up, center)
        #self.lookAt(V3(0,-0.2,1),V3(0,1,0),V3(0,0,0))
        self.loadViewportMatrix()
        if texture==None:
            pass
        else:
            modelT = Texture(texture)
        if mtlFilename:
            mtlInstance= objMtl(mtlFilename)
            RValue = mtlInstance.material[0][0]
            GValue = mtlInstance.material[0][1]
            BValue = mtlInstance.material[0][2]
        else:
            RValue = color(255,255,255)
            GValue = color(255,255,255)
            BValue = color(255,255,255)
        light = V3(0,0,1)

        for face in model.vfaces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                
                a = self.transform(V3(*model.vertices[f1]))
                b = self.transform(V3(*model.vertices[f2]))
                c = self.transform(V3(*model.vertices[f3]))

                #Para proposito de debug...
                '''
                Se tiene problema que no detecta los parametros de triangle_Sweep(a,b,c) como structs(??)
                como cosa rara, el print de abajo demuestra que SI es una tupla.

                **resuelto**
                '''

               # Si no hay textura, procedemos a un flatshading con mtl...
                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)
                if not texture:
                    #aqui bajamos intensidad de la tonalidad asignada al render
                    r = round(255*RValue * intensity)
                    g = round(255*GValue * intensity)
                    blue = round(255*BValue * intensity)
                    if r < 0 or g<0 or blue<0:
                        continue  
                    self.triangle(a, b, c, color(r, g, blue))
                #Si hay material...
                else:
                    t1 = face[0][1] - 1
                    t2 = face[1][1] - 1
                    t3 = face[2][1] - 1
                    
                    #rompemos array y pasamos values
                    tA = V3(*model.textVert[t1],0)
                    tB = V3(*model.textVert[t2],0)
                    tC = V3(*model.textVert[t3],0)

                    #rendeer...
                    self.triangle(tA,tB,tC,color(),None, texture=modelT, texture_coords=(tA, tB, tC), intensity=intensity)

    
#el rotate tiene los angulos medidos en radianes
    def loadModelMatrix(self, translate, scale, rotate):
        translate = V3(*translate)
        rotate = V3(*rotate)
        scale = V3(*scale)
        #declaracion de matriz...
        translate_matrix = [[0 for x in range(4)] for y in range(4)]
        '''
        translate_matrix = numpy.matrix([
            [1,0,0,translate.x],
            [0,1,0,translate.y],
            [0,0,1,translate.z],
            [0,0,0,1],
        ])
        '''
        
        translate_matrix=[
        [1,0,0,translate.x],
        [0,1,0,translate.y],
        [0,0,1,translate.z],
        [0,0,0,1]]
        #declaracion de matriz
        scale_matrix=[
        [scale.x,0,0,0],
        [0,scale.y,0,0],
        [0,0,scale.z,0],
        [0,0,0,1]]
       
        #declaracion de matriz
        rotation_matrix_x = [
            [1,0,0,0],
            [0,cos(rotate.x),-sin(rotate.x),0],
            [0,sin(rotate.x), cos(rotate.x),0],
            [0,0,0,1]
        ]
        
        #declaracion de matriz
        rotation_matrix_y =[
            [cos(rotate.y),0,sin(rotate.y),0],
            [0,1,0,0],
            [-sin(rotate.y),0, cos(rotate.y),0],
            [0,0,0,1]
        ]

        #declaracion de matriz
        rotation_matrix_z = [
            [cos(rotate.z),-sin(rotate.z),0,0],
            [sin(rotate.z), cos(rotate.z),0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]

        #rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z

        #rotation_matrix = self.matrix_multiply(rotation_matrix_x,self.matrix_multiply(rotation_matrix_z,rotation_matrix_y))
        '''rotation_matrix = self.matrix_multiply(rotation_matrix_z, rotation_matrix_y)
        rotation_matrixFinal = self.matrix_multiply(rotation_matrix, rotation_matrix_x)
        #self.Model = translate_matrix @ rotation_matrix @ scale_matrix
        rotMul=self.matrix_multiply(rotation_matrixFinal, translate_matrix)
        transMul=self.matrix_multiply(rotMul, scale_matrix)
        self.Model = transMul'''
        rotation_matrix = self.matrix_multiply(self.matrix_multiply(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
        self.Model = self.matrix_multiply(self.matrix_multiply(translate_matrix, rotation_matrix), scale_matrix)

    '''
    eye: Desde donde miramos 
    center: punto de observacion
    up: da el norte?
    '''
    def lookAt(self, eye, up, center):
        #z es el vector mas facil de obtener, es el vector que va del centro al ojo
        z = norm(sub(eye,center))
        x = norm(cross(up,z))
        y = norm(cross(z,x))

        self.loadViewMatrix(x,y,z, center)
        cc = norm(sub(eye,center))
        self.loadProyectionMatrix(-1/length(sub(eye,center)))
        
    def loadViewportMatrix(self, x=0, y=0):
        self.Viewport = [
            [self.width/2,0,0,x + self.width/2],
            [0,self.height/2,0,y+ self.height/2],
            [0,0,128,128],
            [0,0,0,1]
        ]

    def loadProyectionMatrix(self, coeff):
        
        
        self.Projection =[
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,coeff,1]
        ]

    def loadViewMatrix(self, x, y, z, center):
        
        M =[
            [x.x, x.y, x.z,0],
            [y.x, y.y, y.z,0],
            [z.x, z.y, z.z,0],
            [0,0,0,1]
        ]
        
        
        O = [
            [1,0,0,-center.x],
            [0,1,0,-center.y],
            [0,0,1,-center.z],
            [0,0,0,1]
        ]
        #self.View = M @ O
        self.View = self.matrix_multiply(M,O)
    
    
    def zeros_matrix(self,rows, cols):
        A = []
        for i in range(rows):
            A.append([])
            for j in range(cols):
                A[-1].append(0.0)

        return A

    '''
    IN: Dos matrices de dimensiones distintas, donde col = row  
    OUT: res: mat1 x mat2 
    '''
    def matrix_multiply(self,A,B):
        result=self.zeros_matrix(len(A),len(B[0]))
        for i in range(len(A)):
            # iterate through columns of Y
            for j in range(len(B[0])):
            # iterate through rows of Y
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
