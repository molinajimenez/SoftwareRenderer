import struct

def color(r,g,b):
    return bytes([b,g,r])

class objReader(object):
	def __init__(self, filename):
		with open(filename) as f:
			self.lines=f.read().splitlines()

		self.vertices = []
		self.faces = []
		self.vfaces = []
		self.textVert=[]
		self.read()

	
	def read(self):
		for line in self.lines:
			if line:
				prefix, value = line.split(" ",1)
				#print(prefix,value)
				if prefix == 'v':
					#append de una lista que se agrega por cada espacio, cadaa valor de la lista se traduce automaticamente a un valor float :)
					self.vertices.append(list(map(float, value.split(" "))))
				elif prefix == 'vt':
					self.textVert.append(list(map(float, value.split(" "))))
				elif prefix == 'f': 
					#append de una lista que se agrega por cada espacio, cadaa valor de la lista se traduce automaticamente a un valor int :)
					self.vfaces.append([list(map(int, face.split('/'))) for face in value.split(' ')])

class objMtl(object):
	def __init__(self,filename):
		with open(filename) as f:
			self.lines=f.read().splitlines()

		self.material = []
		self.read()

	def read(self):
		for line in self.lines:
			if line:
				prefix, value = line.split(" ", 1)
				if prefix == 'Kd':
					#valores RGB se obtienen leyendo 
					self.material.append(list(map(float, value.split(" "))))


class Texture(object):
	def __init__(self, loc):
		# mandamos el relative path del .bmp 
		self.loc = loc
		self.read()
    
	# mismo read anterior, deberiamos usar herencia pero que hueva jajajaj... quizas luego.
	def read(self):
		textureFile = open(self.loc, "rb")
		textureFile.seek(2 + 4 + 4)
		header_size = struct.unpack("=l", textureFile.read(4))[0]
		textureFile.seek(2 + 4 + 4 + 4 + 4)
		self.width = struct.unpack("=l", textureFile.read(4))[0]
		self.height = struct.unpack("=l", textureFile.read(4))[0]
		self.txtColor = []
		textureFile.seek(header_size)

		#agregamos los colores encontrados
		for y in range(self.height):
			self.txtColor.append([])
			for x in range (self.width):
				b = ord(textureFile.read(1))
				g = ord(textureFile.read(1)) 
				r = ord(textureFile.read(1)) 
				self.txtColor[y].append(color(r, g, b))
        #terminan ops con el bmp...
		textureFile.close()

	def get_color(self, tx, ty, intensity=1):
		x = int(tx * self.width)
		y = int(ty * self.height)
		#same shit, si es 0 porque lo agregariamos(?), se crea lambda porque
		#supongo que dennis le da hueva crear nueva funcion, anyways ahi esta.. jaja
		return bytes(map(lambda b: round(b*intensity)if b*intensity>0 else 0, self.txtColor[y][x]))