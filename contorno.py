import cv2
path = 'contorno.jpg'

#Leer imagen
imagen=cv2.imread(path)
#Cambiar de color a grises
grays=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
#Obtener umbral
_,umbral= cv2.threshold(grays, 100, 255, cv2.THRESH_BINARY)
#Obtener contorno
contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#Dibujar contorno
cv2.drawContours(imagen, contorno,-1,(0,0,255),3)

#Mostrar imagen
cv2.imshow('Original', imagen)
cv2.imshow('Grises', grays)
cv2.imshow('Umbral', umbral)

cv2.waitKey(0)
cv2.destroyAllWindows()
