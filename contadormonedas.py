import cv2
import numpy as np

valorGauss=3
valorKernel=3
path = 'monedas.jpg'
#Imagen original
original = cv2.imread(path)
#Escala Gris
gris= cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#Desenfoque
gauss = cv2.GaussianBlur(gris,(valorGauss,valorKernel), 0)
canny=cv2.Canny(gauss,60,100)
#Kernel para el cierre
kernel = np.ones((valorKernel,valorKernel),np.uint8)
#Cierre
cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
#obtenemos contornos
contornos, jerarquia = cv2.findContours(cierre.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#Cantidad de objetos encontrados
print("monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos,-1,(0,0,255),3)

#Mostrar Proceso
# cv2.imshow("Original", original)
# cv2.imshow("Grises", gris)
# cv2.imshow("Gauss", gauss)
# cv2.imshow("Canny", canny)
# cv2.imshow("Cierre", cierre)

#Resultado Final
cv2.imshow("Final", original)

cv2.waitKey(0)