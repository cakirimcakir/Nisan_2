import cv2
import numpy as np

foto=cv2.imread("ornek.jpg") #fotoğrafı yüklüyoruz

gri_foto=cv2.cvtColor(foto,cv2.COLOR_BGR2GRAY)
cv2.imwrite("gri_foto.jpg",gri_foto)

hsv_foto=cv2.cvtColor(foto, cv2.COLOR_BGR2HSV)
cv2.imwrite("hsv_gorsel.jpg",hsv_foto)

b,g,r=cv2.split(foto)
swapped_foto=cv2.merge([r,g,b])
cv2.imwrite("kanal_degistirilmis.jpg",swapped_foto)

outpot_foto=foto.copy()
font=cv2.FONT_ITALIC
cv2.putText(outpot_foto, "Ali Emir Cakir",(10,foto.shape[0]-10),font,1,(255,255,255),2,cv2.LINE_AA)

cv2.imwrite("isim_foto.jpg",outpot_foto)


