

import os
import cv2  
import numpy as np 
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

NewDir = 'ImagenesGen'
img_incre = 8 #Número de imagenes que obtendremos de una imagen original

try:
    os.mkdir(NewDir)
except:
    pass

Img_Gen = ImageDataGenerator(rotation_range= 10, #Rotación de 10 grados de la imagen
        zoom_range=.1, #Hacer un zoom del 10% de la imagen 
        width_shift_range =0.05, #Desplazmiento horizontal de la imagen en un 10%
        height_shift_range = 0.05, #Desplazmiento horizontal de la imagen en un 10%
        horizontal_flip= True, #Inversión horizontal de la imagen (efecto espejo)
        vertical_flip = False) #Inversión vertical de la imagen (poner de cabeza)

DP_Path = 'D:/Users/Jesús Rojas/Documents/RedesNeuronales/Red de Rostros/EVRR/archive/FPersonalesMod'
DP_Path_D = os.listdir(DP_Path) 

W_Img, H_Img = 100, 150
i = 0
num_img = 0

for img in DP_Path_D:
    Img_list = os.listdir(DP_Path)
    Img_Path = DP_Path + '/' + img 
    LImg = load_img(Img_Path)
    LImg = cv2.resize(img_to_array(LImg), (W_Img, H_Img), interpolation = cv2.INTER_AREA)
    X = LImg/255
    X = np.expand_dims(X, axis = 0)
    t = 1
    for OI in Img_Gen.flow(X, batch_size = 1):
        A = img_to_array(OI[0])
        Imagen = OI[0,:,:]*255
        FI = cv2.cvtColor(Imagen, cv2.COLOR_BGR2RGB)
        cv2.imwrite(NewDir + "/%i%i.jpg"%(i,t), FI)
        t += 1
        num_img += 1
        
        if t > img_incre:
            break
    i +=1
 
print('No de Imagenes Generadas', num_img)
