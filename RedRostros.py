
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import keras
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
from keras.optimizers import RMSprop
import pandas as pd 

epocas = 15 # Número de epocas
b_s = 10 #Tamaño del minibatch  

DirectMW = "D:/Users/Jesús Rojas/Documents/RedesNeuronales/Red de Rostros/EVRR/archive/TrainData/"
DirectP = "D:/Users/Jesús Rojas/Documents/RedesNeuronales/Red de Rostros/EVRR/archive/ImagenesGen/"

LWM = os.listdir(DirectMW)
LP = os.listdir(DirectP)

dataP = pd.DataFrame(LP)
dataMW = pd.DataFrame(LWM)

dataP['Persona'] = 0 #Etiqueta 0 para fotos personales
dataMW['Persona'] = 1 #Etiqueta 1 para fotos de otras personas

Frame = [dataP, dataMW]
dataT = pd.concat(Frame)

#Barajear los datos del dataframe 
dataT = dataT.iloc[np.random.permutation(len(dataT))] 


files = tf.data.Dataset.from_tensor_slices(dataT[0])
attr = tf.data.Dataset.from_tensor_slices(dataT.iloc[:,1].to_numpy())
Data = tf.data.Dataset.zip((files, attr))


"""Procesando imagenes con tensorflow"""
def PImages(file_name, attrs):
    if attrs == 0:
        Img = tf.io.read_file(DirectP + file_name)
    else:
        Img = tf.io.read_file(DirectMW + file_name)

    Img = tf.image.decode_jpeg(Img, channels = 3)
    Img = tf.image.resize(Img, [64, 64])
    Img /= 255.
    return Img, attrs


Eti_Imgs = Data.map(PImages)

#Datos de prueba y datos de entrenamiento 
Len_Data = len(LWM)+len(LP)
Len_TrainD = round(Len_Data*0.8) #80% de los datos destinados a entrenar
#El resto de los datos para test

Train_D = Eti_Imgs.take(Len_TrainD).batch(b_s)

X_Test =[] 
Y_Test = [] 


for img, attr in Eti_Imgs.skip(Len_TrainD):
    X_Test.append(img.numpy())
    Y_Test.append(attr.numpy())

X_Test = np.asarray(X_Test)
Y_Test = np.asarray(Y_Test)

#Modelo 
Entradas= keras.layers.Input(shape = (64, 64, 3))

def block(X, No_Filtros = 32, pooling = True):
    Res = X 
    X = keras.layers.Conv2D( No_Filtros, 3, activation = 'relu', padding = 'same')(X)
    X = keras.layers.Conv2D(2*No_Filtros, 3, activation = 'relu', padding = 'same')(X) 

    if (pooling == 'True'):
        X = keras.layers.MaxPooling2D(2, padding= 'same')(X)
        Res = keras.layers.Conv2D(No_Filtros, 1, strides = 2)(Res)
    else:
        Res = keras.layers.Conv2D(2*No_Filtros, 1)(Res)
    X = keras.layers.add([X, Res])
    return X

X = block(Entradas, No_Filtros = 8)
X = block(X, No_Filtros = 16)
X = block(X, No_Filtros = 32)
X = block(X, No_Filtros = 64, pooling = False)

X = keras.layers.GlobalAveragePooling2D()(X)
X = keras.layers.Dropout(0.2)(X)
Salida = keras.layers.Dense(1, activation = 'sigmoid')(X)

Model = keras.Model(inputs = Entradas, outputs = Salida)
Model.summary()

Model.compile(loss = 'binary_crossentropy', optimizer = 'RMSprop',
            metrics = ['accuracy'])

Model.fit(Train_D, batch_size = b_s, verbose = 1, validation_data =(X_Test, Y_Test), epochs=epocas, shuffle=True )

acc = Model.history.history['accuracy']
val_acc = Model.history.history['val_accuracy']

plt.plot(acc, label ='Datos de Entrenamiento')
plt.plot(val_acc, label ='Datos de Validación')
plt.title('Precisión de la Red')
plt.ylabel('Precisión')
plt.xlabel('Épocas')
plt.show()

loss = Model.history.history['loss']
val_loss = Model.history.history['val_loss']

plt.plot(loss, label ='Datos de Entrenamiento')
plt.plot(val_loss, label ='Datos de Validación')
plt.title('Precisión de la Red')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.show()

Model.save('Modelos/Prueba2RedRostros.hf5')

#Para evaluar la red. 
#score = Model.evaluate(X_Test,Y_Test, verbose = 2)
#print(score)
