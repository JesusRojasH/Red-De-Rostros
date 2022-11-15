# Red De Rostros
## Luis de Jesús Rojas Hernández

## Introducción. 
Este trabajo tiene como propósito presentar el desarrollo de una red neuronal artificial enfocada a detectar rostros, específicamente para la detección personal. La red se basará en el lenguaje de programación Python con ayuda principal de las librerías de Keras y Tensorflow, así mismo se hará uso de otras librerías con funciones complementarias. 

## Datos Generados y Descargados.
Las imágenes que se usaron para el entrenamiento y prueba de la red se obtuvieron de la base de datos Biggest gender/face recognition dataset. de la página Kaggle, que cuenta con alrededor de 27000 imágenes de las cuales 17000 son de hombres y 9400 de mujeres. Ya que la cantidad de archivos es bastante grande, se necesitó una cantidad considerable de fotos personales para poder entrenar la red y facilitar el reconocimiento, para ello, se utilizó el método de ImageDataGenerator de Tensorflow, que sirve para generar varias imágenes partiendo de una imagen base y haciendo ligeras modificaciones como el ángulo de la imagen, aumentar o disminuir el brillo, cortar una fracción de la imagen, etc. Una vez realizada la generación de imágenes, se llegaron a obtener 728 archivos modificados de 73 elementos originales, es decir fotos personales tomadas desde diferentes ángulos, vestimenta e iluminación.

En el código se hace una reducción de de las dimensiones de las imágenes a 150 píxeles por cada lado y para evitar la deformación de a la hora de reducirlas, se tomaron con la cámara de un teléfono y su configuración para tomarlas con la calidad más baja posible.

## Carga de Datos y Estructura de la Red

El código de la red y de la carga de datos se realizan en un único script de Python haciendo uso de diversas librerías, principalmente de Keras, TensorFlow y MatplotLib.
El script comienza con el procesamiento de las imágenes y las etiquetas que se añadirán a cada imagen, en este caso la única etiqueta que tendrán será para la identificación personal, siendo 0 las etiquetas correspondientes a mis fotos personales y 1 a las fotos de otras personas, dichas etiquetas se colocan en un DataFrame de pandas. Cómo las imágenes se encuentran en directorios diferentes al llevarlos a un solo DataFrame, serán concatenadas en orden, es decir, tendremos ambos conjuntos de imágenes separados, para evitar este inconveniente, modificaremos el DataFrame intercambiando sus filas aleatoriamente, de forma que queden intercaladas las imágenes junto con sus etiquetas.

Después de manipular el DataFrame, se utilizó TensotFlow para convertirlo en un Dataset y se creo la función 'PImages' para procesar las imágenes, se les redujo aún más las dimensiones para poder entrenar de manera más rápida (pasaron de ser cuadradas de 150px por lado a 64px por lado, manteniendo los 3 canales RGB) ya que al trabajar con las casi 28000 imágenes, el tiempo de cómputo por época era de alrededor de 18min, con la reducción de imágenes se llegó a disminuir hasta 6min. por época. Se utilizó el método 'map' para iterar la función sobre todos los elementos del Dataset.

Posteriormente se dividieron los datos de entrenamiento (80% de los datos) y de prueba (20% de los datos),  para el ajuste y evaluación del modelo, sin embargo, los datos de prueba se pasaron a un arreglo de numpy ya que el parámetro de validation_split marcaba un error pasando los datos como un objeto map.

Una vez los datos están listos, se define el modelo de la red neuronal, empezando con una capa de entrada de dimensiones $64px \times 64px \times 3ch $, el modelo se basa en una red residual, definimos la función 'block' donde, cómo su nombre lo dice, contiene un bloque de 3 capas obligatorias y 2 capas condicionales correspondientes a la ejecución de 'MaxPooling' (todas con una función de activación 'relu'), este bloque se ejecutará 4 veces, aumentando el número de filtros, aplicando un 'MaxPooling' en todas las ejecuciones exceptuando la última, añadimos una capa 'GlobalAveragePooling' junto con un 'Dropout' del 30% de las conexiones, la capa de salida será una densa de 1 neurona, pues solo deseo saber si la persona de la foto soy yo o no.

Por último se compilo el modelo y se utilizaron X_Test y Y_Test como datos de validación para poder graficar la función de costo y la precisión de la red. Se trató de entrenar con varias configuraciones de los parámetros de la red y diferentes tamaños de las imágenes buscando que el tiempo de computo fuera el mínimo, la función de costo mínima y la precisión la máxima posible, esta prueba en particular fue realizada con un tamaño de mini-batch =10 y  epoch = 50, la cantidad de parámetros de la red neuronal fueron alrededor de 158mil.

## Resultados, Modificaciones y Conclusiones.
El entrenamiento de la red con la configuración antes mencionada comenzaba a tener buenos resultados, llegando a un 97% de precisión durante la primera época y la función de costo disminuía considerablemente, sin embargo a partir de la época 15, la función de costo aumentaba drásticamente, aunque la precisión de la red no disminuía, llegando a tener un 99% de precisión con los datos de entrenamiento y con los datos de validación.

Es por eso que decidí disminuir el número de épocas de la red, para evitar ese aumento en la función de costo, además, consideré aumentar las dimensiones de las imágenes, pues con una imagen de 64 píxeles por lado creo que no es suficiente para abstraer detalles más minuciosos del rostro, lamentablemente por la falta de tiempo ya que me di cuenta de este factor demasiado tarde (por mi propia negligencia) tuve que dejar esa dimensión de imágenes, pues como mencione anteriormente, el tiempo de computo cuando se aumentan las dimensiones es considerable. Al volver a entrenar la red, ahora con 15 épocas, los resultados parecen ser mejor, considerando que la función de costo de ambos conjuntos de datos no aumenta drásticamente. Creo que un mejor resultado se obtendría aumentando las dimensiones de las imágenes a 100px por lado, así la red podría abstraer más detalles y ser mas robusta a la hora de determinar si una foto corresponde a mi persona o a la de alguien más, también tengo cierta incertidumbre con la precisión de la red, me parece que es bastante alta incluso en las primeras épocas, trate de averiguar si modificar la cantidad de imágenes de entrenamiento afectaba la precisión, y si, pero no de forma considerable, tratando de entrenar con 5000 imágenes, la primera época tuvo una precisión del 89\% que es la que más se diferenciaba, pero en la segunda época aumentaba al 95% y así progresivamente hasta llegar al 99%, desconozco si sea por la forma en que las imágenes son introducidas a la red.

### Actualizción 1.
Después de entrenar la red neuronal ahora con una configuración de 50 épocas y cambiar las dimensiones de las imágenes a $100px \times 100px$, podemos observar que la precisión de la red es bastante buena con los datos de entrenamiento, y con los datos de validación también se obtiene un buen resultado, a pesar de lo que se puede apreciar en la gráfica dicha fluctuación no baja del 95% de precisión. Con la función Loss de los datos de entrenamiento se puede apreciar que alcanza un mínimo cerca de las 15 épocas, igual que con la configuración de imágenes de 64px y no sube de forma tan drástica, cuyos valores rondaban entre 1 y 3 durante la época 20, sin embargo, la función Loss en los datos de validación fluctua demasiado, hasta en las últimas épocas incrementa asintóticamente. 
