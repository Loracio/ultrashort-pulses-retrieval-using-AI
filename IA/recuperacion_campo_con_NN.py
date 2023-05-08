"""
En este script probaremos a diseñar una NN que intentará recuperar los valores
del campo eléctrico directamente a partir de los valores de su traza.

Es decir, como input a la red neuronal tendremos el vector de la traza, y como
salida la parte real e imaginaria del campo eléctrico.

Este script simplemente contiene la arquitectura de la red y su entrenamiento,
para ver los resultados usar el archivo 'visualizacion_campo_NN.py'
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD

import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *

from lee_database import formateador


if __name__ == '__main__':
    # Parámetros lectura
    N = 128
    NUMERO_PULSOS = 1000

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"
 
    # Cargamos los datos como np.ararys. Formateador devuelve TBP, Traza, Campo
    TBP, y, x = formateador(direccion_archivo, N, NUMERO_PULSOS, shuffle=True, buffer_size=500)

    # Normalizamos las trazas:
    for i in range(NUMERO_PULSOS):
        x[i] /= np.max(x)
        y[i] /= np.max(y)
 
    # Separamos entre conjunto de entrenamiento y validación
    tamaño_entrenamiento = int(0.8 * NUMERO_PULSOS)
    tamaño_validacion = NUMERO_PULSOS - tamaño_entrenamiento

    print(f"Tamaño del conjunto de entrenamiento: {tamaño_entrenamiento}")
    print(f"Tamaño del conjunto de validación: {tamaño_validacion}")


    # Parámetros de la red
    EPOCHS = 100
    BATCH_SIZE = 32

    """
    Vamos a hacer una red muy simple, con una única capa densa.
    Como entrada tendrá la traza del pulso y como salida tendrá
    la parte real e imaginaria del campo, una detrás de la otra.
    """
    input_shape = ((2*N -1)*N,)
    hidden_layer_neurons = N
    output_neurons = 2 * N

    # Construcción de la arquitectura
    input_tensor = Input(shape=input_shape ,name='input')
    hidden_tensor = Dense(hidden_layer_neurons, activation='relu', name='hidden')(input_tensor)
    hidden_tensor_1 = Dense(int(hidden_layer_neurons/2), activation='relu', name='hidden1')(hidden_tensor)
    hidden_tensor_2 = Dense(hidden_layer_neurons, activation='relu', name='hidden2')(hidden_tensor_1)
    output_tensor = Dense(output_neurons, name='output')(hidden_tensor_2)   
    model_dense = Model(input_tensor, output_tensor)

    opt = Adam()
    # Compilación del modelo
    model_dense.compile(loss="mse", optimizer=opt, metrics=['mse'])

    # Mostramos cómo es
    model_dense.summary()

    # Entrenamiento
    history = model_dense.fit(x[:tamaño_entrenamiento], y[:tamaño_entrenamiento], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x[tamaño_validacion:], y[tamaño_validacion:]))

    # Guardamos el modelo entrenado para ver los resultados
    model_dense.save("./IA/NN_models/campo_model_simple_dense.h5")


    """
    Vamos a hacer un segundo modelo, en este caso utilizando capas
    convolucionales para ver si funciona mejor que las capas densas.
    
    En este caso, al tener de input la traza, tenemos una capa de input
    con muchas neuronas, por lo que será computacionalmente más costoso
    el hacer el entrenamiento de la red. También tendremos un escalado
    bastante grande en el número de hiperparámetros.
    """

    input_channels=1
    def x_from_dense_to_conv2d(x,NUM_PULSOS, N, input_channels):
        return x.reshape((NUM_PULSOS, 2*N - 1, N, input_channels))

    x = x_from_dense_to_conv2d(x, NUMERO_PULSOS, N, input_channels)

    img_cols = x.shape[1]
    img_rows = x.shape[2]
    input_shape = (img_cols,img_rows,input_channels)
    hidden_layer_neurons = 32
    output_neurons = 2 * N

    # Construcción de la arquitectura
    input_tensor = Input(shape=input_shape ,name='input')
    hidden_conv = Conv2D(64, kernel_size=(3, 3),activation='relu')(input_tensor)
    hidden_maxpool = MaxPooling2D((2,2))(hidden_conv)
    hidden_conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(hidden_maxpool)
    hidden_maxpool1 = MaxPooling2D((2,2))(hidden_conv1)
    hidden_conv2 = Conv2D(16, kernel_size=(3, 3),activation='relu')(hidden_maxpool1)
    hidden_maxpool2 = MaxPooling2D((2,2))(hidden_conv2)
    hidden_flatten = Flatten()(hidden_maxpool2)
    hidden_tensor = Dense(hidden_layer_neurons, activation='relu',name='hidden')(hidden_flatten) 
    output_tensor = Dense(output_neurons, name='output')(hidden_tensor)               
    model_conv = Model(input_tensor,output_tensor) 

    # Compilación del modelo
    model_conv.compile(loss="mse", optimizer="adam", metrics=['mse'])

    # Mostramos cómo es
    model_conv.summary()

    # Entrenamiento
    history = model_conv.fit(x[:tamaño_entrenamiento], y[:tamaño_entrenamiento], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x[tamaño_validacion:], y[tamaño_validacion:]))

    # Guardamos el modelo entrenado para ver los resultados
    model_conv.save("./IA/NN_models/campo_model_convolucional.h5")
