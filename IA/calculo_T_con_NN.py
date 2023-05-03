"""
En este script probaremos a diseñar una NN capaz de calcular la traza
de un pulso dados los valores de su campo eléctrico (partes real e imaginaria).

Este script simplemente contiene la arquitectura de la red y su entrenamiento,
para ver los resultados usar el archivo 'visualizacion_NN.py'
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout

import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *

from lee_database import formateador


if __name__ == '__main__':
    # Parámetros lectura
    N = 128
    NUMERO_PULSOS = 2000

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"
 
    # Cargamos los datos como np.ararys
    TBP, x, y = formateador(direccion_archivo, N, NUMERO_PULSOS, shuffle=True, buffer_size=500)
 
    # Separamos entre conjunto de entrenamiento y validación
    tamaño_entrenamiento = int(0.8 * NUMERO_PULSOS)
    tamaño_validacion = NUMERO_PULSOS - tamaño_entrenamiento

    print(f"Tamaño del conjunto de entrenamiento: {tamaño_entrenamiento}")
    print(f"Tamaño del conjunto de validación: {tamaño_validacion}")


    # Parámetros de la red
    EPOCHS = 25
    BATCH_SIZE = 32

    """
    Vamos a hacer una red muy simple, con una única capa densa.
    Como entradas tendrá el vector concatenado de las partes reales
    e imaginarias del pulso.
    La capa de salida nos dará el vector Tmn en versión 1D.
    """
    # input_shape = (2*N,)
    hidden_layer_neurons = 128
    output_neurons = (2*N - 1) * N

    # # Construcción de la arquitectura
    # input_tensor = Input(shape=input_shape ,name='input')
    # hidden_tensor = Dense(hidden_layer_neurons, activation='relu',name='hidden')(input_tensor)  
    # output_tensor = Dense(output_neurons,activation='relu',name='output')(hidden_tensor)               
    # model_dense = Model(input_tensor,output_tensor) 

    # # Compilación del modelo
    # model_dense.compile(loss="mse", optimizer="adam")

    # # Mostramos cómo es
    # model_dense.summary()

    # # Entrenamiento
    # history = model_dense.fit(x[:tamaño_entrenamiento], y[:tamaño_entrenamiento], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x[tamaño_validacion:], y[tamaño_validacion:]))

    # # Guardamos el modelo entrenado para ver los resultados
    # model_dense.save("./IA/NN_models/pulse_trace_model_simple_dense.h5")


    """
    Vamos a hacer un segundo modelo, en este caso utilizando capas
    convolucionales para ver si funciona mejor que las capas densas.
    Es esperable que consiga encontrar los patrones entre las partes
    real e imaginaria del pulso.
    """

    input_channels=1
    def x_from_dense_to_conv2d(x,NUM_PULSOS, N,input_channels):
        return x.reshape((NUM_PULSOS, N, 2, 1))

    x = x_from_dense_to_conv2d(x, NUMERO_PULSOS, N, input_channels)

    img_cols = x.shape[1]
    img_rows = x.shape[2]
    input_shape = (img_cols,img_rows,input_channels)

    # Construcción de la arquitectura
    input_tensor = Input(shape=input_shape ,name='input')
    hidden_conv = Conv2D(64, kernel_size=(2, 1),activation='relu')(input_tensor)
    hidden_maxpool = MaxPooling2D((1,1))(hidden_conv)
    hidden_conv_1 = Conv2D(32, kernel_size=(2, 1),activation='relu')(hidden_maxpool)
    hidden_maxpool_1 = MaxPooling2D((1,1))(hidden_conv_1)
    hidden_flatten = Flatten()(hidden_conv_1)
    hidden_tensor = Dense(hidden_layer_neurons, activation='relu',name='hidden')(hidden_flatten) 
    output_tensor = Dense(output_neurons,activation='relu',name='output')(hidden_tensor)               
    model_conv = Model(input_tensor,output_tensor) 

    # Compilación del modelo
    model_conv.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

    # Mostramos cómo es
    model_conv.summary()

    # Entrenamiento
    history = model_conv.fit(x[:tamaño_entrenamiento], y[:tamaño_entrenamiento], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x[tamaño_validacion:], y[tamaño_validacion:]))

    # Guardamos el modelo entrenado para ver los resultados
    model_conv.save("./IA/NN_models/pulse_trace_model_convolucional.h5")
