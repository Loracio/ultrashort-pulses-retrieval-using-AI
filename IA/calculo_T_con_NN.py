"""
En este script probaremos a diseñar una NN capaz de calcular la traza
de un pulso dados los valores de su campo eléctrico (partes real e imaginaria).

Este script simplemente contiene la arquitectura de la red y su entrenamiento,
para ver los resultados usar el archivo 'visualizacion_NN.py'
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *

from lee_database import formateador


if __name__ == '__main__':
    # Parámetros lectura
    N = 128
    NUMERO_PULSOS = 5000

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"
 
    # Cargamos los datos como np.ararys
    TBP, x, y = formateador(direccion_archivo, N, NUMERO_PULSOS, shuffle=True)
 
    # Separamos entre conjunto de entrenamiento y validación
    tamaño_entrenamiento = int(0.9 * NUMERO_PULSOS)
    tamaño_validacion = NUMERO_PULSOS - tamaño_entrenamiento

    print(f"Tamaño del conjunto de entrenamiento: {tamaño_entrenamiento}")
    print(f"Tamaño del conjunto de validación: {tamaño_validacion}")


    # Parámetros de la red
    EPOCHS = 50
    BATCH_SIZE = 32

    input_shape = (2*N,)
    hidden_layer_neurons = 64
    output_neurons = (2*N - 1) * N

    # Construcción de la arquitectura
    input_tensor = Input(shape=input_shape ,name='input')
    hidden_tensor = Dense(hidden_layer_neurons, activation='relu',name='hidden')(input_tensor)  
    output_tensor = Dense(output_neurons,activation='relu',name='output')(hidden_tensor)               
    model = Model(input_tensor,output_tensor) 

    # Compilación del modelo
    model.compile(loss="mse", optimizer="adam")

    # Mostramos cómo es
    model.summary()

    # Entrenamiento
    history = model.fit(x[:tamaño_entrenamiento], y[:tamaño_entrenamiento], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x[tamaño_validacion:], y[tamaño_validacion:]))

    # Guardamos el modelo entrenado para ver los resultados
    model.save("./IA/NN_models/pulse_trace_model.h5")
