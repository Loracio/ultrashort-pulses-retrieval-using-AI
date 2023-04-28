"""
En este archivo podremos ver los resultados de la NN entrenada
en el archivo 'calculo_T_con_NN.py'

(De momento está un poco chafardero)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *
from lee_database import formateador

if __name__ == '__main__':

    N = 128
    NUMERO_PULSOS = 5000

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"

    # Cargamos los datos como np.ararys
    x, y = formateador(direccion_archivo, N, NUMERO_PULSOS)

    model = tf.keras.models.load_model("./IA/NN_models/pulse_trace_model.h5")
    model.summary()

    y_pred = model.predict(x)

    for i in range(NUMERO_PULSOS):

        traza_pred = y_pred[i].reshape(2*N - 1, N) # Traza de tamaño (2*N - 1) * N
        traza_leida = y[i][:].reshape(2*N - 1, N) # Traza de tamaño (2*N - 1) * N

        # Plot rápido de la traza de la base 
        fig, ax = plt.subplots()
        ax.imshow(traza_pred, aspect='auto', cmap='inferno')
        ax.set_title("Traza producida por la NN")

        fig1, ax1 = plt.subplots()

        ax1.imshow(traza_leida, aspect='auto', cmap='inferno')
        ax1.set_title("Traza leída de la base de datos")

        plt.show()