"""
Este script proporciona las subrutinas para leer y formatear la base de datos creada
en 'crea_database.py'
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *


def preprocesamiento(fila, N):
    """
    Función de preprocesamiento de los datos, pensada para llamarse
    como un mapeo a los elementos de un objeto del tipo tensofrflow.Dataset

    El cuerpo de la función puede modificarse en función de cómo queramos
    formatear nuestros datos. 

    Los resultados serán guardados en un nuevo dataset que contendrá los
    tensores devueltos. Más adelante se puede aplicar un post-procesado
    para trabajar con np.arrays en vez de con estos tensores-objeto.

    Args:
        fila : fila de la base de datos
        N (int): número de muestras del pulso
    """
    # Dividimos cada fila por sus columnas
    columns = tf.strings.split(fila, sep=',')

    # Valor TBP del pulso
    TBP = tf.strings.to_number(columns[0])

    # Valores de la parte real del campo eléctrico
    E_r = tf.strings.to_number(columns[1:(N+1)])

    # Valores de la parte imaginaria del campo eléctrico
    E_i = tf.strings.to_number(columns[(N+1):(2*N + 1)])

    concat_Er_Ei = tf.concat([E_r, E_i], axis=0)

    # Valores de la traza
    T = tf.strings.to_number(columns[(2*N + 1):], out_type=tf.float32)

    return TBP, concat_Er_Ei, T

def wrapper_preprocesamiento(N):
    def wrapper_fn(row):
        return preprocesamiento(row, N)
    return wrapper_fn

def dataset_to_numpy(ds, N, NUMERO_PULSOS):
    """
    A partir de un objeto del tipo tensorflow.Dataset ya preprocesado,
    extraemos los tensores de entrada y de salida y los formateamos para
    que sean del tipo np.array.

    Args:
        ds (tensorflow.Dataset object): base de datos de los pulsos
        N (int): número de muestras de un pulso
        NUMERO_PULSOS (int): número total de pulsos de la base de datos

    Devuelve:
        x, y (np.array): arrays con la información de los tensores de entrada y salida
    """
    TBP = np.zeros(NUMERO_PULSOS)
    x = np.zeros((NUMERO_PULSOS, 2*N))
    y = np.zeros((NUMERO_PULSOS, (2*N - 1) * N))
    for i, example in enumerate(ds):
        TBP[i] = example[0].numpy()
        x[i][:] = example[1].numpy()
        y[i][:] = example[2].numpy()

    return TBP, x, y

def formateador(direccion_archivo, N, NUMERO_PULSOS, buffer_size=None, shuffle=False):
    """
    Lee y formatea los datos de la base de datos proporcionada, devolviéndolos
    en formato np.array con los tenores de entrada y de salida.

    Args:
        direccion_archivo (str): ruta a la base de datos a cargar (en formato csv)
        N (int): número de muestras de un pulso
        NUMERO_PULSOS (int): número total de pulsos de la base de datos
        buffer_size (int, optional): Tamaño del buffer de datos. Por defecto se coge toda la base.

    Devuelve:
        x, y (np.array): arrays con la información de los tensores de entrada y salida
    """

    if buffer_size is None:
        buffer_size = NUMERO_PULSOS

    # Función de lectura de TensorFlow
    dataset = tf.data.TextLineDataset(direccion_archivo)

    # No leemos la cabecera del archivo, que contiene el contenido de la columna
    dataset = dataset.skip(1)

    if shuffle:
        # Mezclamos aleatoriamente los pulsos
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Aplicamos el preprocesamiento a cada fila
    dataset = dataset.map(wrapper_preprocesamiento(N))

    return dataset_to_numpy(dataset, N, NUMERO_PULSOS)

if __name__ == '__main__':
    # Parámetros lectura
    N = 128
    NUMERO_PULSOS = 1000

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"

    TBP, x, y = formateador(direccion_archivo, N, NUMERO_PULSOS, shuffle=True)

    # Cogemos un elemento del conjunto de datos para ver que todo está bien cargado

    pulso = x[0][:N] + 1j * x[0][N:]
    traza_leida = y[0][:].reshape(2*N - 1, N) # Traza de tamaño (2*N - 1) * N

    # Comprobamos si el pulso y su traza calculada coinciden
    duracion_temporal = 1 # Duración temporal (ps)
    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=N, retstep=True)
    frecuencias = frecuencias_DFT(N, Δt)
    Δω = 2*np.pi / (N*Δt)

    espectro = DFT(pulso, t, Δt, convertir(frecuencias, 'f', 'ω'), Δω)

    plot_traza(t, Δt, pulso, frecuencias, espectro)

    # Plot rápido de la traza de la base 
    fig, ax = plt.subplots()
    ax.imshow(traza_leida, aspect='auto', cmap='inferno')
    ax.set_title("Traza leída de la base de datos")

    plt.show()