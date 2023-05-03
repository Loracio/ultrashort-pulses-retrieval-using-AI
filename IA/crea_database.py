"""
Este script genera una base de datos con la siguiente arquitectura:

Columna   |     Tipo datos	 |   Descripción
-------------------------------------------------------------------------------------------------------------------------------
TBP	      |      float	     |   Producto tiempo-ancho de banda del pulso
E_r	      |      array	     |   Parte real del campo eléctrico del pulso (N columnas)
E_i	      |      array	     |   Parte imaginaria del campo eléctrico del pulso (N columnas)
T	      |      array	     |   Traza del pulso ((2N -1)xN columnas)

Para así poder ser leída y procesada posteriormente para entrenar una red neuronal
capaz de calcular la traza de un pulso dado su campo eléctrico.
"""

import numpy as np
import time 
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import pulso_aleatorio, traza
import csv


def genera_pulso_traza(t, Δt, N, TBP):
    """
    Genera un pulso aleatorio del TBP especificado y calcula su traza.

    Args:
        t (np.array[np.float64]): vector de tiempos
        N (int): número de muestras
        TBP (float): producto tiempo ancho de banda del pulso a generar

    Returns:
        E_r, E_i, T: partes real, imaginaria y traza del pulso generado
    """

    pulso, espectro = pulso_aleatorio(t, Δt, N, TBP)

    T = traza(pulso, t, Δt, N)

    return pulso.real, pulso.imag, T

if __name__ == '__main__':

    # Parámetros del pulso
    N = 128
    duracion_temporal = 1 # Duración temporal (ps)


    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=N, retstep=True) # Vector de tiempos del pulso

    TBP_inicial = 0.51
    TBP_final = 1.51
    paso_TBP = 0.10

    pulsos_por_TBP = 200

    numero_total_pulsos = int(np.ceil(pulsos_por_TBP * (TBP_final - TBP_inicial) / paso_TBP))

    print("Total pulsos generados: ", numero_total_pulsos)

    start = time.perf_counter()
    with open(f"./IA/DataBase/{numero_total_pulsos}_pulsos_aleatorios_N{N}.csv", "w", newline='') as file_out:
        writer = csv.writer(file_out)

        header = ['TBP'] + ['E_r['+str(j)+']' for j in range(N)] + ['E_i[]'+str(j)+']' for j in range(N)] + ['T['+str(m)+ '][' + str(n) + ']' for m in range(2 * N - 1) for n in range(N)]
        writer.writerow(header)

        for TBP in np.arange(TBP_inicial, TBP_final, step=paso_TBP):
            for i in range(pulsos_por_TBP):

                E_r, E_i, T = genera_pulso_traza(t, Δt, N, TBP)

                fila = [TBP] + E_r.tolist() + E_i.tolist() + T.flatten().tolist()
                writer.writerow(fila)

    end = time.perf_counter()

    print(f'Datos generados en {end-start} segundo(s)')