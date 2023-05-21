import numpy as np
import matplotlib.pyplot as plt
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import * 

if __name__ == '__main__':

# Parámetros de la medida
    numero_de_muestras = 512
    duracion_temporal = .5 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2 , duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0.02# Tiempo en el que el pulso tiene su máximo (ps)
    A = 2 # Amplitud del pulso
    ω_0 = 10e2 # Frecuencia angular del pulso (rad / ps)
    τ = 0.015 # Duración del pulso (ps)
    a = 0 # Parámetro de chirp del pulso 
    φ =  0 # Chirpeo lineal

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")


    delays = np.linspace(-(numero_de_muestras - 1) * Δt, (numero_de_muestras - 1) * Δt, num=2 * numero_de_muestras - 1)

    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ)

    frecuencias = frecuencias_DFT(numero_de_muestras, Δt)
    ω = convertir(frecuencias, 'f', 'ω')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    espectro = DFT(pulso, t, Δt, ω, Δω)

    plot_traza(t, Δt, pulso, frecuencias, espectro)

    plt.show()