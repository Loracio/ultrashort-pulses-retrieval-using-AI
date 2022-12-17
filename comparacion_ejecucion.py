import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from time import process_time as timer
from core import *
import scipy.fft

plt.rcParams.update({'font.size': 16}) # Tamaño de la fuente del plot


def media_tiempo_ejecucion(funcion, pulso, n_repeticiones):
    """
    Esta función aplica la transformada de Fourier pasada como argumento para calcular
    el tiempo promedio que tarda en ejecutarse. Notar que el tiempo de ejecución dependerá
    de la longitud del array del pulso.

    Argumentos:
        funcion (funcion): funcion para calcular su tiempo de ejecucion promedio
        pulso (float, np.array): array de la señal a la que realizaremos su transformada de fourier
        n_repeticiones (int): numero de repeticiones para el promedio

    Devuelve:
        t (float): media de los tiempos de ejecución
        std (float): desviación estándar de la media de tiempos de ejecución
    """

    tiempos = np.empty(n_repeticiones)

    for i in range(n_repeticiones):
        tic = timer()
        funcion(pulso)
        toc = timer()
        tiempos[i] = toc-tic

    tiempos *= 1e6 # pasar a µs

    t   = np.mean(tiempos)
    std = np.std(tiempos)

    return t, std

if __name__ == '__main__':

    numero_de_muestras = 4096

    # -- Parámetros del pulso --
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 = np.array([np.pi / 4 for i in range(numero_de_muestras)]) # Fase (constante en este caso)
    τ = 1 # Duración del pulso (ps)

    t = np.linspace(-5, 5, num=numero_de_muestras) # Vector de tiempos (centrado en cero, ps)
    pulso = pulso_gaussiano(t, A, τ, ω_0, φ_0) # Vector con el campo complejo del pulso

    n_repeticiones = 10000

    tiempo_DFT, std_DFT = media_tiempo_ejecucion(DFT, pulso, n_repeticiones)
    print(f"timing DFT: {tiempo_DFT:<.5f} ± {std_DFT:<.5f} μs")

    tiempo_fft_propia, std_fft_propia = media_tiempo_ejecucion(fft, pulso, n_repeticiones)
    print(f"timing fft propia: {tiempo_fft_propia:<.5f} ± {std_fft_propia:<.5f} μs")

    tiempo_fft_numpy, std_fft_numpy = media_tiempo_ejecucion(np.fft.fft, pulso, n_repeticiones)
    print(f"timing fft numpy: {tiempo_fft_numpy:<.5f} ± {std_fft_numpy:<.5f} μs")

    tiempo_fft_scipy, std_fft_scipy = media_tiempo_ejecucion(scipy.fft.fft, pulso, n_repeticiones)
    print(f"timing fft scipy: {tiempo_fft_scipy:<.5f} ± {std_fft_scipy:<.5f} μs")
    
    tiempo_DFT_Julia, std_DFT_Julia = 134254.53038799998, 14095.362552302218
    tiempo_FFT_Julia, std_FFT_Julia = 6154.7076990000005, 974.121709443616
    tiempo_FFTW_Julia, std_FFTW_Julia = 57.06965399999999, 18.22369417059855

    fig, ax = plt.subplots()
    ax.scatter('DFT', tiempo_DFT)
    ax.errorbar('DFT', tiempo_DFT, yerr=std_DFT)
    ax.scatter('DFT Julia', tiempo_DFT_Julia)
    ax.errorbar('DFT Julia', tiempo_DFT_Julia, yerr=std_DFT_Julia)
    ax.scatter('FFT', tiempo_fft_propia)
    ax.errorbar('FFT', tiempo_fft_propia, yerr=std_fft_propia)
    ax.scatter('FFT Julia', tiempo_FFT_Julia)
    ax.errorbar('FFT Julia', tiempo_FFT_Julia, yerr=std_FFT_Julia)
    ax.scatter('NumPy', tiempo_fft_numpy)
    ax.errorbar('NumPy', tiempo_fft_numpy, yerr=std_fft_numpy)
    ax.scatter('SciPy', tiempo_fft_scipy)
    ax.errorbar('SciPy', tiempo_fft_scipy, yerr=std_fft_scipy)
    ax.scatter('FFTW Julia', tiempo_FFTW_Julia)
    ax.errorbar('FFTW Julia', tiempo_FFTW_Julia, yerr=std_FFTW_Julia)

    ax.set_ylabel("Tiempo de ejecución (μs)")

    ax.set_title(f"Comparación ejecución con N = {numero_de_muestras}, ejecutado {n_repeticiones} veces")
    

    ax.grid()
    plt.show()