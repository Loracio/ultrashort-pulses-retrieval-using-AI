import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from core import * 

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot


if __name__ == '__main__':

    """
    En este script voy a realizar pruebas sencillas con distintos pulsos
    y las distintas autocorrelaciones definidas:

        - Autocorrelación de segundo orden:
            A⁽²⁾(τ) = ∫I(t)I(t-τ) dt

        - Autocorrelación de tercer orden:
            A⁽³⁾(τ) = ∫I(t)I²(t-τ) dt

        - Autocorrelación "FRAC":
            I_FRAC(τ) = ∫|[E(t) + E(t-τ)]²|² dt
    """

    # Parámetros de la medida
    numero_de_muestras = 2 * 4096
    duracion_temporal = 2 * 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0 # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 # Fase del pulso (constante en este caso)
    τ = 1 # Duración del pulso (ps)
    a = 3 * np.pi / 4 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ ) # Chirpeo lineal

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")

    # Definimos el aray de los delays (es igual que el de tiempos)
    delays = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras)

    """
    Ejemplos de autocorrelación de segundo orden:
        - Pulso gaussiano genérico
        - Doble pulso

    Para emplear la función de autocorrelación implementada tenemos que definir 
    una función que calcule la intensidad del pulso en función de los parámetros
    del mismo.

    Es de esperar que sea simétrico, A⁽²⁾(τ) = A⁽²⁾(-τ)
    """

    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ)

    def doble_pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2):
        """
        Crea un doble pulso gaussiano como suma de dos pulsos con diferentes parámetros.
        """
        return pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1) + pulso_gaussiano(t, t0_2, A_2, τ_2, ω_0_2, φ_2)

    doble_pulso = doble_pulso_gaussiano(t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ)

    def intensidad(t, t0, A, τ, ω_0, φ):
        """
        Definimos una función que calcule la intensidad a partir de los parámetros del pulso
        """
        return np.abs(pulso_gaussiano(t, t0, A, τ, ω_0, φ))**2

    def intensidad_doble_pulso(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2):
        """
        Definimos una función que calcule la intensidad a partir de los parámetros del doble pulso
        """
        return np.abs(doble_pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2))**2

    
    # Representación de la intensidad del pulso con su correspondiente autocorrelación de segundo orden
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(t, np.abs(pulso)**2, label='Pulso')
    ax[0][0].set_xlabel("Tiempo (ps)")
    ax[0][0].set_ylabel("Intensidad (u.a.)")
    ax[0][0].grid()
    ax[0][0].legend()

    ax[0][1].plot(delays, autocorrelacion_2orden(delays, Δt, intensidad, t, t0, A, τ, ω_0, φ), label=r'$A^{(2)}(\tau)$')
    ax[0][1].set_xlabel("Retardo (ps)")
    ax[0][1].set_ylabel("Intensidad (u.a.)")
    ax[0][1].grid()
    ax[0][1].legend()

    ax[1][0].plot(t, np.abs(doble_pulso)**2, color='r', label='Pulso')
    ax[1][0].set_xlabel("Tiempo (ps)")
    ax[1][0].set_ylabel("Intensidad (u.a.)")
    ax[1][0].grid()
    ax[1][0].legend()

    ax[1][1].plot(delays, autocorrelacion_2orden(delays, Δt, intensidad_doble_pulso, t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), color='r', label=r'$A^{(2)}(\tau)$')
    ax[1][1].set_xlabel("Retardo (ps)")
    ax[1][1].set_ylabel("Intensidad (u.a.)")
    ax[1][1].grid()
    ax[1][1].legend()

    fig.suptitle("Autocorrelación de segundo orden")


    """
    Ejemplos cálculo autocorrelación de tercer orden.
    En el caso del doble pulso, es de esperar que no sea simétrica.
    """

    # Representación de la intensidad del pulso con su correspondiente autocorrelación de tercer orden
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(t, np.abs(pulso)**2, label='Pulso')
    ax[0][0].set_xlabel("Tiempo (ps)")
    ax[0][0].set_ylabel("Intensidad (u.a.)")
    ax[0][0].grid()
    ax[0][0].legend()

    ax[0][1].plot(delays, autocorrelacion_3orden(delays, Δt, intensidad, t, t0, A, τ, ω_0, φ), label=r'$A^{(3)}(\tau)$')
    ax[0][1].set_xlabel("Retardo (ps)")
    ax[0][1].set_ylabel("Intensidad (u.a.)")
    ax[0][1].grid()
    ax[0][1].legend()

    ax[1][0].plot(t, np.abs(doble_pulso)**2, color='r', label='Pulso')
    ax[1][0].set_xlabel("Tiempo (ps)")
    ax[1][0].set_ylabel("Intensidad (u.a.)")
    ax[1][0].grid()
    ax[1][0].legend()

    ax[1][1].plot(delays, autocorrelacion_3orden(delays, Δt, intensidad_doble_pulso, t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), color='r', label=r'$A^{(3)}(\tau)$')
    ax[1][1].set_xlabel("Retardo (ps)")
    ax[1][1].set_ylabel("Intensidad (u.a.)")
    ax[1][1].grid()
    ax[1][1].legend()

    fig.suptitle("Autocorrelación de tercer orden")

    """
    Ejemplos cálculo autocorrelación "FRAC".
    """

    # Representación de la intensidad del pulso con su correspondiente autocorrelación de tercer orden
    fig, ax = plt.subplots(2, 2)
    ax[0][0].plot(t, np.abs(pulso)**2, label='Pulso')
    ax[0][0].set_xlabel("Tiempo (ps)")
    ax[0][0].set_ylabel("Intensidad (u.a.)")
    ax[0][0].grid()
    ax[0][0].legend()

    ax[0][1].plot(delays, FRAC(delays, Δt, pulso_gaussiano, t, t0, A, τ, ω_0, φ), label="FRAC")
    ax[0][1].set_xlabel("Retardo (ps)")
    ax[0][1].set_ylabel("Intensidad (u.a.)")
    ax[0][1].grid()
    ax[0][1].legend()

    ax[1][0].plot(t, np.abs(doble_pulso)**2, color='r', label='Pulso')
    ax[1][0].set_xlabel("Tiempo (ps)")
    ax[1][0].set_ylabel("Intensidad (u.a.)")
    ax[1][0].grid()
    ax[1][0].legend()

    ax[1][1].plot(delays, FRAC(delays, Δt, doble_pulso_gaussiano, t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), color='r', label="FRAC")
    ax[1][1].set_xlabel("Retardo (ps)")
    ax[1][1].set_ylabel("Intensidad (u.a.)")
    ax[1][1].grid()
    ax[1][1].legend()

    fig.suptitle("Autocorrelación de tercer orden")

    plt.show()