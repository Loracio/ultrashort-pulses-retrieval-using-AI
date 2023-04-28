import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

if __name__ == '__main__':

# Parámetros de la medida
    numero_de_muestras = 256
    duracion_temporal = 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia central del pulso (rad / ps)

    TBP = 2.501

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # Definimos el aray de los delays
    delays = np.linspace(-(numero_de_muestras - 1) * Δt, (numero_de_muestras - 1) * Δt, num=2 * numero_de_muestras - 1)
    # Array de frecuencias
    frecuencias = convertir(ω_0, 'frecuencia angular', 'frecuencia') + frecuencias_DFT(numero_de_muestras, Δt)

    # Creamos pulso aleatorio con TBP especificado
    pulso, espectro = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)

    # Comprobamos TBP obtenido
    TBP_obtenido = desviacion_estandar(t, np.abs(pulso)**2) * desviacion_estandar(convertir(frecuencias, 'f', 'ω'), np.abs(espectro)**2)
    print(f'El pulso obtenido tiene un producto tiempo ancho de banda = {TBP_obtenido}')

    """
    Representación de la intensidad del pulso en el dominio temporal
    y la intensidad espectral en el dominio frecuencial

    A su derecha representaremos su traza
    """

    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2


    fig = plt.figure()

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])


    twin_ax1 = ax1.twinx()
    twin_ax2 = ax2.twinx()

    fig.suptitle(f"Pulso aleatorio con TBP={TBP_obtenido:.2f}")

    fase_campo = np.unwrap(np.angle(pulso)) 
    fase_campo -=  media(fase_campo, I_pulso)

    fase_espectro = np.unwrap(np.angle(espectro)) 
    fase_espectro -=  media(fase_espectro, I_espectro)

    ax1.plot(t, I_pulso, color='blue', label='Intensidad')
    twin_ax1.plot(t, fase_campo, '-.', color='red')
    ax1.plot(np.nan, '-.', label='Fase', color='red')
    ax1.set_xlabel("Tiempo (ps)")
    ax1.set_ylabel("Intensidad (u.a.)")
    twin_ax1.set_ylabel("Fase (rad)")
    ax1.set_title("Dominio temporal")
    ax1.grid()
    ax1.legend()

    ax2.plot(frecuencias, I_espectro, color='blue', label='Intensidad espectral')
    twin_ax2.plot(frecuencias, fase_espectro, '-.', color='red')
    ax2.plot(np.nan, '-.', label='Fase', color='red')
    ax2.set_xlabel("Frecuencia (1 / ps)")
    ax2.set_ylabel("Intensidad (u.a.)")
    twin_ax2.set_ylabel("Fase (rad)")
    ax2.set_title("Dominio frecuencial")
    ax2.grid()
    ax2.legend()


    """
    Representación de la traza del pulso, definida por:

        T(ω, τ) =  | ∫ E(t)E(t - τ) exp(-i ω t) dt |²

    Es decir:

        T(ω, τ) = |FT[E(t)E(t - τ)]|²

    """

    T = traza(pulso, t, Δt, numero_de_muestras)
    T_normalizada = T / np.max(T)

    im = ax3.pcolormesh(frecuencias, delays, T_normalizada, cmap='inferno')
    fig.colorbar(im, ax=ax3)
    ax3.set_xlabel("Frecuencia (1/ps)")
    ax3.set_ylabel("Retraso (ps)")
    ax3.set_title(r"$\tilde{T}(\omega, \tau) = |\int_{-\infty}^{\infty} E(t)E(t - \tau) \exp^{- i \omega t} dt|^2$")

    plt.show()