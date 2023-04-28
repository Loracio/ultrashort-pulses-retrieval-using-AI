import numpy as np
import matplotlib.pyplot as plt
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import * 

if __name__ == '__main__':

# Parámetros de la medida
    numero_de_muestras = 4*512
    duracion_temporal = 1 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2 , duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0# Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 # Fase del pulso (constante en este caso)
    τ = 0.2 # Duración del pulso (ps)
    a = 0 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ ) # Chirpeo lineal

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

    # plot_traza(t, Δt, pulso, frecuencias, espectro)

    # plt.show()

    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2

    fig = plt.figure()

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])


    twin_ax1 = ax1.twinx()
    twin_ax2 = ax2.twinx()


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