import numpy as np
import matplotlib.pyplot as plt
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import * 

plt.rcParams.update({'font.size': 20}) # Tamaño de la fuente del plot


if __name__ == '__main__':

    # Parámetros de la medida
    numero_de_muestras = 512
    duracion_temporal = .5 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2 , duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0 # Tiempo en el que el pulso tiene su máximo (ps)
    A = 2 # Amplitud del pulso
    ω_0 = 10e2 # Frecuencia angular del pulso (rad / ps)
    τ = 0.0005 # Duración del pulso (ps)
    a =  0 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ) # Chirpeo lineal

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")


    delays = np.linspace(-(numero_de_muestras - 1) * Δt, (numero_de_muestras - 1) * Δt, num=2 * numero_de_muestras - 1)

    pulso = np.exp(- (t-t0)*(t-t0) / (2 * τ)) * np.exp(1j *  φ)

    frecuencias = frecuencias_DFT(numero_de_muestras, Δt)
    ω = convertir(frecuencias, 'f', 'ω')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    espectro = DFT(pulso, t, Δt, ω, Δω)

    # plot_traza(t, Δt, pulso, frecuencias, espectro)


    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2

    fase_campo = np.unwrap(np.angle(pulso)) 
    fase_campo -=  media(fase_campo, I_pulso)
    fase_campo = np.where(I_pulso < 1e-10, np.nan, fase_campo)

    fase_espectro = np.unwrap(np.angle(espectro)) 
    fase_espectro -=  media(fase_espectro, I_espectro)
    fase_espectro = np.where(I_espectro < 1e-10, np.nan, fase_espectro)


    fig, ax = plt.subplots(2, 3)

    twin_ax1 = ax[0][0].twinx()
    ax[0][0].plot(t, I_pulso, color='blue')
    twin_ax1.plot(t, fase_campo, '-.', color='red')
    ax[0][0].plot(np.nan, '-.', color='red')
    ax[0][0].set_xlabel("Tiempo (ps)")
    ax[0][0].set_ylabel("Intensidad (u.a.)")
    twin_ax1.set_ylabel("Fase (rad)")
    # ax[0][0].set_title("Dominio temporal")
    ax[0][0].grid()
    # ax[0][0].legend()
    ax[0][0].yaxis.label.set_color('blue')
    ax[0][0].tick_params(axis='y', colors='blue')
    ax[0][0].spines['left'].set_color('blue')
    twin_ax1.spines['right'].set_color('red')
    twin_ax1.tick_params(axis='y', colors='red')
    twin_ax1.yaxis.label.set_color('red')

    T = traza(pulso, t, Δt, t.size)
    traza_normalizada = T / np.max(T)

    delays = np.linspace(-(t.size - 1) * Δt, (t.size - 1) * Δt, num=2 * t.size - 1)

    im = ax[1][0].pcolormesh(frecuencias, delays, traza_normalizada, cmap='YlGnBu_r')
    fig.colorbar(im, ax=ax[1][0])
    ax[1][0].set_xlabel("Frecuencia (1/ps)")
    ax[1][0].set_ylabel("Retraso (ps)")
    # ax[1][0].set_title(r"$\tilde{T}(\omega, \tau) = |\int_{-\infty}^{\infty} E(t)E(t - \tau) \exp^{- i \omega t} dt|^2$")


    a =  0.001# Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ) # Chirpeo lineal
    pulso = np.exp(- (t-t0)*(t-t0) / (2 * τ)) * np.exp(-1j * φ)

    espectro = DFT(pulso, t, Δt, ω, Δω)

    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2

    fase_campo = np.unwrap(np.angle(pulso)) 
    fase_campo -=  media(fase_campo, I_pulso)
    fase_campo = np.where(I_pulso < 1e-10, np.nan, fase_campo)

    fase_espectro = np.unwrap(np.angle(espectro)) 
    fase_espectro -=  media(fase_espectro, I_espectro)
    fase_espectro = np.where(I_espectro < 1e-10, np.nan, fase_espectro)

    twin_ax1 = ax[0][1].twinx()
    ax[0][1].plot(t, I_pulso, color='blue') 
    twin_ax1.plot(t, fase_campo, '-.', color='red')
    ax[0][1].plot(np.nan, '-.',  color='red')
    ax[0][1].set_xlabel("Tiempo (ps)")
    ax[0][1].set_ylabel("Intensidad (u.a.)")
    twin_ax1.set_ylabel("Fase (rad)")
    # ax[0][1].set_title("Dominio temporal")
    ax[0][1].grid()
    # ax[0][1].legend()
    ax[0][1].yaxis.label.set_color('blue')
    ax[0][1].tick_params(axis='y', colors='blue')
    ax[0][1].spines['left'].set_color('blue')
    twin_ax1.spines['right'].set_color('red')
    twin_ax1.tick_params(axis='y', colors='red')
    twin_ax1.yaxis.label.set_color('red')

    T = traza(pulso, t, Δt, t.size)
    traza_normalizada = T / np.max(T)

    delays = np.linspace(-(t.size - 1) * Δt, (t.size - 1) * Δt, num=2 * t.size - 1)

    im = ax[1][1].pcolormesh(frecuencias, delays, traza_normalizada, cmap='YlGnBu_r')
    fig.colorbar(im, ax=ax[1][1])
    ax[1][1].set_xlabel("Frecuencia (1/ps)")
    ax[1][1].set_ylabel("Retraso (ps)")
    # ax[1][1].set_title(r"$\tilde{T}(\omega, \tau) = |\int_{-\infty}^{\infty} E(t)E(t - \tau) \exp^{- i \omega t} dt|^2$")


    a =  -0.001# Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ) # Chirpeo lineal
    pulso = np.exp(- (t-t0)*(t-t0) / (2 * τ)) * np.exp(-1j * φ)
    espectro = DFT(pulso, t, Δt, ω, Δω)

    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2

    fase_campo = np.unwrap(np.angle(pulso)) 
    fase_campo -=  media(fase_campo, I_pulso)
    fase_campo = np.where(I_pulso < 1e-10, np.nan, fase_campo)

    fase_espectro = np.unwrap(np.angle(espectro)) 
    fase_espectro -=  media(fase_espectro, I_espectro)
    fase_espectro = np.where(I_espectro < 1e-10, np.nan, fase_espectro)

    twin_ax1 = ax[0][2].twinx()
    ax[0][2].plot(t, I_pulso, color='blue') 
    twin_ax1.plot(t, fase_campo, '-.', color='red')
    ax[0][2].plot(np.nan, '-.',  color='red')
    ax[0][2].set_xlabel("Tiempo (ps)")
    ax[0][2].set_ylabel("Intensidad (u.a.)")
    twin_ax1.set_ylabel("Fase (rad)")
    # ax[0][2].set_title("Dominio temporal")
    ax[0][2].grid()
    # ax[0][2].legend()
    ax[0][2].yaxis.label.set_color('blue')
    ax[0][2].tick_params(axis='y', colors='blue')
    ax[0][2].spines['left'].set_color('blue')
    twin_ax1.spines['right'].set_color('red')
    twin_ax1.tick_params(axis='y', colors='red')
    twin_ax1.yaxis.label.set_color('red')

    T = traza(pulso, t, Δt, t.size)
    traza_normalizada = T / np.max(T)

    delays = np.linspace(-(t.size - 1) * Δt, (t.size - 1) * Δt, num=2 * t.size - 1)

    im = ax[1][2].pcolormesh(frecuencias, delays, traza_normalizada, cmap='YlGnBu_r')
    fig.colorbar(im, ax=ax[1][2])
    ax[1][2].set_xlabel("Frecuencia (1/ps)")
    ax[1][2].set_ylabel("Retraso (ps)")
    # ax[1][2].set_title(r"$\tilde{T}(\omega, \tau) = |\int_{-\infty}^{\infty} E(t)E(t - \tau) \exp^{- i \omega t} dt|^2$")

    plt.show()