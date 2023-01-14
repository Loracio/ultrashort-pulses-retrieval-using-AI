import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.constants as constants
from core import * 

plt.rcParams.update({'font.size': 16}) # Tamaño de la fuente del plot


if __name__ == '__main__':

    # Parámetros de la medida
    numero_de_muestras = 2*4096
    duracion_temporal = 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal

    # -- Parámetros del pulso --
    t0 = 5 # # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 * np.ones(numero_de_muestras) # Fase (constante en este caso)
    τ = 1 # Duración del pulso (ps)


    t, Δt = np.linspace(0, duracion_temporal, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)
    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ_0) # Vector con el campo complejo del pulso
    I = np.abs(pulso) * np.abs(pulso) # Vector con la intensidad del pulso

    frecuencias = frecuencias_DFT(numero_de_muestras, Δt)
    ω = convertir(frecuencias, 'frecuencia', 'frecuencia angular')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    transformada_analitica = transformada_pulso_gaussiano(ω, t0, A, τ, ω_0, φ_0) # Transformada analítica de un pulso gaussiano con fase constante

    # Plot partes real e imaginaria del pulso
    plot_real_imag(t, pulso, φ_0, I)
    plt.show()
    # Comprobamos que el hacer la transformada y su inversa nos devuelve el pulso original
    plot_real_imag(t, IDFT(DFT(pulso, t, Δt, ω, Δω), t, Δt, ω, Δω), φ_0, np.abs(IDFT(DFT(pulso, t, Δt, ω, Δω), t, Δt, ω, Δω))**2)
    plt.show()

    # Plot de la intensidad
    plot_intensidad(t, I)
    plt.show()

    # Comprobamos que el hacer su transformada inversa nos devuelve el pulso original
    plot_real_imag(t, IDFT(transformada_analitica, t, Δt, ω, Δω), φ_0, np.abs(IDFT(transformada_analitica, t, Δt, ω, Δω))**2)
    plt.show()
    #! Problemas con los factores de normalización

    # -- Plot : comparación de los resultados obtenidos por np.fft.fft, scipy.fft.fft y mi implementación
    transformada_numpy = np.fft.fft(pulso)
    transformada_scipy = scipy.fft.fft(pulso)
    transformada_propia = DFT(pulso, t, Δt, ω, Δω)

    diferencias_numpy = np.abs(transformada_numpy - transformada_propia) * 1e13
    diferencias_scipy = np.abs(transformada_scipy - transformada_propia) * 1e13
    diferencias_ambos = np.abs(transformada_scipy - transformada_numpy) * 1e13

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(ω, diferencias_numpy, label='Diferencia con NumPy')
    ax[1].plot(ω, diferencias_scipy, label='Diferencia con SciPy', color='orange')
    ax[2].plot(ω, diferencias_ambos, label='Diferencia entre NumPy y SciPy', color='green')
    fig.suptitle("Comprobación valores obtenidos con las distintas funciones")
    fig.supxlabel("ω (rad / ps)")
    fig.supylabel(r"Diferencia entre valores ($\times 10^{-13}$)")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.show()


    # -- Plot : comparación de los resultados obtenidos por fft y DFT
    transformada_DFT = DFT(pulso, t, Δt, ω, Δω)

    diferencias_DFT_fft = np.abs(transformada_propia - transformada_DFT) * 1e10

    fig, ax = plt.subplots()
    ax.plot(ω, diferencias_DFT_fft)
    fig.suptitle("Comprobación valores obtenidos con fft y DFT")
    fig.supxlabel("ω (rad / ps)")
    fig.supylabel(r"Diferencia entre valores ($\times 10^{-10}$)")
    ax.grid()

    plt.show()

    #! Las diferencias de resultados entre algoritmos son muy pequeñas y la velocidad de la fft es muy superior a la DFT

    # Comparacion resultado entre la transformada analítica y la fft. Comparo la transformada normalizada porque no tienen la misma escala

    fft_normalizada = transformada_propia / np.max(transformada_propia)
    analitica_normalizada = transformada_analitica / np.max(transformada_analitica)

    diferencias_analitica_fft = np.abs(fft_normalizada - analitica_normalizada)

    fig, ax = plt.subplots()
    ax.plot(ω, diferencias_analitica_fft)
    fig.supylabel("Diferencia entre amplitud coeficientes")
    fig.supxlabel("ω (rad / ps)")
    ax.set_title("Diferencia entre fft normalizdas")

    ax.grid()

    plt.show()

    # Comparación de resultados entre la transformada inversa ifft y analítica

    diferencias_inversas = np.abs(IDFT(fft_normalizada, t, Δt, ω, Δω) -  IDFT(analitica_normalizada, t, Δt, ω, Δω))

    fig, ax = plt.subplots()
    ax.plot(t, diferencias_inversas)
    fig.supylabel("Diferencia entre valores")
    fig.supxlabel("t (ps)")
    ax.set_title("Diferencia entre ifft normalizadas")
    ax.grid()

    plt.show()

    # Transformadas de pulsos con distintas anchuras temporales 
    τ_1, τ_2, τ_3 = 2.0, 1.0, 0.5 # En ps
    pulso_1 = fft(pulso_gaussiano(t, t0, A, τ_1, ω_0, φ_0))
    pulso_2 = fft(pulso_gaussiano(t, t0, A, τ_2, ω_0, φ_0))
    pulso_3 = fft(pulso_gaussiano(t, t0, A, τ_3, ω_0, φ_0))

    fig, ax = plt.subplots()
    ax.plot(ω, np.abs(pulso_1), label=r"$\tau =$"+f"{τ_1} ps")
    ax.plot(ω, np.abs(pulso_2), label=r"$\tau =$"+f"{τ_2} ps")
    ax.plot(ω, np.abs(pulso_3), label=r"$\tau =$"+f"{τ_3} ps")
    ax.set_ylabel("Amplitud coeficientes")
    ax.set_xlabel("ω (rad / ps)")
    # ax.set_xlim([1200, 1230])
    ax.grid()
    ax.legend()
    # Vemos que se cumple que a mayor anchura temporal menor anchura espectral
    plt.show()