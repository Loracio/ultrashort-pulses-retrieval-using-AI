import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from core import * 

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

if __name__ == '__main__':
    np.random.seed(0)

    # Parámetros de la medida
    numero_de_muestras = 256
    duracion_temporal = 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia central del pulso (rad / ps)

    TBP = 2.501

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # Creamos pulso aleatorio con TBP especificado
    pulso, espectro = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)
    # pulso_candidato, espectro_candidato = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)

    #! Prueba
    # fase = np.exp(2j * np.pi * np.random.rand(N))
    # espectro = filtro_espectral * fase
    # pulso_candidato = IDFT(espectro, t, Δt, ω, Δω)

    fwhm = 1
    phase_max = 0.3 * np.pi
    sigma = 0.5 * fwhm / np.sqrt(np.log(2.0))
    phase = np.exp(1.0j * np.random.uniform(-phase_max, phase_max, numero_de_muestras))
    d = t / sigma
    pulso_candidato = np.exp(-0.5 * d * d) * phase
    pulso_candidato /= np.max(pulso_candidato)

    # fig0, ax0 = plot_real_imag(t, pulso_candidato)
    # plt.show()

    retriever = GPA_retriever(t, Δt, pulso, espectro=espectro)
    retrieved_field, retrieved_spectrum = retriever.recuperacion(pulso_candidato, 1e-7, max_iter=2000)
    fig, ax = retriever.plot()

    plt.show()

