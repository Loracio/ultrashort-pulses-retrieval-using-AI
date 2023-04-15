import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from core import * 

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

if __name__ == '__main__':
    # np.random.seed(0)

    # Parámetros de la medida
    numero_de_muestras = 512
    duracion_temporal = 1 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    TBP = 2.75

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)
    frecuencias = frecuencias_DFT(numero_de_muestras, Δt)
    ω = convertir(frecuencias, 'f', 'ω')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    # Creamos pulso aleatorio con TBP especificado
    pulso, espectro = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)
    # pulso_candidato, espectro_candidato = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)

    # Pulso candidato como gaussiana con fase aleatoria
    fwhm = 5e-2
    phase_max = 0.3 * np.pi
    sigma = 0.5 * fwhm / np.sqrt(np.log(2.0))
    phase = np.exp(1.0j * np.random.uniform(-phase_max, phase_max, numero_de_muestras))
    d = t / sigma

    pulso_candidato = np.exp(-0.5 * d * d) * phase
    espectro_candidato = DFT(pulso_candidato, t, Δt, ω, Δω)

    # Plot intensidad y traza original
    fig0, ax0 = plot_traza(t, Δt, pulso_candidato, frecuencias, espectro_candidato)

    retriever = GPA_retriever(t, Δt, pulso)
    retrieved_field, retrieved_spectrum = retriever.recuperacion(pulso_candidato, 1e-10, max_iter=200)
    fig, ax = retriever.plot()

    plt.show()

