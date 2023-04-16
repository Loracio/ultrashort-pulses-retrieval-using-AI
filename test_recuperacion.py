import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from core import * 

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

if __name__ == '__main__':

    """
    Script para probar los algoritmos de reconstrucción de pulsos.

    Creamos un pulso aleatorio de TBP especificado que vamos a reconstuir.
    Como pulso candidato inicial tomamos una gaussiana con fase aleatoria.

    Probamos los dos métodos implementados: GPA y PCGPA.
    """

    # Parámetros de la medida
    numero_de_muestras = 256
    duracion_temporal = 1 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    TBP = 2.5

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)
    frecuencias = frecuencias_DFT(numero_de_muestras, Δt)
    ω = convertir(frecuencias, 'f', 'ω')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    # Creamos pulso aleatorio con TBP especificado
    pulso, espectro = pulso_aleatorio(t, Δt, numero_de_muestras, TBP)

    # Pulso candidato como gaussiana con fase aleatoria
    fwhm = .125
    fase_max = 0.3 * np.pi
    sigma = 0.5 * fwhm / np.sqrt(np.log(2.0))
    fase = np.exp(1.0j * np.random.uniform(-fase_max, fase_max, numero_de_muestras))
    d = t / sigma

    pulso_candidato = np.exp(-0.5 * d * d) * fase
    espectro_candidato = DFT(pulso_candidato, t, Δt, ω, Δω)

    # Plot intensidad y traza original
    fig0, ax0 = plot_traza(t, Δt, pulso_candidato, frecuencias, espectro_candidato)

    retriever_PCGPA = PCGPA_retriever(t, Δt, pulso)
    campo_recuperado_PCGPA, espectro_recuperado_PCGPA = retriever_PCGPA.recuperacion(pulso_candidato, 1e-10, max_iter=1000)
    fig_PCGPA, ax_PCGPA = retriever_PCGPA.plot()

    retriever_GPA = GPA_retriever(t, Δt, pulso)
    campo_recuperado_GPA, espectro_recuperado_GPA = retriever_GPA.recuperacion(pulso_candidato, 1e-10, max_iter=200)
    fig_GPA, ax_GPA = retriever_GPA.plot()

    plt.show()

