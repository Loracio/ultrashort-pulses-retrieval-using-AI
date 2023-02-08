import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from core import * 
from matplotlib.widgets import Slider

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot


if __name__ == '__main__':

    """
    Vamos a realizar una prueba con un pulso Gaussiano con un 'chirpeo' lineal.
    Esto quiere decir, que el pulso Gaussiano que es de la forma:
        E(t) = A₀ * exp(-(t - t₀)² / 2*τ²) * exp(i * ( ω_0 * t + φ(t) ) )

    Tiene una fase φ(t) dada por:
        φ(t) = a·(t-t₀)²/τ²

    Donde 'a' es el parámetro de 'chirp'
    """

    # Parámetros de la medida
    numero_de_muestras = 6 * 4096
    duracion_temporal = 4 * 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(0, duracion_temporal, num=numero_de_muestras, retstep=True) # Vector de tiempos (centrado en cero). Guardamos la separación entre datos (inversa de la frecuencia de muestreo)
    # Construimos las frecuencias con la restricción del Tma Nyquist y la relación de reciprocidad.
    frecuencias = frecuencias_DFT(numero_de_muestras, Δt) # Nos lo devuelve en unidades de 1/ps (THz)
    ω = convertir(frecuencias, 'frecuencia', 'frecuencia angular') # Convertimos a rad / ps para utilizar la función DFT
    Δω = 2 * np.pi / (numero_de_muestras * Δt) # Relación de reciprocidad Δt Δω = 2π/N

    # -- Parámetros del pulso --
    t0 = 5 # # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    τ = 2 # Duración del pulso (ps)
    a = 3 * np.pi / 4 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ) # Chirpeo lineal

    """
    Comprobamos que se cumple el teorema de muestreo. Es decir, que la frecuencia de muestreo es mayor que 
    el doble de la frecuencia máxima de la señal.
    """

    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia máxima de la señal: {(convertir(ω_0 + 2 * a * duracion_temporal / (τ * τ), 'frecuencia angular', 'frecuencia'))} [THz]")
    if frecuencia_muestreo/2 > (convertir(ω_0 + 2 * a * duracion_temporal / (τ * τ), 'frecuencia angular', 'frecuencia')):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")

    """
    Podemos calcular la transformada analítica de un pulso Gaussiano con un 'chirpeo' lineal.
    Vendrá dada por:

        Ẽ(ω) = A₀ * τ sqrt(2π) / (sqrt(1 - 2 * i * a)) * exp(- (1j * (ω - ω_0) * (τ * τ  * (ω - ω_0) + (4 * a + 2 * i) * t₀)) / (4 * a + 2i) )
    """

    def transformada_pulso_chirpeado_linealmente(ω, t0, A, τ, ω_0, a):
        """
        Transformada analítica de un pulso Gaussiano con un 'chirpeo' lineal, es decir,
        que la fase venga dada por:
            φ(t) = a·(t-t₀)²/τ²

        Donde 'a' es el parámetro del chirp.

        La transformada vendrá dada por:
            Ẽ(ω) = A₀ * τ sqrt(2π) / (sqrt(1 - 2 * i * a)) * exp(- (1j * (ω - ω_0) * (τ * τ  * (ω - ω_0) + (4 * a + 2 * i) * t₀)) / (4 * a + 2i) )
        
        Las unidades han de ser consistentes entre ω, t0, τ y ω_0.

        Args:
            ω (np.array[float]): array de frecuencias en los que evaluar la transformada
            t0 (float): tiempo en el que la amplitud del pulso es máxima
            A (float): Amplitud del pulso
            τ (float): anchura del pulso
            ω_0 (float): frecuencia de la onda moduladora del pulso gaussiano (radianes / unidad de tiempo)
            a (np.array[float]): parámetro del chirp del pulso

        Devuelve:
            np.array: array de los valores de la transformada de Fourier en las frecuencias dadas
        """
        return A * τ * np.sqrt(2 * np.pi) / np.sqrt(1 - 2j * a) * np.exp(- (1j * (ω - ω_0) * (τ * τ  * (ω - ω_0) + (4 * a + 2 * 1j) * t0)) / (4 * a + 2j))


    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ) # Vector con el campo complejo del pulso

    transformada_analitica = transformada_pulso_chirpeado_linealmente(ω, t0, A, τ, ω_0, a) # Transformada analítica de un pulso gaussiano con fase constante
    transformada_numerica = DFT(pulso, t, Δt, ω, Δω) # Transformada numérica con el algoritmo de la transformada de Fourier discreta (DFT)

    """
    Vamos a ver las partes real e imaginaria y el valor absoluto en el dominio frecuencial.
    Incorporo un 'slider' al gráfico que nos permite interactuar con él y ver qué ocurre al
    cambiar el parámetro del chirp, 'a'.

    Hago zoom en el intervalo (192, 205) THz que es donde estará el pulso.

    Vamos a comprobar si se cumple la relación teórica de la anchura espectral para el pulso
    chirpeado linealmente. Será:
        Δν = 0.375 / τ * sqrt(1 + a²)

    /!\ Tengo que comprobar esta relación, la he cogido tal cual del pdf y puede que haya cosas
    que sean distintas, porque el resultado no está siendo el esperado

    Para comprobarlo, representaremos en el gráfico esta anchura visualmente, y veremos
    'a ojo' si coincide.
    """
    
    fig, ax = plt.subplots(3,1)

    def update_parametro(valor):
        """
        Función que actualiza los valores de la representación cuando se cambia el valor
        de 'a' utilizando el slider.
        """
        a = valor
        φ =  a * (t - t0) * (t - t0) / (τ * τ) # Chirpeo lineal

        pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ) # Vector con el campo complejo del pulso
        transformada_analitica = transformada_pulso_chirpeado_linealmente(ω, t0, A, τ, ω_0, a) # Transformada analítica de un pulso gaussiano con fase constante
        transformada_numerica = DFT(pulso, t, Δt, ω, Δω) # Transformada numérica con el algoritmo de la transformada de Fourier discreta (DFT)

        line0.set_ydata(np.abs(transformada_analitica))
        line1.set_ydata(np.abs(transformada_numerica))
        line2.set_ydata(np.real(transformada_analitica))
        line3.set_ydata(np.real(transformada_numerica))
        line4.set_ydata(np.imag(transformada_analitica))
        line5.set_ydata(np.imag(transformada_numerica))

        # Calculamos la anchura espectral a ver si se cumple la relacion:
        anchura_espectral_teorica = np.sqrt((4*a*a + 1) * 2 * np.log(2)) / τ # rad/unidad tiempo
        x_vals = [convertir(ω_0 - anchura_espectral_teorica, 'frecuencia angular', 'frecuencia'), convertir(ω_0 + anchura_espectral_teorica, 'frecuencia angular', 'frecuencia')]
        y_vals = [np.amax(np.abs(transformada_analitica)/2), np.amax(np.abs(transformada_analitica)/2)]
        line6.set_xdata(x_vals)
        line6.set_ydata(y_vals)
        line7.set_xdata(x_vals)
        line7.set_ydata(y_vals)

    slider_a = Slider(plt.axes([0.15, 0.025, 0.65, 0.03]), "Parámetro de chirp", -2*np.pi, 2*np.pi, valstep=0.01, valinit=a, initcolor='none')
    slider_a.on_changed(update_parametro)
    

    line0, = ax[0].plot(frecuencias, np.abs(transformada_analitica), '-', label='Transformada analítica')
    line1, = ax[0].plot(frecuencias, np.abs(transformada_numerica), '--', label='Transformada numérica')
    # Flecha de la anchura espectral
    anchura_espectral_teorica = np.sqrt((4*a*a + 1) * 2 * np.log(2)) / τ # rad unidad tiempo
    x_vals = [convertir(ω_0 - anchura_espectral_teorica, 'frecuencia angular', 'frecuencia'), convertir(ω_0 + anchura_espectral_teorica, 'frecuencia angular', 'frecuencia')]
    y_vals = [np.amax(np.abs(transformada_analitica)/2), np.amax(np.abs(transformada_analitica)/2)]
    line6, = ax[0].plot(x_vals, y_vals, '.', color='black')
    line7, = ax[0].plot(x_vals, y_vals, '--', color='black', label="Anchura a media altura")
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlim(185,205)
    ax[0].set_ylim(-6,6)

    line2, = ax[1].plot(frecuencias, np.real(transformada_analitica), '-', label='Transformada analítica')
    line3, = ax[1].plot(frecuencias, np.real(transformada_numerica), '--', label='Transformada numérica')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlim(185,205)
    ax[1].set_ylim(-6,6)

    line4, = ax[2].plot(frecuencias, np.imag(transformada_analitica), '-', label='Transformada analítica')
    line5, = ax[2].plot(frecuencias, np.imag(transformada_numerica), '--', label='Transformada numérica')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlim(185,205)
    ax[2].set_ylim(-6,6)

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    plt.show()