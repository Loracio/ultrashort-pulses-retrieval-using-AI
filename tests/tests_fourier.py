import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import * 

plt.rcParams.update({'font.size': 16}) # Tamaño de la fuente del plot


if __name__ == '__main__':

    # Parámetros de la medida
    numero_de_muestras = 4096
    duracion_temporal = 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(0, duracion_temporal, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)
    # Construimos las frecuencias con la restricción del Tma Nyquist y la relación de reciprocidad.
    frecuencias = frecuencias_DFT(numero_de_muestras, Δt) # Nos lo devuelve en unidades de 1/ps (THz)
    ω = convertir(frecuencias, 'frecuencia', 'frecuencia angular') # Convertimos a rad / ps para utilizar la función DFT
    Δω = 2 * np.pi / (numero_de_muestras * Δt) # Relación de reciprocidad Δt Δω = 2π/N


    # -- Parámetros del pulso --
    t0 = 5 # # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 # Fase del pulso (constante en este caso)
    τ = 1 # Duración del pulso (ps)

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")


    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ_0) # Vector con el campo complejo del pulso
    
    transformada_analitica = transformada_pulso_gaussiano(ω, t0, A, τ, ω_0, φ_0) # Transformada analítica de un pulso gaussiano con fase constante
    transformada_numerica = DFT(pulso, t, Δt, ω, Δω) # Transformada numérica con el algoritmo de la transformada de Fourier discreta (DFT)

    """
    Vamos a comprobar si coinciden la transformada obtenida analíticamente con la obtenida teóricamente.
    Para ello, representamos las partes real e imaginaria de ambos pulsos en el dominio de frecuencias.
    La transformada analítica se representa con trazo continuo, y con trazo discontinuo la obtenida
    numéricamente.
    """

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Comparación transformada analítica y numérica')

    ax[0].plot(frecuencias, np.abs(transformada_analitica), '-', label='Transformada analítica')
    ax[0].plot(frecuencias, np.abs(transformada_numerica), '--', label='Transformada numérica')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(frecuencias, np.real(transformada_analitica), '-', label='Transformada analítica')
    ax[1].plot(frecuencias, np.real(transformada_numerica), '--', label='Transformada numérica')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(frecuencias, np.imag(transformada_analitica), '-', label='Transformada analítica')
    ax[2].plot(frecuencias, np.imag(transformada_numerica), '--', label='Transformada numérica')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    # plt.show()

    """
    Vemos que todo coincide perfectamente, pero la resolución no es muy buena. ¿Podemos mejorarla?
    Por la relación de reciprocidad, Δω = 2π/(NΔt), así que si mantenemos fijo Δt y aumentamos el 
    número de muestras, tendremos una mayor resolución. Para mantener fijo Δt, deberemos aumentar
    a la misma razón el número de muestras que la duración temporal, ya que:
        Δt = duración_temporal / numero_de_muestras

    Así, volvemos a definir los pulsos aumentando la duración temporal y el número de muestras por
    cierto factor y al representar en el dominio de las frecuencias, obtenemos una mayor resolución.

    Hacemos zoom en el espectro en la zona donde se situa el máximo para ver el aumento de resolución.
    """

    # Aumentamos en un factor de 10 la resolución:
    numero_de_muestras_2 = 10 * 4096
    duracion_temporal_2 = 10 * 10 

    t_2, Δt_2 = np.linspace(0, duracion_temporal_2, num=numero_de_muestras_2, retstep=True)
    pulso_2 = pulso_gaussiano(t_2, t0, A, τ, ω_0, φ_0)

    # Construimos las frecuencias con la restricción del Tma Nyquist y la relación de reciprocidad.
    frecuencias_2 = frecuencias_DFT(numero_de_muestras_2, Δt_2)
    ω_2 = convertir(frecuencias_2, 'frecuencia', 'frecuencia angular') 
    Δω_2 = 2 * np.pi / (numero_de_muestras_2 * Δt_2) # Relación de reciprocidad Δt Δω = 2π/N

    transformada_analitica_2 = transformada_pulso_gaussiano(ω_2, t0, A, τ, ω_0, φ_0) # Transformada analítica de un pulso gaussiano con fase constante
    transformada_numerica_2 = DFT(pulso_2, t_2, Δt_2, ω_2, Δω_2) # Transformada numérica con el algoritmo de la transformada de Fourier discreta (DFT)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Comparación transformada analítica y numérica, con menor Δω')

    ax[0].plot(frecuencias_2, np.abs(transformada_analitica_2), '-', label='Transformada analítica')
    ax[0].plot(frecuencias_2, np.abs(transformada_numerica_2), '.', label='Transformada numérica')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_xlim(192,195)

    ax[1].plot(frecuencias_2, np.real(transformada_analitica_2), '-', label='Transformada analítica')
    ax[1].plot(frecuencias_2, np.real(transformada_numerica_2), '.', label='Transformada numérica')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_xlim(192,195)

    ax[2].plot(frecuencias_2, np.imag(transformada_analitica_2), '-', label='Transformada analítica')
    ax[2].plot(frecuencias_2, np.imag(transformada_numerica_2), '.', label='Transformada numérica')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xlim(192,195)

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    # plt.show()


    """
    Con la transformada directa todo parece ir bien. Realicemos la transformada inversa de la transformada
    analítica y comprobemos si coincide con el pulso original.

    Lo hacemos para los dos casos anteriores, con mayor resolución espectral y con la original.
    """

    señal_recuperada = IDFT(transformada_analitica, t, Δt, ω, Δω)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Recuperación del pulso original haciendo IDFT(transformada_analitica)')

    ax[0].plot(t, np.abs(pulso), '-', label='Pulso original')
    ax[0].plot(t, np.abs(señal_recuperada), '--', label='Pulso recuperado')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, np.real(pulso), '-', label='Pulso original')
    ax[1].plot(t, np.real(señal_recuperada), '--', label='Pulso recuperado')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, np.imag(pulso), '-', label='Pulso original')
    ax[2].plot(t, np.imag(señal_recuperada), '--', label='Pulso recuperado')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    # plt.show()


    señal_recuperada_2 = IDFT(transformada_analitica_2, t_2, Δt_2, ω_2, Δω_2)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Recuperación del pulso original haciendo IDFT(transformada_analitica) con menor Δω')

    ax[0].plot(t_2, np.abs(pulso_2), '-', label='Pulso original')
    ax[0].plot(t_2, np.abs(señal_recuperada_2), '--', label='Pulso recuperado')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t_2, np.real(pulso_2), '-', label='Pulso original')
    ax[1].plot(t_2, np.real(señal_recuperada_2), '--', label='Pulso recuperado')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t_2, np.imag(pulso_2), '-', label='Pulso original')
    ax[2].plot(t_2, np.imag(señal_recuperada_2), '--', label='Pulso recuperado')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')
    
    # plt.show()

    """
    Se cumple todo a la perfección. Comprobemos que también cuando se realiza IDF(DFT(pulso)).
    """

    pulso_recuperado = IDFT(transformada_numerica, t, Δt, ω, Δω)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Recuperación del pulso original haciendo IDFT(DFT(pulso))')

    ax[0].plot(t, np.abs(pulso), '-', label='Pulso original')
    ax[0].plot(t, np.abs(pulso_recuperado), '--', label='IDFT(DFT(pulso))')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, np.real(pulso), '-', label='Pulso original')
    ax[1].plot(t, np.real(pulso_recuperado), '--', label='IDFT(DFT(pulso))')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, np.imag(pulso), '-', label='Pulso original')
    ax[2].plot(t, np.imag(pulso_recuperado), '--', label='IDFT(DFT(pulso))')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    # plt.show()

    """
    Uno de los problemas que presenta trabajar con las funciones fft de numpy y scipy sin realizar
    shifteos es que si tenemos el pulso centrado en cero en el dominio temporal, no se recupera bien.

    Vamos a hacer una prueba de un pulso centrado en cero y comprobar resultados entre numpy y scipy.
    Tendremos que multiplicar la fft de numpy por Δt para que nos de la misma escala en amplitudes
    y poder comparar visualmente.
    """
    numero_de_muestras = 10 * 4096
    duracion_temporal = 10 * 10 

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True)
    pulso_centrado = pulso_gaussiano(t, 0, A, τ, ω_0, φ_0)

    frecuencias = frecuencias_DFT(numero_de_muestras, Δt) 
    ω = convertir(frecuencias, 'frecuencia', 'frecuencia angular')
    Δω = 2 * np.pi / (numero_de_muestras * Δt)

    transformada_pulso_centrado = DFT(pulso_centrado, t, Δt, ω, Δω)
    transformada_analitica_pulso_centrado = transformada_pulso_gaussiano(ω, 0, A, τ, ω_0, φ_0)
    transformada_numpy = np.fft.fft(pulso_centrado) * Δt

    fig, ax = plt.subplots(3,1)

    fig.suptitle("Comparación DFT con pulso centrado en cero. La transformada de numpy tiene un desfase.")

    ax[0].plot(frecuencias, np.abs(transformada_analitica_pulso_centrado), '-', label='Transformada analítica')
    ax[0].plot(frecuencias, np.abs(transformada_pulso_centrado), '--', label='Transformada numérica')
    ax[0].plot(frecuencias, np.abs(transformada_numpy), '--', label='Transformada numpy')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(frecuencias_2, np.real(transformada_analitica_pulso_centrado), '-', label='Transformada analítica')
    ax[1].plot(frecuencias_2, np.real(transformada_pulso_centrado), '--', label='Transformada numérica')
    ax[1].plot(frecuencias_2, np.real(transformada_numpy), '--', label='Transformada numpy')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(frecuencias_2, np.imag(transformada_analitica_pulso_centrado), '-', label='Transformada analítica')
    ax[2].plot(frecuencias_2, np.imag(transformada_pulso_centrado), '--', label='Transformada numérica')
    ax[2].plot(frecuencias_2, np.imag(transformada_numpy), '--', label='Transformada numpy')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    # plt.show()

    """
    Vemos pues, que la fft de numpy al no estar multiplicada por los factores de fase, no reproduce bien los resultados.
    """
    
    """
    Por último, vamos a ver qué ocurre cuando NO satisfacemos la relación de reciprocidad. Como nos interesa ver la región del pico de
    frecuencias, vamos a centrarnos en la región de 190 THz hasta 200 THz, construyendo el array de frecuencias partiendo desde
    ω₀ = 2π * 190 hasta ωₘₐₓ = 2π * 200. De esta manera,  Δω = (ω_max - ω_min) / numero_de_muestras y será menor que el impuesto
    por la relación de reciprocidad.

    Definimos la función de DFT_naive que realiza la transformada con la forma más simplificada de la discretización de la integral
    de la transformada directa. Asimismo, definimos IDFT_naive como una función que realiza la transformada inversa con la discretización
    más simple posible.

    Al no cumplirse la relación de reciprocidad, no estaremos recuperando bien el pulso.
    """

    def DFT_naive(E, t, Δt, ω, Δω):
        """
        Utiliza la versión discretizada de la transformada de Fourier directa más básica:
            Ẽ(ωₙ) := Ẽₙ = ∑ⱼ₌₀ᴺ⁻¹ E(tⱼ) e^{-i ωₙ tⱼ} Δt
        
        Y devuelve el array de coeficientes. Notar que puede ser llamada con cualquier ω, Δω.
        """
        N = E.size
        result = np.zeros(N, dtype=complex)
        for i in range(N):
            for j in range(N):
                result[i] += E[j] * np.exp(-1j * ω[i] * t[j]) 
            result[i] *= Δt
        return result 

    numero_de_muestras =  4096
    duracion_temporal = 10 

    t, Δt = np.linspace(0, duracion_temporal, num=numero_de_muestras, retstep=True)
    pulso = pulso_gaussiano(t, t0, A, τ, ω_0, φ_0)

    ω_min = 2 * np.pi * 190
    ω_max = 2 * np.pi * 200 
    Δω = (ω_max - ω_min) / numero_de_muestras
    print(f"Δω={Δω}, relación reciprocidad = {2 * np.pi / (numero_de_muestras * Δt)}")
    ω = ω_min + Δω * np.arange(numero_de_muestras)
    frecuencias = convertir(ω, 'frecuencia angular', 'frecuencia')

    transformada_numerica = DFT_naive(pulso, t, Δt, ω, Δω)
    transformada_analitica = transformada_pulso_gaussiano(ω, t0, A, τ, ω_0, φ_0)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('No se cumple Δt Δω = 2π/N')

    ax[0].plot(frecuencias, np.abs(transformada_analitica), '-', label='Transformada analítica')
    ax[0].plot(frecuencias, np.abs(transformada_numerica), '.', label='Transformada numérica')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(frecuencias, np.real(transformada_analitica), '-', label='Transformada analítica')
    ax[1].plot(frecuencias, np.real(transformada_numerica), '.', label='Transformada numérica')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(frecuencias, np.imag(transformada_analitica), '-', label='Transformada analítica')
    ax[2].plot(frecuencias, np.imag(transformada_numerica), '.', label='Transformada numérica')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')


    def IDFT_naive(Ẽ, t, Δt, ω, Δω):
        """
        Utiliza la versión discretizada de la transformada de Fourier inversa más básica:
            E(tⱼ) := Eⱼ = 1/2π · ∑ₙ₌₀ᴺ⁻¹ Ẽ(ωₙ) e^{i tⱼ ωₙ} Δω
        
        Y devuelve el array de coeficientes. Notar que puede ser llamada con cualquier ω, Δω.
        """
        N = Ẽ.size
        result = np.zeros(N, dtype=complex)
        for i in range(N):
            for j in range(N):
                result[i] += Ẽ[j] * np.exp(1j * ω[i] * t[j]) 
            result[i] *= Δω/(2*np.pi)
        return result 

    señal_recuperada = IDFT_naive(transformada_numerica, t, Δt, ω, Δω)

    fig, ax = plt.subplots(3,1)

    fig.suptitle('Recuperación del pulso original sin cumplir relación reciprocidad')

    ax[0].plot(t, np.abs(pulso), '-', label='Pulso original')
    ax[0].plot(t, np.abs(señal_recuperada), '--', label='Pulso recuperado')
    ax[0].set_title('Valor absoluto del coeficiente')
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, np.real(pulso), '-', label='Pulso original')
    ax[1].plot(t, np.real(señal_recuperada), '--', label='Pulso recuperado')
    ax[1].set_title('Parte real')
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t, np.imag(pulso), '-', label='Pulso original')
    ax[2].plot(t, np.imag(señal_recuperada), '--', label='Pulso recuperado')
    ax[2].set_title('Parte imaginaria')
    ax[2].grid()
    ax[2].legend()

    ax[2].set_xlabel('Frecuencia (THZ)')
    ax[1].set_ylabel('Amplitud coeficiente')

    """
    Vemos que no se recupera bien el pulso al escoger esa pequeña ventana de frecuencias. Si
    escogemos las frecuencias con el espaciado correcto, cumpliendo la relación de reciprocidad
    y el teorema de muestreo, el resultado al recuperar la señal debería ser el correcto.
    """
    plt.show()