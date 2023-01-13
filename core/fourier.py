import numpy as np

def DFT(x):
    """
    Implementación de la transformada discreta de Fourier (DFT).

    La transformada de Fourier viene dada por la siguiente integral:
        F(ω) = ∫f(x)e^{-i 2π ω x} dx

    Que en el caso de tener datos discretos se transforma en un sumatorio:
        Fₙ = ∑₀ᴺ⁻¹ fₖ e^{-i 2π k n / N}

    Cada Fₙ será el resultado de la transformada para el dato fₖ

    El número de operaciones requerido es del orden de O(N²), por lo que para grandes cantidades de datos no es muy eficiente.

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada de fourier

    Devuelve:
        (np.ndarray[np.complex]): array de datos con la transformada de los datos
    """

    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    return np.dot(e,x)




def IDFT(x):
    """
    Implementación de la transformada discreta de Fourier inversa (IDFT).

    La transformada inversa de Fourier viene dada por la siguiente integral:
        f(x) = ∫F(ω)e^{i 2π ω x} dω

    Que en el caso de tener datos discretos se transforma en un sumatorio:
        fₖ = 1 / N · ∑₀ᴺ⁻¹ Fₙ e^{i 2π k n / N}

    Cada fₖ será el resultado de la transformada para el dato Fₙ

    El número de operaciones requerido es del orden de O(N²), por lo que para grandes cantidades de datos no es muy eficiente.

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada de fourier inversa

    Devuelve:
        np.ndarray[np.complex]: array de datos con la transformada inversa de los datos
    """

    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    
    return np.dot(e,x) / N




def fft(x):
    """
    Implementación de la transformada rápida de Fourier (FFT) mediante el algoritmo de Cooley-Tukey.

    La transformada discreta de Fourier viene dada por:
        Fₙ = ∑₀ᴺ⁻¹ fₖ e^{-i 2π k n / N}

    Cada Fₙ será el resultado de la transformada para el dato fₖ

    El número de operaciones requerido es del orden de O(N²), por lo que puede llegar a ser computacionalmente costoso.

    El algoritmo de Cooley-Tukey es un algoritmo con una complejidad de O(n log n) que consigue reducir el número de operaciones
    dividiendo los datos de entrada en dos secuencias más pequeñas y calculando sus DFT recursivamente.

    Este algoritmo requiere que los datos de entrada tengan una longitud que sea una potencia de 2.
    En el caso que no se cumpla esto, puede rellenarse el array con ceros hasta la siguiente potencia de 2.
    El problema de esto es que la longitud del array devuelto será la de la longitud de la siguiente potencia de 2.
    Una solución para conservar la longitud del array origianal es usar el algoritmo de la transformada de Bluestein.

    Esta función realiza el rellenado del array si es necesario y llama recursivamente a una función
    que contiene el cuerpo del algoritmo.

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada de fourier inversa

    Devuelve:
        np.ndarray[np.complex]: array de datos con la transformada inversa de los datos
    """
    n = x.size

    # Si el array no tiene una longitud igual a una potencia de dos se aplica el algoritmo de la transformada de bluestein
    if((n & (n-1) == 0) and n != 0) is not True:
        return transformada_bluestein(x, 1)

    return _fft_core(x, 1)




def ifft(x):
    """
    Implementación de la transformada rápida de Fourier inversa (IFFT).

    La transformada discreta de Fourier inversa viene dada por:
        fₖ = 1 / N · ∑₀ᴺ⁻¹ Fₙ e^{i 2π k n / N}

    Cada fₖ será el resultado de la transformada para el dato Fₙ

    El número de operaciones requerido es del orden de O(N²), por lo que puede llegar a ser computacionalmente costoso.

    El algoritmo de Cooley-Tukey es un algoritmo con una complejidad de O(n log n) que consigue reducir el número de operaciones
    dividiendo los datos de entrada en dos secuencias más pequeñas y calculando sus DFT recursivamente.

    Este algoritmo requiere que los datos de entrada tengan una longitud que sea una potencia de 2.
    En el caso que no se cumpla esto, puede rellenarse el array con ceros hasta la siguiente potencia de 2.
    El problema de esto es que la longitud del array devuelto será la de la longitud de la siguiente potencia de 2.
    Una solución para conservar la longitud del array origianal es usar el algoritmo de la transformada de Bluestein.

    Esta función realiza el rellenado del array si es necesario y llama recursivamente a una función
    que tiene el cuerpo del algoritmo.

    Finalmente, divide entre el número de muestras para dar el resultado final.

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada de fourier inversa

    Devuelve:
        np.ndarray[np.complex]: array de datos con la transformada inversa de los datos
    """
    n = x.size

    # Comprobacion de la longitud del array y añadir ceros si es necesario
    if((n & (n-1) == 0) and n != 0) is not True:
        return transformada_bluestein(x, -1) / n

    return (_fft_core(x, -1)/n)




def _fft_core(x, signo):
    """
    Cuerpo del algoritmo de la transformada rápida de Fourier.

    Esta función es llamada recursivamente para calcular la transformada de los índices pares
    e impares de los datos pasados como argumento.

    Una vez obtenidos se utilizan las ecuaciones del algoritmo de Cooley-Tukey que proporcionan
    los datos finales.

    Este algoritmo no es más que un algoritmo de multiplicación eficiente de polinomios.
    Más información sobre el funcionamiento del algoritmo en: https://www.youtube.com/watch?v=h7apO7q16V0

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada
        signo (float): signo del exponente de la transformada, que permite diferenciar en el caso de hacer la transformada o su inversa

    Devuelve:
        np.ndarray[np.complex]: array de datos con la transformada de los datos
    """
    n = x.size

    if n == 1:

        return x

    par, impar = _fft_core(x[::2], signo), _fft_core(x[1::2], signo)

    ω = np.exp(- signo * 2 * np.pi * 1j * np.arange(0, n//2) / n)

    y1 = par[0:n//2] + ω * impar[0:n//2]
    y2 = par[0:n//2] - ω * impar[0:n//2]


    return np.concatenate((y1, y2))



def transformada_bluestein(x, signo):
    """
    El algoritmo de Bluestein sirve para calcular la FFT de un array cuya longitud no es una potencia de 2. 
    El algoritmo de Bluestein consiste en los siguientes pasos:

        1) Rellenar el array de entrada con ceros para hacer que tenga como longitud una potencia de 2.
        2) Multiplicar el array rellenado por una secuencia especial de "chirp" para "desenrollar" la FFT.
        3) Calcular la FFT de el array resultante utilizando el algoritmo Cooley-Tukey.
        4) Multiplicar el array transformada por otra secuencia especial para "enrollar" de nuevo la DFT.

    Args:
        x (np.ndarray[np.complex]): array de datos para obtener su transformada
        signo (float): signo del exponente de la transformada, que permite diferenciar en el caso de hacer la transformada o su inversa

    Devuelve:
        np.ndarray[np.complex]: array de datos con la transformada de los datos
    """
    n = x.size
    m = 2**((n * 2).bit_length())
    
    coeficiente = - signo * np.pi / n
    exptable = np.exp(1j * (np.arange(n)**2 % (n * 2)) * coeficiente) # Secuencia de chirp

    a = np.concatenate((x * exptable, np.zeros(m - n)))

    b = np.concatenate((exptable, np.zeros(m - (n * 2 - 1)), np.flip(exptable)[:-1]))
    b = np.conjugate(b)


    c = convolucion(a, b)[:n]

    return c * exptable



def convolucion(x, y):
    """
    Calcula la convolución entre dos arrays de la misma longitud. 
    La convolución en el dominio temporal es igual al producto en
    el espacio de frecuencias, por lo que calculamos la fft de ambos
    arrays y luego multiplicamos sus elementos, para después calcular
    su transformada inversa.

    Args:
        x (np.ndarray[np.complex]): array de datos para calcular su convolución con y
        y (np.ndarray[np.complex]): array de datos para calcular su convolución con x

    Devuelve:
        np.ndarray[np.complex]: convolución entre los dos arrays
    """
    n = x.size

    x_transformada = fft(x)
    y_transformada = fft(y)

    z = ifft(x_transformada * y_transformada)

    return z


def convertir_tiempo_frecuencia(numero_de_muestras, Δt):
    """
    Crea un array de frecuencias angulares desde cero hasta la frecuencia máxima
    dada por el número de muestras y espaciado entre las muestras de un array
    que contiene los tiempos.

    Las unidades del array creado serán rad / unidad de tiempo del array de entrada.

    Args:
        numero_de_muestras (float): número de muestras temporales
        Δt (float): separación entre las muestras (inversa de la frecuencia de muestreo)

    Devuelve:
        np.ndarray: array con las frecuencias angulares (rad / unidad de tiempo)
    """
    return 2 * np.pi * np.arange(numero_de_muestras) / (Δt * numero_de_muestras)


def DFT_naive(E, t, Δt , ω, Δω):
    """
    Escogiendo como definición de la transformada directa:
        Ẽ(ω) = ∫E(t)e^{-i ω t} dt

    Y discretizándola, el coeficiente n-ésimo será:
        Ẽ(ωₙ) := Ẽₙ = ∑ⱼ₌₀ᴺ⁻¹ E(tⱼ) e^{-i ωₙ tⱼ} Δt

    Donde tⱼ será el j-esimo elemento del array de tiempos y ωₙ será el elemento
    n-ésimo del array de frecuencias.

    El array de tiempos será de la forma: tⱼ = t₀ + j·Δt con j = 0, ..., N - 1,
    donde N-1 es el número de muestras. El array de frecuencias será de la forma
    ωₙ = ω₀ + n·Δω con j = 0, ..., N - 1.

    Teniendo en cuenta la relación de reciprocidad, que establece que:
        Δt Δω = 2π/N

    Podemos sustituir en la expresión discretizada obtenida:
        Ẽₙ = ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{-i (ω₀ + n·Δω) (t₀ + j·Δt)} Δt =
           = Δt e^{-i n t₀ Δω} ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{-i ω₀ tⱼ} e^{-i n j Δω Δt} =
           = Δt e^{-i n t₀ Δω} ∑ⱼ₌₀ᴺ⁻¹ Eⱼ e^{-i ω₀ tⱼ} e^{-i 2π n j / N}

    Si definimos:
        rₙ = e^{-i n t₀ Δω}   ;   sⱼ = e^{-i ω₀ tⱼ}

    Podemos expresar finalmente la transformada discrezitada como:
        Ẽₙ = Δt·rₙ ∑ⱼ₌₀ᴺ⁻¹ Eⱼ·sⱼ e^{-i 2π n j / N}

    De manera que podemos denotar:
        DFTₙ = ∑ⱼ₌₀ᴺ⁻¹ Eⱼ' e^{-i 2π n j / N}

    (Donde Eⱼ' = Eⱼ·sⱼ)
    Al coeficiente n-ésimo de la transformada de Fourier discreta, y esta definición
    coincide con la empleada por numpy y scipy para el cálculo de la transformada directa.

    Por lo tanto, podremos emplear la transformada de Fourier rápida para calcular los coeficientes.
    Así, el array de coeficientes del espacio de frecuencia vendrá dado por:
        Ẽₙ = Δt·rₙ fft(Eⱼ·sⱼ)
    

    Args:
        E (np.ndarray[np.complex]): array de datos con la información temporal
        t (np.ndarray): array de tiempos equiespaciados Δt
        Δt (float): espaciado del array de tiempos
        ω (np.ndarray): array de frecuencias equiespaciadas Δω
        Δω (float): espaciado del array de frecuencias

    Devuelve:
        (np.ndarray[np.complex]): array de datos con la transformada de los datos en el intervalo
                                  de frecuencias especificado
    """

    r_n = np.exp(-1j * 2 * np.pi * np.arange(np.size(ω)) * t[0] * Δω)
    s_j = np.exp(-1j * 2 * np.pi * ω[0] * t)

    return Δt * r_n * fft(E * s_j)
    
def IDFT_naive(Ẽ, t, Δt, ω, Δω):
    """
    Escogiendo como definición de la transformada inversa:
        E(t) = 1/2π ∫Ẽ(ω)e^{i t ω} dω

    Y discretizándola, el coeficiente n-ésimo será:
        E(tⱼ) := Eⱼ = 1/2π · ∑ₙ₌₀ᴺ⁻¹ Ẽ(ωₙ) e^{i tⱼ ωₙ} Δω

    Donde tⱼ será el j-esimo elemento del array de tiempos y ωₙ será el elemento
    n-ésimo del array de frecuencias.

    El array de tiempos será de la forma: tⱼ = t₀ + j·Δt con j = 0, ..., N - 1,
    donde N-1 es el número de muestras. El array de frecuencias será de la forma
    ωₙ = ω₀ + n·Δω con j = 0, ..., N - 1.

    Teniendo en cuenta la relación de reciprocidad, que establece que:
        Δt Δω = 2π/N

    Podemos sustituir en la expresión discretizada obtenida:
        Eⱼ = 1/2π · ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{i (t₀ + j·Δt) (ω₀ + n·Δω)} Δω =
           = Δω/2π e^{i ω₀ tⱼ} ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{i n t₀ Δω} e^{i n j Δt Δω} =
           = Δω/2π e^{i ω₀ tⱼ} ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{i n t₀ Δω} e^{i 2π n j / N} =
           = 1/Δt e^{i ω₀ tⱼ} · 1/N ∑ₙ₌₀ᴺ⁻¹ Ẽₙ e^{i n t₀ Δω} e^{i 2π n j / N}

    Si definimos:
        rₙ = e^{-i n t₀ Δω}   ;   sⱼ = e^{-i ω₀ tⱼ}

    Podemos expresar finalmente la transformada inversa discrezitada como:
        Eⱼ = 1/Δt·sⱼ* · 1/N ∑ₙ₌₀ᴺ⁻¹ Ẽₙ·rₙ* e^{i 2π n j / N}

    De manera que podemos denotar:
        IDFTⱼ = 1/N ∑ₙ₌₀ᴺ⁻¹ Ẽₙ' e^{i 2π n j / N}

    (Donde Ẽₙ' = Ẽₙ·rₙ*)
    Al coeficiente n-ésimo de la transformada inversa de Fourier discreta, y esta definición
    coincide con la empleada por numpy y scipy para el cálculo de la transformada inversa.

    Por lo tanto, podremos emplear la transformada de Fourier rápida para calcular los coeficientes.
    Así, el array de coeficientes temporales vendrá dado por:
         Eⱼ = 1/Δt·sⱼ* · ifft(Ẽₙ·rₙ*)
    

    Args:
        Ẽ (np.ndarray[np.complex]): array de datos con la información frecuencial
        t (np.ndarray): array de tiempos equiespaciados Δt
        Δt (float): espaciado del array de tiempos
        ω (np.ndarray): array de frecuencias equiespaciadas Δω
        Δω (float): espaciado del array de frecuencias

    Devuelve:
        (np.ndarray[np.complex]): array con los coeficientes temporales
    """
    r_n_conj = np.exp(1j * 2 * np.pi * np.arange(np.size(ω)) * t[0] * Δω)
    s_j_conj = np.exp(1j * 2 * np.pi * ω[0] * t)

    return s_j_conj / Δt * ifft(Ẽ * r_n_conj)