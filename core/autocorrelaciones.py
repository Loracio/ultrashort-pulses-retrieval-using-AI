"""
En este módulo se encuentran las diferentes funciones utilizadas para calcular
autocorrelaciones entre pulsos, además de la función que calcula la traza de un
pulso.

Incluye:
    - trapecio : realiza integración numérica por el método del trapecio.

    - autocorrelacion_2orden : calcula la autocorrelacion de segundo orden
                               desplazando los valores de la función pasada

    - autocorrelacion_3orden : calcula la autocorrelacion de tercer orden
                               desplazando los valores de la función pasada

    - autocorrelacion_interferometrica : calcula la autocorrelacion interferométrica
                               desplazando los valores de la función pasada

    - traza : calcula la traza de un pulso.
"""

import numpy as np
from .fourier import *
from .unidades import *

def trapecio(func_vals, h):
    """
    Integración numérica por el método del trapecio.

    Argumentoss:
        func_vals (np.ndarray): valores de la función a integrar.
        h (float): paso numérico de la integración.

    Devuelve:
        integral: valor numérico calculado
    """

    weights = 2 * np.ones(np.size(func_vals))
    weights[0] = 1
    weights[-1] = 1

    integral = h/2 * np.dot(weights,func_vals)

    return integral


def autocorrelacion_2orden(valores, N, Δt):
    """
    Calcula la autocorrelacion de segundo orden para una funcion dada.
        A⁽²⁾(τ) = ∫f(t)f(t-τ) dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        A⁽²⁾(τ) = ∑ⱼ₌₀ᴺ⁻¹ f(tⱼ)f(tⱼ - τ) Δt

    Donde τ tomará los valores desde -(N - 1)·Δt hasta +(N - 1)·Δt con un espaciado Δt.

    De esta manera, lo que haremos será guardar los valores del integrando multiplicando
    la señal en cada instante de tiempo por sí misma desplazada cierta cantidad. Es decir,
    multiplicaremos por un elemento del array movido cierta cantidad de elementos hacia la
    derecha o la izquierda.

    La integración se realiza usando el método del trapecio.

    Args:
        valores (np.ndarray): vector con los valores de la función sobre la que calcular la autocorrelación
        N (int): número de muestras
        Δt (float): espaciado del vector de tiempos

    Returns:
        A_2: valores de la función de autocorrelación de segundo orden
    """
    A_2 = np.zeros(2 * N - 1, dtype=float)
    valores_integrando = np.zeros(N, dtype=valores.dtype)

    for τ in range(N):
        valores_integrando[:τ + 1] = valores[:τ + 1] * valores[N - τ - 1:]
        valores_integrando[τ + 1:] = 0
        A_2[τ] = trapecio(valores_integrando, Δt)

    for τ in range(N - 1):
        valores_integrando[:τ + 1] = 0
        valores_integrando[τ + 1:] = valores[τ + 1:] * valores[: N - τ - 1]
        A_2[N + τ] = trapecio(valores_integrando, Δt)
    
    return A_2

def autocorrelacion_3orden(valores, N, Δt):
    """
    Calcula la autocorrelacion de tercer orden para una funcion dada.    
        A⁽³⁾(τ) = ∫f(t)f²(t-τ) dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        A⁽³⁾(τ) = ∑ⱼ₌₀ᴺ⁻¹ f(tⱼ)f²(tⱼ - τ) Δt

    Donde τ tomará los valores desde -(N - 1)·Δt hasta +(N - 1)·Δt con un espaciado Δt.

    De esta manera, lo que haremos será guardar los valores del integrando multiplicando
    la señal en cada instante de tiempo por sí misma desplazada cierta cantidad. Es decir,
    multiplicaremos por un elemento del array movido cierta cantidad de elementos hacia la
    derecha o la izquierda.

    La integración se realiza usando el método del trapecio.

    Args:
        valores (np.ndarray): vector con los valores de la función sobre la que calcular la autocorrelación
        N (int): número de muestras
        Δt (float): espaciado del vector de tiempos

    Returns:
        A_3: valores de la función de autocorrelación de tercer orden
    """
    A_3 = np.zeros(2 * N - 1, dtype=float)
    valores_integrando = np.zeros(N, dtype=valores.dtype)

    for τ in range(N):
        valores_integrando[:τ + 1] = valores[:τ + 1] * valores[N - τ - 1:]**2
        valores_integrando[τ + 1:] = 0
        A_3[τ] = trapecio(valores_integrando, Δt)

    for τ in range(N - 1):
        valores_integrando[:τ + 1] = 0
        valores_integrando[τ + 1:] = valores[τ + 1:] * valores[: N - τ - 1]**2
        A_3[N + τ] = trapecio(valores_integrando, Δt)

    return A_3

def autocorrelacion_interferometrica(valores, N, Δt):
    """
    Calcula la autocorrelacion "FRAC" (fringe-resolved autocorrelation):
        IA(τ) = ∫|[f(t) + f(t-τ)]²|² dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        IA(τ) = ∑ⱼ₌₀ᴺ⁻¹ [f(tⱼ) + f(tⱼ-τ)]²|² Δt

    Donde τ tomará los valores desde -(N - 1)·Δt hasta +(N - 1)·Δt con un espaciado Δt.

    De esta manera, lo que haremos será guardar los valores del integrando multiplicando
    la señal en cada instante de tiempo por sí misma desplazada cierta cantidad. Es decir,
    multiplicaremos por un elemento del array movido cierta cantidad de elementos hacia la
    derecha o la izquierda.

    La integración se realiza usando el método del trapecio.
    
    Args:
        valores (np.ndarray): vector con los valores de la función sobre la que calcular la autocorrelación
        N (int): número de muestras
        Δt (float): espaciado del vector de tiempos

    Returns:
        IA: valores de la autocorrelación interferométrica (IA)
    """
    IA = np.zeros(2 * N - 1, dtype=float)
    
    valores_extendidos = np.zeros(2 * N - 1, dtype=valores.dtype)
    valores_desplazados = np.zeros(2 * N - 1, dtype=valores.dtype)

    valores_extendidos[N - 1 : 2 * N - 1] = valores
    valores_desplazados[0 : N] = valores

    for τ in range(2 * N - 1):
        valores_desplazados = np.roll(valores_desplazados, 1)
        IA[τ] = trapecio(np.abs((valores_extendidos + valores_desplazados)**2)**2, Δt)


    return IA


def traza(E, t, Δt, N):
    """
    Calcula la traza dada por:

        T(ω, τ) =  | ∫ E(t)E(t - τ) exp(-i ω t) dt |²

    Es decir:
        T(ω, τ) = |FT[E(t)E(t - τ)]|²

    Así, usaremos el algoritmo de la transformada discreta de Fourier para obtenerla.

    Para un vector de tiempos dado, el vector de retrasos (delays) se construye como
    un vector de retrasos equiespaciado Δτ = Δt desde -(N-1)·Δt hasta +(N-1)·Δt.

    El vector de frecuencias viene dado por las restricciones del Tma de Nyquist,
    desde -fₘ/2 hasta +fₘ/2, donde fₘ es la frecuencia de muestreo de la señal, 
    dada por fₘ = 1 / Δt.

    Así, para cada valor de τ podemos obtener los valores para cada una de estas frecuencias
    a partir del cálculo de la DFT, y construimos una malla (2N-1)xN donde se almacenan cada uno
    de los valores correspondientes a un valor de ω y τ.

    Se guardan en la malla T y se devuelven como resultado de la función.

    Argumentos:
        E (np.array): array con el campo complejo del pulso
        t (np.array): vector de tiempos
        Δt (float): espaciado entre tiempos en el vector de tiempos
        N (int): número de muestras temporales

    Devuelve:
        T (np.meshgrid): valores de la traza en cada uno de los puntos de la malla (ω, τ)
    """
    ω = convertir(frecuencias_DFT(N, Δt), 'frecuencia', 'frecuencia angular')
    Δω = 2 * np.pi / (N * Δt) # Relación de reciprocidad Δt Δω = 2π/N

    T = np.zeros((2*N - 1, N), dtype=np.float64)

    # Calculamos el valor de E(t)*E(t - τ) y lo pasamos como argumento para hacer la transformada de Fourier
    valores_integrando = np.zeros(N, dtype=E.dtype)

    for τ in range(N):
        valores_integrando[:τ + 1] = E[:τ + 1] * E[N - τ - 1:]
        valores_integrando[τ + 1:] = 0
        T[τ][:] = np.abs(DFT(valores_integrando, t, Δt, ω, Δω))**2

    for τ in range(N - 1):
        valores_integrando[:τ + 1] = 0
        valores_integrando[τ + 1:] = E[τ + 1:] * E[: N - τ - 1]
        T[N + τ][:] = np.abs(DFT(valores_integrando, t, Δt, ω, Δω))**2

    return T