import numpy as np

def pulso_gaussiano(t, t0, A, τ, ω_0, φ):
    """
    Genera un pulso gaussiano dada su duración, frecuencia central y fase.
    Las unidades han de ser consistentes entre t, t0, τ y ω_0.
    El pulso está normalizado.

    Un pulso gaussiano viene caracterizado por una envolvente en forma de gaussiana de expresión:

    E_envolvente = A * exp(-(t - t₀)² / 2*τ²)

    Donde t₀ es el instante temporal en el que la amplitud del pulso es máxima, y τ es la duración 
    temporal del pulso, que está relacionada con el ancho de banda por la expresión:

    FWHM = 2 * √log(2) * τ

    FHWM es la anchura a media altura (full width half maximum).

    La envolvente viene modulada por un término exponencial complejo que depende de la frecuencia central de la onda,
    de manera que el pulso vendrá dado por el producto de la envolvente y esta exponeFourier Transform of Gaussian Modulated Functionncial, además del producto
    con la exponencial compleja que lleva la fase de la envolvente de la onda portadora.

    E(t) = E_envolvente * exp(i * ω_0 * t) * exp(i * φ(t)) = A * exp(-(t - t₀)² / 2*τ) * exp(i * ( ω_0 * t + φ(t) ) )

    Argumentos:
        t (np.array[float]): vector de tiempos
        t0 (float): tiempo en el que la amplitud del pulso es máxima
        A (float): amplitud del pulso
        τ (float): anchura del pulso
        ω_0 (float): frecuencia de la moduladora del pulso gaussiano (radianes / unidad de tiempo)
        φ (np.array[float]): fase de la envolvente de la onda portadora (rad)

    Devuelve:
        E_pulso (float): forma del pulso gaussiano en el tiempo especificado
    """

    return A * np.exp(-(t-t0)*(t-t0) / (2 * τ * τ)) * np.exp(1j * ( ω_0 * t + φ ))



def transformada_pulso_gaussiano(ω, t0, A, τ, ω_0, φ):
    """
    Calcula la transformada de Fourier analítica de un pulso gaussiano con una onda moduladora y fase constante.

    El pulso viene dado por:
        E(t) = A * exp(-(t - t₀)² / 2*τ²) * exp(i * ( ω_0 * t + φ) )

    Su transformada de Fourier será:
        Ẽ(ω) = A  * sqrt(2π * τ²) * exp(i * (t₀ * (ω₀ - ω) +  φ)) * exp(-(ω - ω₀)²  / (2 * τ²))

    Las unidades han de ser consistentes entre ω, t0, τ y ω_0.

    Args:
        ω (np.array[float]): array de frecuencias en los que evaluar la transformada
        t0 (float): tiempo en el que la amplitud del pulso es máxima
        A (float): Amplitud del pulso
        τ (float): anchura del pulso
        ω_0 (float): frecuencia de la onda moduladora del pulso gaussiano (radianes / unidad de tiempo)
        φ (np.array[float]): fase de la envolvente de la onda portadora (rad) [cte]

    Devuelve:
        np.array: array de los valores de la transformada de Fourier en las frecuencias dadas
    """

    return A * τ * np.sqrt(2 * np.pi) * np.exp(1j * (φ + t0 * (ω_0 - ω)) - (ω - ω_0)*(ω - ω_0) / (2 * τ * τ))