import numpy as np
from .utiles import pulso_gaussiano

def trapecio(func_vals, h):
    """
    Integración numérica por el método del trapecio

    Argumentoss:
        func_vals (np.array): valores de la función a integrar
        h (float): paso numérico de la integración.

    Devuelve:
        integral: valor numérico calculado
    """

    weights = 2*np.ones(np.size(func_vals))
    weights[0] = 1
    weights[-1] = 1

    integral = h/2 * (weights@func_vals)

    return integral

def autocorrelacion_2orden(delays, Δt, tiempos, t0, A, τ, ω_0, φ):
    """
    Calcula la autocorrelación de segundo orden de la función introducida.
    Pero esto es una movida que te cagas porque habría de pasar mogollón de argumentos que podría
    requerir la función a calcular, (sí, estoy pensando en la turra del pulso gaussiano).
    Puedo hacer un alarde de programación y usar los kwargs! Es mucha movida, ¿no?
    Sí, porque la función debería estar construida de forma explícita en que el delay se le pudiera meter
    en el argumento i-ésimo siempre. Esto para mi que lo estoy usando pues de puta madre pero puede 
    ser confuso si lo usan otros. La funcii

    Args:
        delays (_type_): _description_
        tiempos (_type_): _description_
        funcion (_type_): _description_

    Returns:
        _type_: _description_
    """
    A_2 = np.zeros(np.size(delays))

    pulso = pulso_gaussiano(tiempos, t0, A, τ, ω_0, φ)
    I = np.abs(pulso)**2

    for i, delay in enumerate(delays):
        pulso_retardado = pulso_gaussiano(tiempos, t0 - delay, A, τ, ω_0, φ)
        I_retardado = np.abs(pulso_retardado)**2
        A_2[i] = trapecio(I*I_retardado, Δt)

    return A_2

    def autocorrelacion_2(delays, Δt, tiempos, funcion):
        """
        Calcula la autocorrelación de segundo orden de la función introducida.
        Pero esto es una movida que te cagas porque habría de pasar mogollón de argumentos que podría
        requerir la función a calcular, (sí, estoy pensando en la turra del pulso gaussiano).
        Puedo hacer un alarde de programación y usar los kwargs! Es mucha movida, ¿no?
        Sí, porque la función debería estar construida de forma explícita en que el delay se le pudiera meter
        en el argumento i-ésimo siempre. Esto para mi que lo estoy usando pues de puta madre pero puede 
        ser confuso si lo usan otros. La funcii

        Args:
            delays (_type_): _description_
            tiempos (_type_): _description_
            funcion (_type_): _description_

        Returns:
            _type_: _description_
        """
        A_2 = np.zeros(np.size(delays))

        pulso = pulso_gaussiano(tiempos, t0, A, τ, ω_0, φ)
        I = np.abs(pulso)**2

        for i, delay in enumerate(delays):
            pulso_retardado = pulso_gaussiano(tiempos, t0 - delay, A, τ, ω_0, φ)
            I_retardado = np.abs(pulso_retardado)**2
            A_2[i] = trapecio(I*I_retardado, Δt)

    return A_2


def autocorrelacion_3orden(τ_array, I):
    pass

def FRAC(τ_array):
    pass