import numpy as np

def autocorrelacion_2orden(delays, Δt, funcion, tiempos, *args):
    """
    Calcula la autocorrelacion de segundo orden para una funcion dada.
        A⁽²⁾(τ) = ∫f(t)f(t-τ) dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        A⁽²⁾(τ) = ∑ⱼ₌₀ᴺ⁻¹ f(tⱼ)f(tⱼ - τ) Δt

    Para ello, deberemos especificar para qué valores de retrasos (delays)
    queremos evaluar la función, y el espaciado temporal de los valores de
    nuestra función, que dependerá del tiempo (y de otras variables).

    Para poder hacer esta función 'de propósito general' y poder utilizarla
    con diversas funciones, el único argumento requerido es el vector de tiempos
    donde se evaluará la función. Si la función sobre la que queremos calcular la
    autocorrelación depende de más parámetros, deberemos pasarlos como argumentos
    en el orden en el que aparecen en la definición de nuestra función, justo después
    del argumento del vector de tiempos.

    Por ejemplo si tenemos una función que depende de (t0, A, τ, ω_0, φ) además
    del tiempo, deberemos de llamar a esta función como:
        autocorrelacion_2orden(delays, Δt, funcion, tiempos, t0, A, τ, ω_0, φ)

    De esta manera internamente se podrá evaluar la función para los distintos retardos
    en los que queremos obtener el valor de la autocorrelación y así poder realizar el
    cálculo de la integral.
    
    Args:
        delays (np.ndarray): vector con los retardos en los que queremos evaluar la autocorrelación.
        Δt (float): espaciado del vector de tiempos
        tiempos (np.ndarray): vector de tiempos
        funcion (función): función sobre la que queremos calcular la autocorrelación
        *args : argumentos extra de la función (opcional).

    Returns:
        A_2: valores de la función de autocorrelación de segundo orden
    """
    A_2 = np.zeros(np.size(delays))

    valores = funcion(tiempos, *args)

    for i, τ in enumerate(delays):
        valores_retardo = funcion(tiempos - τ, *args)
        A_2[i] = trapecio(valores * valores_retardo, Δt)

    return A_2


def autocorrelacion_3orden(delays, Δt, funcion, tiempos, *args):
    """
    Calcula la autocorrelacion de segundo orden para una funcion dada.
        A⁽³⁾(τ) = ∫f(t)f²(t-τ) dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        A⁽³⁾(τ) = ∑ⱼ₌₀ᴺ⁻¹ f(tⱼ)f²(tⱼ - τ) Δt

    Para ello, deberemos especificar para qué valores de retrasos (delays)
    queremos evaluar la función, y el espaciado temporal de los valores de
    nuestra función, que dependerá del tiempo (y de otras variables).

    Para poder hacer esta función 'de propósito general' y poder utilizarla
    con diversas funciones, el único argumento requerido es el vector de tiempos
    donde se evaluará la función. Si la función sobre la que queremos calcular la
    autocorrelación depende de más parámetros, deberemos pasarlos como argumentos
    en el orden en el que aparecen en la definición de nuestra función, justo después
    del argumento del vector de tiempos.

    Por ejemplo si tenemos una función que depende de (t0, A, τ, ω_0, φ) además
    del tiempo, deberemos de llamar a esta función como:
        autocorrelacion_3orden(delays, Δt, funcion, tiempos, t0, A, τ, ω_0, φ)

    De esta manera internamente se podrá evaluar la función para los distintos retardos
    en los que queremos obtener el valor de la autocorrelación y así poder realizar el
    cálculo de la integral.
    
    Args:
        delays (np.ndarray): vector con los retardos en los que queremos evaluar la autocorrelación.
        Δt (float): espaciado del vector de tiempos
        tiempos (np.ndarray): vector de tiempos
        funcion (función): función sobre la que queremos calcular la autocorrelación
        *args : argumentos extra de la función (opcional).

    Returns:
        A_3: valores de la función de autocorrelación de tercer orden
    """
    A_3 = np.zeros(np.size(delays))

    valores = funcion(tiempos, *args)

    for i, τ in enumerate(delays):
        valores_retardo = funcion(tiempos - τ, *args)
        A_3[i] = trapecio(valores * valores_retardo**2, Δt)

    return A_3

def FRAC(delays, Δt, funcion, tiempos, *args):
    """
    Calcula la autocorrelacion "FRAC" (fringe-resolved autocorrelation):
        I_FRAC(τ) = ∫|[f(t) + f(t-τ)]²|² dt
    
    Utilizando integración numérica por el método del trapecio. Es decir:
        I_FRAC(τ) = ∑ⱼ₌₀ᴺ⁻¹ [f(tⱼ) + f(tⱼ-τ)]²|² Δt

    Para ello, deberemos especificar para qué valores de retrasos (delays)
    queremos evaluar la función, y el espaciado temporal de los valores de
    nuestra función, que dependerá del tiempo (y de otras variables).

    Para poder hacer esta función 'de propósito general' y poder utilizarla
    con diversas funciones, el único argumento requerido es el vector de tiempos
    donde se evaluará la función. Si la función sobre la que queremos calcular la
    autocorrelación depende de más parámetros, deberemos pasarlos como argumentos
    en el orden en el que aparecen en la definición de nuestra función, justo después
    del argumento del vector de tiempos.

    Por ejemplo si tenemos una función que depende de (t0, A, τ, ω_0, φ) además
    del tiempo, deberemos de llamar a esta función como:
        autocorrelacion_3orden(delays, Δt, funcion, tiempos, t0, A, τ, ω_0, φ)

    De esta manera internamente se podrá evaluar la función para los distintos retardos
    en los que queremos obtener el valor de la autocorrelación y así poder realizar el
    cálculo de la integral.
    
    Args:
        delays (np.ndarray): vector con los retardos en los que queremos evaluar la autocorrelación.
        Δt (float): espaciado del vector de tiempos
        tiempos (np.ndarray): vector de tiempos
        funcion (función): función sobre la que queremos calcular la autocorrelación
        *args : argumentos extra de la función (opcional).

    Returns:
        FRAC: valores de la función de autocorrelación "FRAC"
    """
    FRAC = np.zeros(np.size(delays))

    valores = funcion(tiempos, *args)

    for i, τ in enumerate(delays):
        valores_retardo = funcion(tiempos - τ, *args)
        FRAC[i] = trapecio(np.abs((valores + valores_retardo)**2)**2, Δt)

    return FRAC


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
