import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

plt.rcParams.update({'font.size': 16}) # Tamaño de la fuente del plot

def pulso_gaussiano(t, τ, ω_0, φ):
    """
    Genera un pulso gaussiano dada su duración, frecuencia central y fase.
    Las unidades han de ser consistentes entre t, τ y ω_0.
    El pulso está normalizado.

    Un pulso gaussiano viene caracterizado por una envolvente en forma de gaussiana de expresión:

    E_envolvente = exp(-t² / 2*τ)

    Donde τ es la duración temporal del pulso, que está relacionada con el ancho de banda por la expresión:

    τ = FWHM / (2 * √log(2))

    FHWM es la anchura a media altura (full width half maximum).

    La envolvente viene modulada por un término exponencial complejo que depende de la frecuencia central de la onda,
    de manera que el pulso vendrá dado por el producto de la envolvente y esta exponencial, además del producto
    con la exponencial compleja que lleva la fase de la envolvente de la onda portadora.

    E(t) = E_envolvente * exp(i * ω_0 * t) * exp(i * φ(t)) = exp(-t² / 2*τ) * exp(i * ( ω_0 * t + φ(t) ) )

    Argumentos:
        t (float): vector de tiempos
        τ (float): anchura del pulso
        ω_0 (float): frecuencia central (radianes / unidad de tiempo)
        φ (float): fase de la envolvente de la onda portadora (rad)

    Devuelve:
        E_pulso (float): forma del pulso gaussiano en el tiempo especificado
    """

    return np.exp(-t*t / (2 * τ * τ)) * np.exp(1j * ( ω_0 * t + φ ))

if __name__ == '__main__':

    # -- Parámetros del pulso --
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 = np.array([np.pi / 4 for i in range(2000)]) # Fase (constante en este caso)
    τ = 1 # Duración del pulso (ps)

    t = np.linspace(-5, 5, num=2000) # Vector de tiempos (centrado en cero, ps)
    pulso = pulso_gaussiano(t, τ, ω_0, φ_0) # Vector con el campo complejo del pulso
    I = np.real(pulso) * np.real(pulso) + np.imag(pulso) * np.imag(pulso)

    # -- Plot 1: partes real e imaginaria del pulso--
    fig, ax = plt.subplots(2,1)

    # -- Parte real del pulso + envolvente --
    ax[0].plot(t, np.real(pulso), label = r'$\Re \{E(t)\}$')
    ax[0].plot(t, np.sqrt(I), '--', label = 'Envolvente')
    ax[0].plot(t, φ_0, '-.', label = r'$\phi (t)$')
    ax[0].set_xlabel("Tiempo (ps)")
    ax[0].set_ylabel("Campo / Envolvente (Normalizado)")
    ax[0].grid()
    ax[0].legend()


    # -- Parte imaginaria del pulso + envolvente --
    ax[1].plot(t, np.imag(pulso), label = r'$\Im \{E(t)\}$')
    ax[1].plot(t, np.sqrt(I), '--', label = 'Envolvente')
    ax[1].plot(t, φ_0, '-.', label = r'$\phi (t)$')
    ax[1].set_xlabel("Tiempo (ps)")
    ax[1].set_ylabel("Campo / Envolvente (Normalizado)")
    ax[1].grid()
    ax[1].legend()

    # -- Plot 2: intensidad --
    fig1, ax1 = plt.subplots()
    ax1.plot(t, I)
    ax1.set_xlabel("Tiempo (ps)")
    ax1.set_ylabel("Intensidad")
    ax1.grid()

    plt.show()