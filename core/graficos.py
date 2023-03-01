import numpy as np
import matplotlib.pyplot as plt
from .autocorrelaciones import espectrograma
from .utiles import media

def plot_real_imag(t, pulso, φ=None):
    """
    Realiza una representación de las partes real e imaginaria del pulso pasado

    Args:
        t (float, np.array): array de tiempos
        pulso (np.ndarray[np.complex]): array con los valores del pulso en un tiempo t
        φ (float, np.array): array de la fase en un tiempo t

    Devuelve:
        tuple(matplotlib Figure, matplotlib Axis)
    """
    envolvente = np.abs(pulso)

    fig, ax = plt.subplots(2,1)

    # -- Parte real del pulso + envolvente --
    ax[0].plot(t, np.real(pulso), label = r'$\Re \{E(t)\}$')
    ax[0].plot(t, envolvente, '--', label = 'Envolvente')

    if φ is not None:
        ax[0].twinx().plot(t, φ, '-.', color='red')
        ax[0].plot(np.nan, '-.', label=r'$\phi (t)$', color = 'red')
        ax[0].twinx().set_ylabel("Fase (rad)")

    ax[0].set_xlabel("Tiempo (ps)")
    ax[0].set_ylabel("Campo / Envolvente")
    ax[0].grid()
    ax[0].legend()


    # -- Parte imaginaria del pulso + envolvente --
    ax[1].plot(t, np.imag(pulso), label = r'$\Im \{E(t)\}$')
    ax[1].plot(t, envolvente, '--', label = 'Envolvente')

    if φ is not None:
        ax[1].twinx().plot(t, φ, '-.', color='red')
        ax[1].plot(np.nan, '-.', label=r'$\phi (t)$', color = 'red')
        ax[1].twinx().set_ylabel("Fase (rad)")

    ax[1].set_xlabel("Tiempo (ps)")
    ax[1].set_ylabel("Campo / Envolvente")
    ax[1].grid()
    ax[1].legend()

    fig.suptitle("Partes real e imaginaria del pulso")

    return fig, ax




def plot_intensidad(t, I):
    """
    Realiza una representación de la intensidad del pulso frente al tiempo

    Args:
        t (float, np.array): array de tiempos
        I (float, np.array): array con la intensidad del pulso en un tiempo t

    Devuelve:
        tuple(matplotlib Figure, matplotlib Axis)
    """
    fig, ax = plt.subplots()
    ax.plot(t, I)
    ax.set_xlabel("Tiempo (ps)")
    ax.set_ylabel("Intensidad")
    ax.grid()
    ax.set_title("Intensidad del pulso")

    return fig, ax

def megaplot(t, Δt, pulso, frecuencias, espectro, TBP=None):
    """
    Crea una representación de la intensidad en el dominio temporal,
    la intensidad espectral en el dominio frecuencial y el espectrograma
    de un pulso.

    Args:
        t (np.array): vector de tiempos donde está definido el pulso
        pulso (np.array): campo eléctrico del pulso
        frecuencias (np.array): frecuencias en las que está definido el espectro del pulso
        espectro (np.array): espectro del pulso
        TBP (float): Producto tiempo ancho de banda (opcional)
    """

    I_pulso = np.abs(pulso)**2
    I_espectro = np.abs(espectro)**2

    fig = plt.figure()

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])


    # fig, ax = plt.subplots(2,1)
    twin_ax1 = ax1.twinx()
    twin_ax2 = ax2.twinx()

    if TBP is not None:
        fig.suptitle(f"Pulso aleatorio con TBP={TBP:.2f}")

    fase_campo = np.unwrap(np.angle(pulso)) 
    fase_campo -=  media(fase_campo, I_pulso)

    fase_espectro = np.unwrap(np.angle(espectro)) 
    fase_espectro -=  media(fase_espectro, I_espectro)

    ax1.plot(t, I_pulso, color='blue', label='Intensidad')
    twin_ax1.plot(t, fase_campo, '-.', color='red')
    ax1.plot(np.nan, '-.', label='Fase', color='red')
    ax1.set_xlabel("Tiempo (ps)")
    ax1.set_ylabel("Intensidad (u.a.)")
    twin_ax1.set_ylabel("Fase (rad)")
    ax1.set_title("Dominio temporal")
    ax1.grid()
    ax1.legend()

    ax2.plot(frecuencias, I_espectro, color='blue', label='Intensidad espectral')
    twin_ax2.plot(frecuencias, fase_espectro, '-.', color='red')
    ax2.plot(np.nan, '-.', label='Fase', color='red')
    ax2.set_xlabel("Frecuencia (1 / ps)")
    ax2.set_ylabel("Intensidad (u.a.)")
    twin_ax2.set_ylabel("Fase (rad)")
    ax2.set_title("Dominio frecuencial")
    ax2.grid()
    ax2.legend()


    Σ_g = espectrograma(I_pulso, t, Δt, t.size)
    espectrograma_normalizado = Σ_g / np.max(Σ_g)

    delays = np.linspace(-(t.size - 1) * Δt, (t.size - 1) * Δt, num=2 * t.size - 1)

    im = ax3.pcolormesh(frecuencias, delays, espectrograma_normalizado, cmap='inferno')
    fig.colorbar(im, ax=ax3)
    ax3.set_xlabel("Frecuencia (1/ps)")
    ax3.set_ylabel("Retraso (ps)")
    ax3.set_title(r"$\Sigma_g (\omega, \tau) = |\int_{-\infty}^{\infty} I(t)I(t - \tau) \exp^{- i \omega t} dt|^2$")