import numpy as np
import matplotlib.pyplot as plt


def plot_real_imag(t, pulso, φ, I):
    """
    Realiza una representación de las partes real e imaginaria del pulso pasado

    Args:
        t (float, np.array): array de tiempos
        pulso (np.ndarray[np.complex]): array con los valores del pulso en un tiempo t
        φ (float, np.array): array de la fase en un tiempo t
        I (float, np.array): array con la intensidad del pulso en un tiempo t

    Devuelve:
        tuple(matplotlib Figure, matplotlib Axis)
    """
    fig, ax = plt.subplots(2,1)

    # -- Parte real del pulso + envolvente --
    ax[0].plot(t, np.real(pulso), label = r'$\Re \{E(t)\}$')
    ax[0].plot(t, np.sqrt(I), '--', label = 'Envolvente')
    ax[0].plot(t, φ, '-.', label = r'$\phi (t)$')
    ax[0].set_xlabel("Tiempo (ps)")
    ax[0].set_ylabel("Campo / Envolvente (Normalizado)")
    ax[0].grid()
    ax[0].legend()


    # -- Parte imaginaria del pulso + envolvente --
    ax[1].plot(t, np.imag(pulso), label = r'$\Im \{E(t)\}$')
    ax[1].plot(t, np.sqrt(I), '--', label = 'Envolvente')
    ax[1].plot(t, φ, '-.', label = r'$\phi (t)$')
    ax[1].set_xlabel("Tiempo (ps)")
    ax[1].set_ylabel("Campo / Envolvente (Normalizado)")
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