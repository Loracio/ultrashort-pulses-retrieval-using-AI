import numpy as np
import matplotlib.pyplot as plt

import importlib
spec = importlib.util.find_spec("core")
if spec is None or spec.origin == "namespace":
    import sys
    from pathlib import Path
    core_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(core_folder))

from core import *


if __name__ == '__main__':
    N = 128

    τ = np.zeros(N)
    ω = np.zeros(N)

    with open("./medidas_experimentales/ejes.csv", 'r') as filein:
        for i, line in enumerate(filein):
            ω[i], τ[i] = line.split(',')

    λ0 = 790
    ω0 = 2 * np.pi * 300 / λ0

    ω += ω0

    Δω = ω[1] - ω[0]
    Δτ = τ[1] - τ[0]

    T = np.zeros((N, N))

    with open("./medidas_experimentales/traza_900mA.csv", 'r') as filein:
        for i, line in enumerate(filein):
            T[i][:] = line.split(',')


    # Pulso candidato como gaussiana con fase aleatoria
    fase_max = 0.3 * np.pi
    sigma = 2
    fase = np.exp(1.0j * np.random.uniform(-fase_max, fase_max, N))
    d = τ / sigma

    pulso_candidato = np.exp(-0.5 * d * d) * fase
    espectro_candidato = DFT(pulso_candidato, τ, Δτ, ω, Δω)

    ret_GPA = GPA(τ, Δτ, ω, Δω, T)
    campo, espectro = ret_GPA.recuperacion(pulso_candidato, 1e-10, max_iter=200)
    ret_GPA.plot()

    ret_PCGPA = PCGPA(τ, Δτ, ω, Δω, T)
    ret_PCGPA.recuperacion(pulso_candidato, 1e-10, max_iter=200)
    ret_PCGPA.plot()

    plt.show()