import numpy as np
import matplotlib.pyplot as plt

import importlib
spec = importlib.util.find_spec("core")
if spec is None or spec.origin == "namespace":
    import sys
    from pathlib import Path
    core_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(core_folder))

import tensorflow as tf
from core import *

plt.rcParams.update({'font.size': 18}) # Tamaño de la fuente del plot

def plot_resultado_NN(campo, T_medido, λ0):
    fig, ax = plt.subplots(2,2)

    twin_ax00 = ax[0][0].twinx()
    twin_ax01 = ax[0][1].twinx()
    

    I_campo_solucion = np.abs(campo)**2

    I_espectral_solucion = np.abs(DFT(campo, τ, Δτ, ω, Δω))**2

    fase_campo_solucion = np.unwrap(np.angle(campo)) 
    fase_campo_solucion -=  media(fase_campo_solucion, I_campo_solucion)
    fase_campo_solucion = np.where(I_campo_solucion < 1e-3, np.nan, fase_campo_solucion)

    fase_espectro_solucion = np.unwrap(np.angle(DFT(campo, τ, Δτ, ω, Δω))) 
    fase_espectro_solucion -=  media(fase_espectro_solucion, I_espectral_solucion)
    fase_espectro_solucion = np.where(I_espectral_solucion < 1e-3, np.nan, fase_espectro_solucion)

    ax[0][0].plot(τ, I_campo_solucion, color='orange', label='Intensidad campo recuperado')
    twin_ax00.plot(τ, fase_campo_solucion, '-.', color='violet')
    ax[0][0].plot(np.nan, '-.', label='Fase campo recuperado', color='violet')
    ax[0][0].set_xlabel("Tiempo (fs)")
    ax[0][0].set_ylabel("Intensidad (u.a.)")
    twin_ax00.set_ylabel("Fase (rad)")
    ax[0][0].set_title("Dominio temporal")
    ax[0][0].grid()

    ax[0][1].plot(2 * np.pi * 300 / (2 * np.pi * 300 / λ0 + ω), I_espectral_solucion, color='orange', label='Intensidad espectral recuperada')
    twin_ax01.plot(2 * np.pi * 300 / (2 * np.pi * 300 / λ0 + ω), fase_espectro_solucion, '-.', color='violet')
    ax[0][1].plot(np.nan, '-.', label='Fase espectral recuperada', color='violet')
    ax[0][1].set_xlabel("λ (nm)")
    ax[0][1].set_ylabel("Intensidad (u.a.)")
    twin_ax01.set_ylabel("Fase (rad)")
    ax[0][1].set_title("Dominio frecuencial")
    ax[0][1].grid()

    fig.legend(*ax[0][0].get_legend_handles_labels(), loc='upper center', ncols=4)

    T_medido_normalizado = T_medido / np.max(T_medido)

    T_recuperado  = traza(campo, τ, Δτ, N)
    T_recuperado_normalizado = T_recuperado / np.max(T_recuperado)

    μ = np.sum(T_medido * T_recuperado[int((N-1)/2):int((N-1)/2) + N]) / np.sum(T_recuperado[int((N-1)/2):int((N-1)/2) + N] * T_recuperado[int((N-1)/2):int((N-1)/2) + N])
    diferencia = np.ravel(T_medido - μ * T_recuperado[int((N-1)/2):int((N-1)/2) + N])
    r = np.sum(diferencia * diferencia)
    R = np.sqrt(r / (N * N * T_medido.max()**2))


    im0 = ax[1][0].pcolormesh(2 * np.pi * 300 / λ0 + ω, τ, T_medido_normalizado, cmap='inferno')
    fig.colorbar(im0, ax=ax[1][0])
    ax[1][0].set_xlabel("ω (2π/fs)")
    ax[1][0].set_ylabel("τ (fs)")
    ax[1][0].set_title("Traza del pulso medido")

    im1 = ax[1][1].pcolormesh(2 * np.pi * 300 / λ0 + ω, τ, T_recuperado_normalizado[int((N-1)/2):int((N-1)/2) + N], cmap='inferno')
    fig.colorbar(im1, ax=ax[1][1])
    ax[1][1].set_xlabel("ω (2π/fs)")
    ax[1][1].set_ylabel("τ (fs)")
    ax[1][1].set_title(f"Traza del pulso recuperado, R = {R:.2E}")

    return fig, ax

if __name__ == '__main__':
    N = 128

    τ = np.zeros(N)
    ω = np.zeros(N)

    with open("./medidas_experimentales/ejes.csv", 'r') as filein:
        for i, line in enumerate(filein):
            ω[i], τ[i] = line.split(',')

    λ0 = 790
    ω0 = 2 * np.pi * 300 / λ0

    Δω = ω[1] - ω[0]
    Δτ = τ[1] - τ[0]

    """
    Leemos las dos medidas, 900mA y 2100mA
    """

    T_900mA = np.zeros((N, N))
    T_2100mA = np.zeros((N, N))

    with open("./medidas_experimentales/traza_900mA.csv", 'r') as filein:
        for i, line in enumerate(filein):
            T_900mA[i][:] = line.split(',')

    with open("./medidas_experimentales/traza_2100mA.csv", 'r') as filein:
        for i, line in enumerate(filein):
            T_2100mA[i][:] = line.split(',')

    """
    Para poder emplear la NN hay que rellenar el la traza con ceros,
    ya que la entrada ha de ser de (2N-1)*N y no de N*N como se tiene
    en la medida experimental
    """


    T_NN = np.zeros((2, (2*N-1) * N))

    T_zero_padded = np.zeros(((2*N-1), N))

    for i in range(N):
        T_zero_padded[int((N-1)/2) + i][:] = T_900mA[i][:] 

    T_NN[0] = T_zero_padded.flatten()

    for i in range(N):
        T_zero_padded[int((N-1)/2) + i][:] = T_2100mA[i][:] 

    T_NN[1] = T_zero_padded.flatten()


    direccion_modelo = "./IA/NN_models/campo_model_simple_dense.h5"

    model = tf.keras.models.load_model(direccion_modelo)

    campos_concat_pred = model.predict(T_NN)

    pulso_900mA_NN = campos_concat_pred[0][:N] + 1j * campos_concat_pred[0][N:]
    pulso_2100mA_NN = campos_concat_pred[1][:N] + 1j * campos_concat_pred[1][N:]

    plot_resultado_NN(pulso_900mA_NN, T_900mA, λ0)
    plot_resultado_NN(pulso_2100mA_NN, T_2100mA, λ0)

    print(f"Anchura temporal pulso 900mA predicha por la NN : {FWHM(np.abs(pulso_900mA_NN)**2, Δτ):.2f} fs")
    print(f"Anchura temporal pulso 2100mA predicha por la NN : {FWHM(np.abs(pulso_2100mA_NN)**2, Δτ):.2f} fs")

    plt.show()
    
    """
    Algoritmos de recuperación:
    """

    # Pulso candidato como gaussiana con fase aleatoria
    fase_max = 0.3 * np.pi
    sigma = 2
    fase = np.exp(1.0j * np.random.uniform(-fase_max, fase_max, N))
    d = τ / sigma

    pulso_candidato = np.exp(-0.5 * d * d) * fase
    espectro_candidato = DFT(pulso_candidato, τ, Δτ, ω, Δω)


    ret_GPA_900mA = GPA(τ, Δτ, ω, Δω, T_900mA)
    campo_GPA_900mA, espectro_GPA_900mA = ret_GPA_900mA.recuperacion(pulso_candidato, 1e-10, max_iter=500)
    ret_GPA_900mA.plot(λ0)

    ret_PCGPA_900mA = PCGPA(τ, Δτ, ω, Δω, T_900mA)
    campo_PCGPA_900mA, espectro_PCGPA_900mA = ret_PCGPA_900mA.recuperacion(pulso_candidato, 1e-10, max_iter=500)
    ret_PCGPA_900mA.plot(λ0)

    print(f"Anchura temporal pulso 900mA predicha por el método GPA : {FWHM(np.abs(campo_GPA_900mA)**2, Δτ):.2f} fs")
    print(f"Anchura temporal pulso 900mA predicha por el método PCGPA : {FWHM(np.abs(campo_PCGPA_900mA)**2, Δτ):.2f} fs")

    plt.show()



    ret_GPA_2100mA = GPA(τ, Δτ, ω, Δω, T_2100mA)
    campo_GPA_2100mA, espectro_GPA_2100mA = ret_GPA_2100mA.recuperacion(pulso_candidato, 1e-10, max_iter=500)
    ret_GPA_2100mA.plot(λ0)

    ret_PCGPA_2100mA = PCGPA(τ, Δτ, ω, Δω, T_2100mA)
    campo_PCGPA_2100mA, espectro_PCGPA_2100mA = ret_PCGPA_2100mA.recuperacion(pulso_candidato, 1e-10, max_iter=500)
    ret_PCGPA_2100mA.plot(λ0)

    print(f"Anchura temporal pulso 2100mA predicha por el método GPA : {FWHM(np.abs(campo_GPA_2100mA)**2, Δτ):.2f} fs")
    print(f"Anchura temporal pulso 2100mA predicha por el método PCGPA : {FWHM(np.abs(campo_PCGPA_2100mA)**2, Δτ):.2f} fs")

    plt.show()
