import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.constants as constants
from utils import *

plt.rcParams.update({'font.size': 10}) # Tamaño de la fuente del plot


if __name__ == '__main__':
    A=30
    numero_de_muestras = A*1024

    # -- Parámetros del pulso --
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)    
    φ_0 =  0 * np.ones(numero_de_muestras) # Fase (constante en este caso)
    τ = 1 # Duración del pulso (ps)

    to=10*τ
    t, Δt = np.linspace(0, 20, num=numero_de_muestras, retstep=True) # Vector de tiempos (centrado en cero, ps). Guardamos la separación entre datos
    print("Δt (ps)=",Δt)
    fs=1/Δt
    print("fs=1/Δt sampling rate (1/ps)=",fs)    
    pulso = pulso_gaussiano(t-to, A, τ, ω_0, φ_0) # Vector con el campo complejo del pulso
    I = np.abs(pulso) * np.abs(pulso) # Vector con la intensidad del pulso

    # Plot partes real e imaginaria del pulso
    plot_real_imag(t, pulso, φ_0, I)
    plt.show()
    # Comprobamos que el hacer la transformada y su inversa nos devuelve el pulso original
    plot_real_imag(t, ifft(fft(pulso)), φ_0, np.abs(ifft(fft(pulso)))**2)
    plt.show()

    # Plot de la intensidad
    plot_intensidad(t, I)
    plt.show()

    # Construimos el array de frecuencias angulares
    ω = 2 * np.pi * np.arange(numero_de_muestras) / (Δt * numero_de_muestras) # Array de frecuencias angulares (rad / ps)
    
    ω_i = 0.5*ω_0 # Frecuencia angular del pulso (rad / ps)
    ω_f = 2.2*ω_0 # Frecuencia angular del pulso (rad / ps)
    Δω=(ω_f-ω_i)/(numero_de_muestras)
    print("Frecuency range (rad/ps):",ω_i,ω_f)
    print("Frecuency range (1/ps):",ω_i/(2.0*np.pi),ω_f/(2.0*np.pi))
    fmax=ω_f/(2.0*np.pi)
    print("fmax (1/ps)",fmax)
    print("fs>2fmax",fs>2*fmax)
    
    print(2 * np.pi / Δt,2 * np.pi /(numero_de_muestras* Δt))
    ω2 = ω_i +Δω*np.arange(numero_de_muestras) # Array de frecuencias angulares (rad / ps)
    
    transformada_analitica = transformada_pulso_gaussiano(ω2, A, τ, ω_0, φ_0) # Transformada analítica de un pulso gaussiano con fase constante
     
    # Comprobamos que el hacer su transformada inversa nos devuelve el pulso original
    plot_real_imag(t, np.fft.ifftshift((ifft(transformada_analitica))), φ_0, np.abs(np.fft.ifftshift((ifft(transformada_analitica))))**2) #! Hay que centrar el pulso en cero al realizar la inversa. Además no recuperamos amplitud original    
    plt.show()
    #! No devuelve el pulso original. Hay diferencia en la amplitud y también en la fase

    # -- Plot : comparación de los resultados obtenidos por np.fft.fft, scipy.fft.fft y mi implementación
    # transformada_numpy = np.fft.fft(pulso)
    # transformada_scipy = scipy.fft.fft(pulso)
    transformada_propia = DFT_sgr(pulso,ω2,t,Δt,Δω)   
    transformaeda_propia_inv=inv_DFT_sgr(transformada_propia,ω2,t,Δt,Δω) 
    
    # Comprobamos que el hacer la transformada y su inversa nos devuelve el pulso original
    plot_real_imag(t, transformaeda_propia_inv, φ_0, np.abs(transformaeda_propia_inv)**2)
    plt.show()
    
    diferencias_numpy = np.abs(transformada_numpy) #* 1e13
    diferencias_scipy = np.abs(transformada_analitica) #* 1e13
    diferencias_ambos = np.abs(transformada_propia) #* 1e13f
    print(diferencias_scipy.shape)
    print(diferencias_ambos)
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(ω, diferencias_numpy, label='TF NumPy',marker='o',linewidth=0.5, markersize=2)
    ax[1].plot(ω2, diferencias_scipy, label='TF analítica', color='orange',marker='o',linewidth=0.5, markersize=2)
    ax[2].plot(ω2, diferencias_ambos, label='TF DFT', color='green',marker='o',linewidth=0.5, markersize=2)
    fig.suptitle("Comprobación valores obtenidos con las distintas funciones")
    fig.supxlabel("ω (rad / ps)")
    fig.supylabel(r"Diferencia entre valores") #" ($\times 10^{-13}$)")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    
    for i in range(0,3):
        ax[i].set_xlim(1200,1240)
      
 
    plt.show()
    
    '''

    # -- Plot : comparación de los resultados obtenidos por fft y DFT
    transformada_DFT = DFT(pulso)

    diferencias_DFT_fft = np.abs(transformada_propia - transformada_DFT) #* 1e10

    fig, ax = plt.subplots()
    ax.plot(ω, diferencias_DFT_fft)
    fig.suptitle("Comprobación valores obtenidos con fft y DFT")
    fig.supxlabel("ω (rad / ps)")
    fig.supylabel(r"Diferencia entre valores") #" ($\times 10^{-10}$)")
    ax.grid()

    plt.show()

    #! Las diferencias de resultados entre algoritmos son muy pequeñas y la velocidad de la fft es muy superior a la DFT

    # Comparacion resultado entre la transformada analítica y la fft. Comparo la transformada normalizada porque no tienen la misma escala

    fft_normalizada = transformada_propia / np.max(transformada_propia)
    analitica_normalizada = transformada_analitica / np.max(transformada_analitica)

    diferencias_analitica_fft = np.abs(fft_normalizada - analitica_normalizada)

    fig, ax = plt.subplots()
    ax.plot(ω, diferencias_analitica_fft)
    fig.supylabel("Diferencia entre amplitud coeficientes")
    fig.supxlabel("ω (rad / ps)")
    ax.set_title("Diferencia entre fft normalizdas")

    ax.grid()

    plt.show()

    # Comparación de resultados entre la transformada inversa ifft y analítica

    diferencias_inversas = np.abs(ifft(fft_normalizada) -  ifft(analitica_normalizada))

    fig, ax = plt.subplots()
    ax.plot(t, diferencias_inversas)
    fig.supylabel("Diferencia entre valores")
    fig.supxlabel("t (ps)")
    ax.set_title("Diferencia entre ifft normalizadas")
    ax.grid()

    plt.show()

    # Parece haber un desfase entre las dos señales

    # Transformadas de pulsos con distintas anchuras temporales 
    τ_1, τ_2, τ_3 = 2.0, 1.0, 0.5 # En ps
    pulso_1 = fft(pulso_gaussiano(t, A, τ_1, ω_0, φ_0))
    pulso_2 = fft(pulso_gaussiano(t, A, τ_2, ω_0, φ_0))
    pulso_3 = fft(pulso_gaussiano(t, A, τ_3, ω_0, φ_0))

    fig, ax = plt.subplots()
    ax.plot(ω, np.abs(pulso_1), label=r"$\tau =$"+f"{τ_1} ps")
    ax.plot(ω, np.abs(pulso_2), label=r"$\tau =$"+f"{τ_2} ps")
    ax.plot(ω, np.abs(pulso_3), label=r"$\tau =$"+f"{τ_3} ps")
    ax.set_ylabel("Amplitud coeficientes")
    ax.set_xlabel("ω (rad / ps)")
    ax.set_xlim([1200, 1230])
    ax.grid()
    ax.legend()
    # Vemos que se cumple que a mayor anchura temporal menor anchura espectral
    plt.show()
    '''