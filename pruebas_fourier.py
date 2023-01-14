import numpy as np
import matplotlib.pyplot as plt
from core import *
from time import process_time as timer

def DFT_sgr(f,ω,t,Δt,Δω):
    N = f.size
    n = np.arange(N)
    ω = ω.reshape((len(ω), 1))    
    e = np.exp(-1j*ω*t )
    f=f*Δt
    return e@f 


def inv_DFT_sgr(f,ω,t,Δt,Δω):
    N = f.size
    n = np.arange(N)
    ω = ω.reshape((len(ω), 1))    
    e = np.exp(1j*ω*t )
    f=f*Δω/(2.0*np.pi)
    return f@e

N = 5096
T = 250
Δt = T / N
f_s = 1 / Δt # Hz

f = 0.1 # Hz

t = np.arange(-T/2, T/2, step=Δt)
señal =  np.sin(2 * np.pi * t * f) / (2 * np.pi * t * f) + 1j*np.cos(2 * np.pi * t * f) / (2 * np.pi * t * f)

frecuencias = frecuencias_DFT(N, Δt)
ω = convertir(frecuencias, 'frecuencia', 'frecuencia angular')
Δω = 2 * np.pi / (Δt * N)

tic = timer()
transformada = DFT(señal, t, Δt, ω, Δω)
toc = timer()
tiempo = (toc-tic)*1e6
print(f"Mia {tiempo}μs") 

Δω_sgr = 2 * np.pi / (Δt * N) #0.001
M = int(1 / Δω)
frecuencias_sgr = np.array([-f_s / 2 + j /(N * Δt) for j in range(N)]) #np.array([-0.5 + j * Δω for j in range(M)])
ω_sgr = convertir(frecuencias_sgr, 'frecuencia', 'frecuencia angular')

tic = timer()
transformada_sergut = DFT_sgr(señal, ω_sgr, t, Δt, Δω_sgr)
toc = timer()
tiempo = (toc-tic)*1e6
print(f"SERGUT {tiempo}μs") 

recuperada_mia = IDFT(transformada, t, Δt, ω, Δω)
recuperada_sergut = inv_DFT_sgr(transformada_sergut, ω_sgr, t, Δt, Δω_sgr)

fig, ax = plt.subplots(4,1)

ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Amplitud")
# ax[0].plot(t, np.real(señal))
ax[0].plot(t, np.real(recuperada_mia))
ax[0].plot(t, np.real(recuperada_sergut), '--')
# ax[0].plot(t, np.real(señal))
ax[0].grid()

ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Amplitud")
ax[1].plot(frecuencias, np.abs(transformada), label="Yo")
# ax[1].plot(frecuencias, np.abs(np.fft.ifftshift(fft(np.fft.fftshift(señal))*Δt)), label="fft")
ax[1].plot(frecuencias_sgr, np.abs(transformada_sergut), '--', label="Sergut")
ax[1].legend()
ax[1].grid()

ax[2].set_xlabel("Frecuencia (Hz)")
ax[2].set_ylabel("Amplitud")
ax[2].plot(frecuencias, np.real(transformada), label="Yo")
# ax[2].plot(frecuencias, np.real(np.fft.ifftshift(fft(np.fft.fftshift(señal))*Δt)), label="fft")
ax[2].plot(frecuencias_sgr, np.real(transformada_sergut), '--', label="Sergut")
ax[2].legend()
ax[2].grid()

ax[3].set_xlabel("Frecuencia (Hz)")
ax[3].set_ylabel("Amplitud")
ax[3].plot(frecuencias, np.imag(transformada), label="Yo")
# ax[3].plot(frecuencias, np.imag(np.fft.ifftshift(fft(np.fft.fftshift(señal))*Δt)), label="fft")
ax[3].plot(frecuencias_sgr, np.imag(transformada_sergut), '--', label="Sergut")
ax[3].legend()
ax[3].grid()

plt.show()