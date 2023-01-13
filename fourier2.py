import numpy as np
import matplotlib.pyplot as plt
from core import *

N = 512
duracion_temporal = 10
f_muestreo = N / duracion_temporal


t, Δt = np.linspace(-5, 5, num=N, retstep=True)

Δω = 2*np.pi / (Δt * N)
ω0 = -np.floor(0.5 * N) * Δω
ω = ω0 + np.arange(N) * Δω
ω = np.fft.fftfreq(N, Δt)

array_frecuencias = convertir(ω, 'frecuencia angular', 'frecuencia')
frecuencia = 1

señal = np.sin(2*np.pi * t * frecuencia)


coeficientes_transformada = DFT_naive(señal, t, Δt, ω, Δω)
señal_recuperada = IDFT_naive(coeficientes_transformada, t, Δt, ω, Δω)

fig, ax = plt.subplots(2,1)

ax[0].plot(t, señal, label='Señal original')
ax[0].plot(t, np.real(señal_recuperada), '--', label='Señal recuperada')
# ax[0].plot(t, np.real(ifft(fft(señal))), '--', label='Señal recuperada NumPy')
ax[0].grid()
ax[0].legend()

ax[1].plot(array_frecuencias, np.abs(coeficientes_transformada))
ax[1].plot(array_frecuencias, np.abs(fft(señal)*Δt), '--')
ax[1].axvline(x=0)
ax[1].grid()
# ax[1].legend()
plt.show()