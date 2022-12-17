import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from core import *


numero_de_muestras = 128
duracion_temporal = 5
frecuencia_muestreo = numero_de_muestras / duracion_temporal


# Array de tiempos
t = np.arange(0, duracion_temporal, 1/frecuencia_muestreo)

# Señal de amplitud constante + desfase
global amplitud # Hay que definirla globalmente para usarla en la función del slider
amplitud = 1
global desfase # Hay que definirla globalmente para usarla en la función del slider
desfase = 0
frecuencia_señal = 1
x = amplitud * np.exp(2j * np.pi * frecuencia_señal * t) * np.exp(-1j * desfase)

# Transformada del seno
X = fft(x)

# Construimos array frecuencias. Hay que dividir entre 2π porque la función devuelve en rad / unidad tiempo
f = convertir_tiempo_frecuencia(numero_de_muestras, 1 / frecuencia_muestreo) / (2 * np.pi)


fig, ax = plt.subplots(2,1)

def update_desfase(valor):
    global amplitud, desfase
    desfase = valor
    x = amplitud * np.exp(2j * np.pi * frecuencia_señal * t) * np.exp(-1j * desfase)
    line0.set_ydata(np.real(x))
    line1.set_ydata(np.abs(fft(x)))
    line2.set_ydata(np.real(ifft(fft(x))))

def update_amplitud(valor):
    global amplitud, desfase
    amplitud = valor
    x = amplitud * np.exp(2j * np.pi * frecuencia_señal * t) * np.exp(-1j * desfase)
    line0.set_ydata(np.real(x))
    line1.set_ydata(np.abs(fft(x)))
    line2.set_ydata(np.real(ifft(fft(x))))

plt.subplots_adjust(bottom=0.25)
slider_desfase = Slider(plt.axes([0.15, 0.15, 0.65, 0.03]), "Desfase", 0, 2 * np.pi, valstep=0.01, valinit=0, initcolor='none')
slider_desfase.on_changed(update_desfase)
slider_amplitud = Slider(plt.axes([0.15, 0.05, 0.65, 0.03]), "Amplitud", 0.1, 1, valstep=0.01, valinit=1, initcolor='none')
slider_amplitud.on_changed(update_amplitud)

line0, = ax[0].plot(t, np.real(x))
line2, = ax[0].plot(t, np.real(ifft(fft(x))), '--')
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Amplitud")
ax[0].grid()


line1, = ax[1].plot(f, np.abs(X))
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Amplitud coeficientes")
ax[1].grid()
plt.show()