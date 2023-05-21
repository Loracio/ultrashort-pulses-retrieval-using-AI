import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *
import scipy.constants as constants

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

class animacionAutocorrelacion2orden:
    def __init__(self, t, Δt, valores, interval=10):
        self.t = t
        self.Δt = Δt
        self.N = self.t.size

        self.M = 2 * self.N - 1 # Número de delays totales
        self.delays = np.linspace(-(self.N - 1), (self.N - 1) , num=self.M, dtype=int) * self.Δt

        self.valores = valores

        self.inmovil = np.zeros(2 * self.N - 1, dtype=valores.dtype)
        self.movil = np.zeros(2 * self.N - 1, dtype=valores.dtype)

        self.inmovil[self.N - 1 : 2 * self.N - 1] = valores
        self.movil[0 : self.N] = valores

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self.pause)

        self.valores_autocorrelacion = autocorrelacion_2orden(self.valores, self.N, self.Δt)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1
        if self.interval * self.frame > np.size(self.delays):
            self.frame = 0
            self.movil[0 : self.N] = self.valores

        self.movil = np.roll(self.movil, self.interval)
        if self.interval * self.frame > self.N:
            self.movil[0:self.N] = 0
        self.line_movil.set_ydata(self.movil)

        self.line_integrando.set_ydata(self.inmovil * self.movil)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        if self.frame > 1:
            self.area1.remove()
            self.area2.remove()

        self.area1 = self.ax[0].fill_between(self.delays, 0, self.inmovil, where=self.movil > self.inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.delays, 0, self.movil, where=self.inmovil > self.movil, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.delays, self.inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.delays, self.movil, label="I(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="Autocorrelación")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.delays, self.inmovil * self.movil, color='green', label="I(t)·I(t-τ)")
        self.ax[1].set_xlabel("Tiempo")
        self.ax[1].set_ylabel("Valor del integrando")
        self.ax[1].legend(loc='upper right')
        self.ax[1].grid()
        self.ax[1].set_ylim(top=np.max(self.inmovil)**2 * 1.10)

        self.ax[2].set_xlabel("Retraso")
        self.ax[2].set_ylabel("Intensidad (u.a.)")
        self.ax[2].legend(loc='upper right')
        self.ax[2].grid()
        self.ax[2].set_xlim([self.delays[0], self.delays[-1]])
        if np.min(self.valores_autocorrelacion) < 0.01:
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.10)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.10)
    
    def pause(self, event):
        """
        Pause animation function.

        Parameters
        ---------
        event : matplotlib keypress event
            Necessary parameter for button update function
        """
        if event.key == ' ':
            if self.paused:
                self.animation.resume()

            else:
                self.animation.pause()

            self.paused = not self.paused


#! ################################################################################################################################################################################################

class animacionAutocorrelacion3orden:
    def __init__(self, t, Δt, valores, interval=10):
        self.t = t
        self.Δt = Δt
        self.N = self.t.size

        self.M = 2 * self.N - 1 # Número de delays totales
        self.delays = np.linspace(-(self.N - 1), (self.N - 1) , num=self.M, dtype=int) * self.Δt

        self.valores = valores

        self.inmovil = np.zeros(2 * self.N - 1, dtype=valores.dtype)
        self.movil = np.zeros(2 * self.N - 1, dtype=valores.dtype)

        self.inmovil[self.N - 1 : 2 * self.N - 1] = valores
        self.movil[0 : self.N] = valores**2

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self.pause)

        self.valores_autocorrelacion = autocorrelacion_3orden(self.valores, self.N, self.Δt)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1
        if self.interval * self.frame > np.size(self.delays):
            self.frame = 0
            self.movil[0 : self.N] = self.valores**2

        self.movil = np.roll(self.movil, self.interval)
        if self.interval * self.frame > self.N:
            self.movil[0:self.N] = 0
        self.line_movil.set_ydata(self.movil)

        self.line_integrando.set_ydata(self.inmovil * self.movil)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        if self.frame > 1:
            self.area1.remove()
            self.area2.remove()

        self.area1 = self.ax[0].fill_between(self.delays, 0, self.inmovil, where=self.movil > self.inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.delays, 0, self.movil, where=self.inmovil > self.movil, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.delays, self.inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.delays, self.movil, label="I²(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="Autocorrelación")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.delays, self.inmovil * self.movil, color='green', label="I(t)·I²(t-τ)")
        self.ax[1].set_xlabel("Tiempo")
        self.ax[1].set_ylabel("Valor del integrando")
        self.ax[1].legend(loc='upper right')
        self.ax[1].grid()
        self.ax[1].set_ylim(top=np.max(self.inmovil)**2 * 1.10)

        self.ax[2].set_xlabel("Retraso")
        self.ax[2].set_ylabel("Intensidad (u.a.)")
        self.ax[2].legend(loc='upper right')
        self.ax[2].grid()
        self.ax[2].set_xlim([self.delays[0], self.delays[-1]])
        if np.min(self.valores_autocorrelacion) < 0.01:
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.10)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.10)
    
    def pause(self, event):
        """
        Pause animation function.

        Parameters
        ---------
        event : matplotlib keypress event
            Necessary parameter for button update function
        """
        if event.key == ' ':
            if self.paused:
                self.animation.resume()

            else:
                self.animation.pause()

            self.paused = not self.paused

#! ################################################################################################################################################################################################

class animacionIA:
    def __init__(self, t, Δt, valores, interval=10):
        self.t = t
        self.Δt = Δt
        self.N = self.t.size

        self.M = 2 * self.N - 1 # Número de delays totales
        self.delays = np.linspace(-(self.N - 1), (self.N - 1) , num=self.M, dtype=int) * self.Δt

        self.valores = valores

        self.campo_inmovil = np.zeros(2 * self.N - 1, dtype=valores.dtype)
        self.intensidad_inmovil = np.zeros(2 * self.N - 1, dtype=valores.dtype)
        
        self.campo_movil = np.zeros(2 * self.N - 1, dtype=valores.dtype)
        self.intensidad_movil = np.zeros(2 * self.N - 1, dtype=valores.dtype)

        self.campo_inmovil[self.N - 1 : 2 * self.N - 1] = valores
        self.intensidad_inmovil[self.N - 1 : 2 * self.N - 1] = np.abs(valores)**2

        self.campo_movil[0 : self.N] = valores
        self.intensidad_movil[0 : self.N] = np.abs(valores)**2

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self.pause)

        self.valores_autocorrelacion = autocorrelacion_interferometrica(self.valores, self.N, self.Δt)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1

        if self.interval * self.frame > np.size(self.delays):
            self.frame = 0
            self.campo_movil[0 : self.N] = self.valores
            self.intensidad_movil[0 : self.N] = np.abs(self.valores)**2

        self.campo_movil = np.roll(self.campo_movil, self.interval)
        self.intensidad_movil = np.roll(self.intensidad_movil, self.interval)

        if self.interval * self.frame > self.N:
            self.campo_movil[0 : self.N] = 0
            self.intensidad_movil[0 : self.N] = 0

        self.line_movil.set_ydata(self.intensidad_movil)

        self.integrando = np.abs((self.campo_movil+ self.campo_inmovil)**2)**2
        self.line_integrando.set_ydata(self.integrando)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        try:
            self.area1.remove()
            self.area2.remove()
            self.area3.remove()

        except:
            pass

        self.area1 = self.ax[0].fill_between(self.delays, 0, self.intensidad_inmovil, where=self.intensidad_movil > self.intensidad_inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.delays, 0, self.intensidad_movil, where=self.intensidad_inmovil > self.intensidad_movil, alpha=0.5, color='mediumseagreen')
        self.area3 = self.ax[1].fill_between(self.delays, 0, self.integrando, where=self.integrando > 0, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.delays, self.intensidad_inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.delays, self.intensidad_movil, label="I(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="Autocorrelación")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.delays, np.abs((self.campo_movil+ self.campo_inmovil)**2)**2, color='green', label="|E²(t) + 2E(t)E(t-τ) + E²(t-τ)|")
        self.ax[1].set_xlabel("Tiempo")
        self.ax[1].set_ylabel("Valor del integrando")
        self.ax[1].legend(loc='upper right')
        self.ax[1].grid()
        self.ax[1].set_ylim(top=16 * np.max(np.abs(self.campo_inmovil**2))**2)

        self.ax[2].set_xlabel("Retraso")
        self.ax[2].set_ylabel("Intensidad (u.a.)")
        self.ax[2].legend(loc='upper right')
        self.ax[2].grid()
        self.ax[2].set_xlim([self.delays[0], self.delays[-1]])
        if np.min(self.valores_autocorrelacion) < 0.01:
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.10)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.10)
    
    def pause(self, event):
        """
        Pause animation function.

        Parameters
        ---------
        event : matplotlib keypress event
            Necessary parameter for button update function
        """
        if event.key == ' ':
            if self.paused:
                self.animation.resume()

            else:
                self.animation.pause()

            self.paused = not self.paused

if __name__ == '__main__':

    def calcula_intensidad_ejemplo(t, β, α=None):
        if α is not None:
            return np.exp(-β * t) * ( 1 - 4* β / α * np.sin(α * t) + 4 * β * β/ (α * α) *  (1 - np.cos(α *t)) )
        else:
            return np.exp(-β * t)

    def intensidad_ejemplo(t, β, α=None):
        return np.piecewise(t, [t < 0, t >= 0], [0, calcula_intensidad_ejemplo], β, α)

    t, Δt = np.linspace(-10, 60, num=4096, retstep=True)

    β = 0.1
    α = 1.6 / 4

    valores = intensidad_ejemplo(t, β, α=α)

    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, valores, interval=50)
    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, valores, interval=50)
    plt.show()

    #! ########################## PRUEBAS CON PULSOS GAUSSIANOS ######################################################

    # Parámetros de la medida
    numero_de_muestras = 2 * 4096
    duracion_temporal = 2 * 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0 # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 # Fase del pulso (constante en este caso)
    τ = 1 # Duración del pulso (ps)
    a = 0 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ ) # Chirpeo lineal

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")


    def doble_pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2):
        """
        Crea un doble pulso gaussiano como suma de dos pulsos con diferentes parámetros.
        """
        return pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1) + pulso_gaussiano(t, t0_2, A_2, τ_2, ω_0_2, φ_2)

    def intensidad(t, t0, A, τ, ω_0, φ):
        """
        Definimos una función que calcule la intensidad a partir de los parámetros del pulso
        """
        return np.abs(pulso_gaussiano(t, t0, A, τ, ω_0, φ))**2

    def intensidad_doble_pulso(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2):
        """
        Definimos una función que calcule la intensidad a partir de los parámetros del doble pulso
        """
        return np.abs(doble_pulso_gaussiano(t, t0_1, A_1, τ_1, ω_0_1, φ_1, t0_2, A_2, τ_2, ω_0_2, φ_2))**2

    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, intensidad(t, t0, A, τ, ω_0, φ), interval=100)
    plt.show()

    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, intensidad_doble_pulso(t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), interval=100)
    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, intensidad(t, t0, A, τ, ω_0, φ), interval=100)
    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, intensidad_doble_pulso(t, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), interval=100)
    plt.show()

    animacion_IA = animacionIA(t, Δt, pulso_gaussiano(t, t0, A, τ, ω_0, φ), interval=100)
    plt.show()

    animacion_IA = animacionIA(t, Δt, doble_pulso_gaussiano(t,-5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ), interval=100)
    plt.show()