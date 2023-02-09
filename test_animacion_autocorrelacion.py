import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from core import *
import scipy.constants as constants


class animacionAutocorrelacion2orden:
    def __init__(self, t, Δt, delays, funcion, *args, interval=10):
        self.t = t
        self.Δt = Δt
        self.delays = delays
        self.func = funcion
        self.args = args

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.pause_button = Button(plt.axes([0.45, 0.9, 0.1, 0.05]), 'Pause', hovercolor='paleturquoise', color='slategrey')
        self.pause_button.on_clicked(self.pause)
        self.paused = False

        self.inmovil = self.func(self.t, *self.args) # Pulso que se queda inmovil para hacer la autocorrelacion
        self.movil = self.func(self.t - self.delays[0], *self.args) # Pulso que se irá moviendo a lo largo de la animación

        self.valores_autocorrelacion = autocorrelacion_2orden(self.delays, self.Δt, self.func, self.t, *self.args)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1
        if self.interval * self.frame > np.size(self.delays):
            # self.animation.pause()
            self.frame = 0
            # return

        self.movil = self.func(self.t - self.delays[self.interval * self.frame], *self.args)
        self.line_movil.set_ydata(self.movil)

        self.line_integrando.set_ydata(self.inmovil * self.movil)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        if self.frame > 1:
            self.area1.remove()
            self.area2.remove()

        self.area1 = self.ax[0].fill_between(self.t, 0, self.inmovil, where=self.movil > self.inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.t, 0, self.movil, where=self.inmovil > self.movil, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.t, self.inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.t, self.movil, label="I(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="Autocorrelación")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.t, self.inmovil * self.movil, color='green', label="I(t)·I(t-τ)")
        self.ax[1].set_xlabel("Tiempo")
        self.ax[1].set_ylabel("Valor del integrando")
        self.ax[1].legend(loc='upper right')
        self.ax[1].grid()
        self.ax[1].set_ylim(top=np.max(self.inmovil)**2 * 1.25)

        self.ax[2].set_xlabel("Retraso")
        self.ax[2].set_ylabel("Intensidad (u.a.)")
        self.ax[2].legend(loc='upper right')
        self.ax[2].grid()
        self.ax[2].set_xlim([self.delays[0], self.delays[-1]])
        if np.min(self.valores_autocorrelacion) < 0.01:
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.25)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.25)
    
    def pause(self, event):
        """
        Pause button updating function. Changes label of the button between 'Pause' and 'Play'

        Parameters
        ---------
        event : matplotlib click event
            Necessary parameter for button update function
        """        
        if self.paused:
            self.pause_button.label.set_text('Pause')
            self.animation.resume()

        else:
            self.pause_button.label.set_text('Play')
            self.animation.pause()

        self.paused = not self.paused


#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################


class animacionAutocorrelacion3orden:
    def __init__(self, t, Δt, delays, funcion, *args, interval=10):
        self.t = t
        self.Δt = Δt
        self.delays = delays
        self.func = funcion
        self.args = args

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.pause_button = Button(plt.axes([0.45, 0.9, 0.1, 0.05]), 'Pause', hovercolor='paleturquoise', color='slategrey')
        self.pause_button.on_clicked(self.pause)
        self.paused = False

        self.inmovil = self.func(self.t, *self.args) # Pulso que se queda inmovil para hacer la autocorrelacion
        self.movil = self.func(self.t - self.delays[0], *self.args)**2 # Pulso que se irá moviendo a lo largo de la animación

        self.valores_autocorrelacion = autocorrelacion_3orden(self.delays, self.Δt, self.func, self.t, *self.args)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1
        if self.interval * self.frame > np.size(self.delays):
            # self.animation.pause()
            self.frame = 0
            # return

        self.movil = self.func(self.t - self.delays[self.interval * self.frame], *self.args)**2
        self.line_movil.set_ydata(self.movil)

        self.line_integrando.set_ydata(self.inmovil * self.movil)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        if self.frame > 1:
            self.area1.remove()
            self.area2.remove()

        self.area1 = self.ax[0].fill_between(self.t, 0, self.inmovil, where=self.movil > self.inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.t, 0, self.movil, where=self.inmovil > self.movil, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.t, self.inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.t, self.movil, label="I²(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="Autocorrelación")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.t, self.inmovil * self.movil, color='green', label="I(t)·I²(t-τ)")
        self.ax[1].set_xlabel("Tiempo")
        self.ax[1].set_ylabel("Valor del integrando")
        self.ax[1].legend(loc='upper right')
        self.ax[1].grid()
        self.ax[1].set_ylim(top=np.max(self.inmovil)*np.max(self.movil)**2 * 1.25)

        self.ax[2].set_xlabel("Retraso")
        self.ax[2].set_ylabel("Intensidad (u.a.)")
        self.ax[2].legend(loc='upper right')
        self.ax[2].grid()
        self.ax[2].set_xlim([self.delays[0], self.delays[-1]])
        if np.min(self.valores_autocorrelacion) < 0.01:
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.25)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.25)

    def pause(self, event):
        """
        Pause button updating function. Changes label of the button between 'Pause' and 'Play'

        Parameters
        ---------
        event : matplotlib click event
            Necessary parameter for button update function
        """        
        if self.paused:
            self.pause_button.label.set_text('Pause')
            self.animation.resume()

        else:
            self.pause_button.label.set_text('Play')
            self.animation.pause()

        self.paused = not self.paused


#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################
#! ################################################################################################################################################################################################


class animacionFRAC:
    def __init__(self, t, Δt, delays, funcion, *args, interval=10):
        self.t = t
        self.Δt = Δt
        self.delays = delays
        self.func = funcion
        self.args = args

        self.fig, self.ax = plt.subplots(3,1)

        self.interval = interval
        self.animation = animation.FuncAnimation(self.fig, self.__call__, interval=self.interval)
        self.frame = 0

        self.pause_button = Button(plt.axes([0.45, 0.9, 0.1, 0.05]), 'Pause', hovercolor='paleturquoise', color='slategrey')
        self.pause_button.on_clicked(self.pause)
        self.paused = False

        self.campo_inmovil = self.func(self.t, *self.args) # Pulso que se queda inmovil para hacer la autocorrelacion
        self.intensidad_inmovil = np.abs(self.campo_inmovil)**2

        self.campo_movil = self.func(self.t - self.delays[0], *self.args) # Pulso que se irá moviendo a lo largo de la animación
        self.intensidad_movil = np.abs(self.campo_movil)**2

        self.valores_autocorrelacion = FRAC(self.delays, self.Δt, self.func, self.t, *self.args)

        self.prepareCanvas()

    def __call__(self, frame):
        self.frame +=1
        if self.interval * self.frame > np.size(self.delays):
            # self.animation.pause()
            self.frame = 0
            # return

        self.campo_movil = self.func(self.t - self.delays[self.interval * self.frame], *self.args)
        self.intensidad_movil = np.abs(self.campo_movil)**2
        self.line_movil.set_ydata(self.intensidad_movil)

        self.integrando = np.abs((self.campo_movil+ self.campo_inmovil)**2)**2
        self.line_integrando.set_ydata(self.integrando)

        self.line_correlacion.set_xdata(self.delays[0 : self.interval * self.frame])
        self.line_correlacion.set_ydata(self.valores_autocorrelacion[0 : self.interval * self.frame])

        if self.frame > 1:
            self.area1.remove()
            self.area2.remove()
            self.area3.remove()

        self.area1 = self.ax[0].fill_between(self.t, 0, self.intensidad_inmovil, where=self.intensidad_movil > self.intensidad_inmovil, alpha=0.5, color='mediumseagreen')
        self.area2 = self.ax[0].fill_between(self.t, 0, self.intensidad_movil, where=self.intensidad_inmovil > self.intensidad_movil, alpha=0.5, color='mediumseagreen')
        self.area3 = self.ax[1].fill_between(self.t, 0, self.integrando, where=self.integrando > 0, alpha=0.5, color='mediumseagreen')

    def prepareCanvas(self):
        self.ax[0].plot(self.t, self.intensidad_inmovil, label="I(t)")
        self.line_movil, = self.ax[0].plot(self.t, self.intensidad_movil, label="I(t-τ)")

        self.line_correlacion, = self.ax[2].plot(self.delays[0], self.valores_autocorrelacion[0], color='red', label="FRAC")

        self.ax[0].set_xlabel("Tiempo")
        self.ax[0].set_ylabel("Intensidad (u.a.)")
        self.ax[0].legend(loc='upper right')
        self.ax[0].grid()


        self.line_integrando, = self.ax[1].plot(self.t, np.abs((self.campo_movil+ self.campo_inmovil)**2)**2, color='green', label="|E²(t) + 2E(t)E(t-τ) + E²(t-τ)|")
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
            self.ax[2].set_ylim(bottom=-0.25, top=np.max(self.valores_autocorrelacion) * 1.25)
        else:
            self.ax[2].set_ylim(np.min(self.valores_autocorrelacion), top=np.max(self.valores_autocorrelacion) * 1.25)

    def pause(self, event):
        """
        Pause button updating function. Changes label of the button between 'Pause' and 'Play'

        Parameters
        ---------
        event : matplotlib click event
            Necessary parameter for button update function
        """        
        if self.paused:
            self.pause_button.label.set_text('Pause')
            self.animation.resume()

        else:
            self.pause_button.label.set_text('Play')
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

    t, Δt = np.linspace(-60, 60, num=4096, retstep=True)
    delays = np.linspace(-60, 60, num=4096)
    β = 0.1
    α = 1.6 / 4

    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, delays, intensidad_ejemplo, β, α, interval=20)
    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, delays, intensidad_ejemplo, β, α, interval=20)
    plt.show()

    #! ########################## PRUEBAS CON PULSOS GAUSSIANOS ######################################################

    # Parámetros de la medida
    numero_de_muestras = 4 * 4096
    duracion_temporal = 4 * 10 # Tiempo total de medida de la señal (ps)
    frecuencia_muestreo = numero_de_muestras / duracion_temporal # En THz

    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=numero_de_muestras, retstep=True) # Vector de tiempos. Guardamos la separación entre datos (inversa de la frecuencia de muestreo)

    # -- Parámetros del pulso --
    t0 = 0 # Tiempo en el que el pulso tiene su máximo (ps)
    A = 1 # Amplitud del pulso
    λ_0 = 1.55 # Longitud de onda de ejemplo (en micrómetros)
    ω_0 = 2 * np.pi * constants.c * 1e-12 / (λ_0 * 1e-6) # Frecuencia angular del pulso (rad / ps)
    φ_0 =  0 # Fase del pulso (constante en este caso)
    τ = 1 # Duración del pulso (ps)
    a = 3 * np.pi / 4 # Parámetro de chirp del pulso 
    φ =  a * (t - t0) * (t - t0) / (τ * τ ) # Chirpeo lineal

    # Comprobamos que se cumple el teorema de muestreo de Nyquist para que los resultados sean correctos
    print(f"Frecuencia de muestreo: {frecuencia_muestreo} [THz]")
    print(f"Frecuencia de la señal: {convertir(ω_0, 'frecuencia angular', 'frecuencia')} [THz]")
    if frecuencia_muestreo/2 > convertir(ω_0, 'frecuencia angular', 'frecuencia'):
        print("Se cumple el teorema de Nyquist")
    else:
        print("¡Atención! No se cumple el teorema de muestreo de Nyquist")

    # Definimos el aray de los delays (es igual que el de tiempos)
    delays = np.linspace(-duracion_temporal, duracion_temporal, num=numero_de_muestras)

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


    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, t, intensidad, t0, A, τ, ω_0, φ, interval=40)

    plt.show()

    animacion_A2 = animacionAutocorrelacion2orden(t, Δt, t, intensidad_doble_pulso, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ, interval=40)

    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, t, intensidad, t0, A, τ, ω_0, φ, interval=40)

    plt.show()

    animacion_A3 = animacionAutocorrelacion3orden(t, Δt, t, intensidad_doble_pulso, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ, interval=40)

    plt.show()

    animacion_FRAC = animacionFRAC(t, Δt, delays, pulso_gaussiano, t0, A, τ, ω_0, φ, interval=40)

    plt.show()

    animacion_FRAC = animacionFRAC(t, Δt, delays, doble_pulso_gaussiano, -5, A, τ, ω_0, φ, 3, A/2, τ/4, ω_0, φ, interval=40)

    plt.show()