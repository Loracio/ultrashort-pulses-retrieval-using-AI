"""
Este archivo contiene la clase visualizador_resultados,
con la que podremos ver cómoda e interactivamente los 
resultados de la traza obtenidos por la NN y compararlos
con los valores teóricos de las trazas de los pulsos.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backend_bases import Event
import path_helper # Para poder cargar el módulo de 'core' sin tener que cambiar el path
from core import *
from lee_database import formateador

plt.rcParams.update({'font.size': 14}) # Tamaño de la fuente del plot

class visualizador_resultados():
    """
    Esta clase tiene como objetivo realizar un plot interactivo en el que se puedan 
    visualizar los resultados del cálculo de la traza de un pulso mediante una NN
    y compararlo con la traza teórica.

    La sintaxis y estructuración no está demasiado trabajada, así que mejor no fijarse
    mucho en este código, simplemente lo he escrito rápido para tener una forma cómoda
    de ver los datos.
    """
    def __init__(self, t, Δt, N, NUMERO_PULSOS, direccion_archivo, direccion_modelo, conv=False):
        self.t = t
        self.Δt = Δt
        self.N = self.t.size
        self.NUMERO_PULSOS = NUMERO_PULSOS

        self.frecuencias = frecuencias_DFT(self.N, self.Δt)
        self.ω = convertir(self.frecuencias, 'frecuencia', 'frecuencia angular')
        self.Δω = 2 * np.pi / (self.N * self.Δt) # Relación reciprocidad

        self.M = 2 * self.N - 1 # Número de delays totales
        self.delays = self.Δt * np.linspace(-(self.N - 1), (self.N - 1) , num=self.M, dtype=int)


        self.direccion_archivo = direccion_archivo
        self.direccion_modelo = direccion_modelo

        self.model = tf.keras.models.load_model(self.direccion_modelo)

        self.TBP, self.campos_concat, self.T = formateador(self.direccion_archivo, self.N, self.NUMERO_PULSOS)
        if conv:
            self.campos_concat = self.campos_concat.reshape((self.NUMERO_PULSOS, self.N, 2, 1))
        self.T_pred = self.model.predict(self.campos_concat)

        self.errores_traza = np.zeros(self.NUMERO_PULSOS)
        self.calcula_errores_traza()

    def calcula_μ(self, indice):
        """
        Calcula el factor de escala μ, que se usa para calcular el error de la traza.
        Su expresión es la siguiente:
            μ = ∑ₘₙ (Tₘₙᵐᵉᵃˢ · Tₘₙᵖʳᵉᵈ) / (∑ₘₙ Tₘₙᵖʳᵉᵈ²)

        Donde Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²}
        y Tₘₙᵖʳᵉᵈ es la traza obtenida como resultado de la red neuronal.
        """
        denominador = np.sum(self.T_pred[indice] * self.T_pred[indice])
        if denominador == 0:
            return float('inf')
        return np.sum(self.T[indice] * self.T_pred[indice]) / np.sum(self.T_pred[indice] * self.T_pred[indice])

    def calcula_residuos(self, indice):
        """
        Calcula la suma de los cuadrados de los residuos, dada por:
            r = ∑ₘₙ [Tₘₙᵐᵉᵃˢ - μ·Tₘₙᵖʳᵉᵈ]²

        Donde Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²}
        y Tₘₙᵖʳᵉᵈ es la traza obtenida como resultado de la red neuronal.
        """
        μ = self.calcula_μ(indice)
        if μ == float('inf'):
            return μ

        diferencia = np.ravel(self.T[indice] - self.calcula_μ(indice) * self.T_pred[indice])
        return np.sum(diferencia * diferencia)

    def calcula_R(self, indice):
        """
        Calcula el error de la traza, R, dado por:
            R = r½ / [M·N (maxₘₙ Tₘₙᵐᵉᵃˢ)²]½

        Donde r es la suma de los cuadrados de los residuos,
        Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²},
        N es el numero de muestras del pulso
        M = 2·N - 1 es el numero total de retrasos introducidos al pulso
        """
        return np.sqrt(self.calcula_residuos(indice) / (self.M * self.N * self.T[indice].max()**2))

    def calcula_errores_traza(self):
        """
        Calcula los errores entre la traza teórica y la traza obtenida por la NN
        """
        for i in range(NUMERO_PULSOS):
            self.errores_traza[i] = self.calcula_R(i)

    def model_summary(self):
        """
        Imprime por pantalla un resumen del modelo de NN empleado
        """
        self.model.summary()  


    def plot(self):
        """
        Plot interactivo para ver los resultados de la traza de la NN comparados
        con la traza teórica.

        El código no está muy bien estructurado, simplemente es funcional.

        Una cosa a pulir es la colocación de los distintos elementos del plot para
        que se puedan ver con distintas resoluciones de pantalla sin problema.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_axis_off()

        self.ax0 = plt.axes([.25/4, 0.3, 0.25, 0.5])
        self.ax1 = plt.axes([(.5/4 + 0.25), 0.3, 0.25, 0.5])
        self.ax2= plt.axes([(1 - .25/4 - 0.25), 0.3, 0.25, 0.5])

        plt.figtext(0.75, 0.2, 'Control visualización', fontweight='bold')
        self.next_button = Button(plt.axes([0.8, 0.125, 0.075, 0.04]), 'Siguiente', hovercolor='aquamarine', color='lightcyan')
        self.next_button.on_clicked(self.next_result)
        self.prev_button = Button(plt.axes([0.7, 0.125, 0.075, 0.04]), 'Anterior', hovercolor='aquamarine', color='lightcyan')
        self.prev_button.on_clicked(self.prev_result)


        plt.figtext(0.3, 0.2, 'Modo visualización', fontweight='bold')
        self.random_button = Button(plt.axes([0.2, 0.125, 0.075, 0.04]), 'Aleatorio', hovercolor='aquamarine', color='mediumseagreen')
        self.random_button.on_clicked(self.display_random)
        self.best_button = Button(plt.axes([0.3, 0.125, 0.075, 0.04]), 'Mejores', hovercolor='aquamarine', color='lightgrey')
        self.best_button.on_clicked(self.display_best)
        self.worst_button = Button(plt.axes([0.4, 0.125, 0.075, 0.04]), 'Peores', hovercolor='aquamarine', color='lightgrey')
        self.worst_button.on_clicked(self.display_worst)

        self.best_index_order = np.argsort(self.errores_traza) # Resultados ordenados de mejor a peor
        self.worst_index_order = np.flip(self.best_index_order) # Resultados ordenados de peor a mejor
        self.random_index_order = np.random.permutation(self.NUMERO_PULSOS) # Resultados de forma aleatoria

        self.last_index = 0
        self.previously_clicked = None
        self.mode = 'random'

        self.im0 = self.ax0.pcolormesh(self.frecuencias, self.delays, self.T[0][:].reshape(2*N - 1, N) / np.max(self.T[0][:]), cmap='inferno')
        self.fig.colorbar(self.im0, ax=self.ax0)
        self.ax0.set_xlabel("Frecuencia (1/ps)")
        self.ax0.set_ylabel("Retraso (ps)")
        self.ax0.set_title("Traza real del pulso")

        self.im1 = self.ax1.pcolormesh(self.frecuencias, self.delays, self.T_pred[0].reshape(2*N - 1, N) / np.max(self.T_pred[0][:]), cmap='inferno')
        self.fig.colorbar(self.im1, ax=self.ax1)
        self.ax1.set_xlabel("Frecuencia (1/ps)")
        self.ax1.set_ylabel("Retraso (ps)")
        self.ax1.set_title("Traza predicha por la NN")

        self.im2 = self.ax2.pcolormesh(self.frecuencias, self.delays, np.abs(self.T[0] - self.T_pred[0]).reshape(2*N - 1, N) , cmap='RdBu')
        self.colorbar2 = self.fig.colorbar(self.im2, ax=self.ax2)
        self.ax2.set_xlabel("Frecuencia (1/ps)")
        self.ax2.set_ylabel("Retraso (ps)")
        self.ax2.set_title("Diferencia")

        self.fig.suptitle(f"TBP = {self.TBP[0]:.2f}      Error en la traza = {self.errores_traza[0]:.2E}", fontweight='bold')


    def display_random(self, event):
        """
        Acción de activar el botón de mostrar los resultados aleatoriamente
        """
        self.random_button.color = "mediumseagreen"
        self.worst_button.color = "lightgrey"
        self.best_button.color = "lightgrey"

        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'random'

        self.next_result(Event('button_press_event', self.fig))


    def display_best(self, event):
        """
        Acción de activar el botón de mostrar los resultados de mejor a peor
        """
        self.random_button.color = "lightgrey"
        self.worst_button.color = "lightgrey"
        self.best_button.color = "mediumseagreen"
        
        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'best'

        self.next_result(Event('button_press_event', self.fig))

    def display_worst(self, event):
        """
        Acción de activar el botón de mostrar los resultados de peor a mejor
        """
        self.random_button.color = "lightgrey"
        self.worst_button.color = "mediumseagreen"
        self.best_button.color = "lightgrey"

        self.previously_clicked = None
        self.last_index = 0
        self.mode = 'worst'

        self.next_result(Event('button_press_event', self.fig))


    def next_result(self, event):
        """
        Acción de mostrar el siguiente resultado según el modo actual.
        
        Tiene bastante código repetido y podría escribirse de una manera más
        limpia, pero es funcional.
        """

        if self.previously_clicked == 'prev':
            self.last_index += 2
        
        self.update_plot()

        self.previously_clicked = 'next'
        self.last_index += 1

        plt.draw()            

    def prev_result(self, event):
        """
        Acción de mostrar el anterior resultado según el modo actual.
        
        Tiene bastante código repetido y podría escribirse de una manera más
        limpia, pero es funcional.
        """

        if self.previously_clicked == 'next':
            self.last_index -= 2

        self.update_plot()

        self.previously_clicked = 'prev'
        self.last_index -= 1

        plt.draw()

    def update_plot(self):
        match self.mode:
            case 'random': i = self.random_index_order[self.last_index]

            case 'best': i = self.best_index_order[self.last_index]
            
            case 'worst': i = self.worst_index_order[self.last_index]

        self.fig.suptitle(f"TBP = {self.TBP[i]:.2f}      Error en la traza = {self.errores_traza[i]:.2E}", fontweight='bold')

        self.im0.set_array(self.T[i].reshape(2*self.N - 1, self.N) / np.max(self.T[i]) )
        self.im0.set_clim(0, 1)

        max_Tpred = np.max(self.T_pred[i])
        if max_Tpred != 0:
            self.im1.set_array(self.T_pred[i].reshape(2*self.N - 1, self.N) / max_Tpred)
            diff = np.abs(self.T[i] / np.max(self.T[i]) - self.T_pred[i] / max_Tpred).reshape(2*self.N - 1, self.N)
        else:
            self.im1.set_array(np.zeros((2*self.N - 1, self.N)))
            diff = np.abs(self.T[i] / np.max(self.T[i])).reshape(2*self.N - 1, self.N)
        self.im1.set_clim(0, 1)

        self.im2.set_array(diff)
        self.im2.set_array(diff)
        self.im2.set_clim(np.min(diff), np.max(diff))
        

if __name__ == '__main__':

    N = 128
    NUMERO_PULSOS = 1000

    duracion_temporal = 1 # Tiempo total de medida de los pulsos de la base de datos (ps)
    t, Δt = np.linspace(-duracion_temporal/2, duracion_temporal/2, num=N, retstep=True)

    # Ruta al fichero de la base de datos
    direccion_archivo = f"./IA/DataBase/{NUMERO_PULSOS}_pulsos_aleatorios_N{N}.csv"

    """
    Vamos a ver los resultados de una red muy simple,
    construida con una única capa densa de 64 neuronas
    """

    direccion_modelo = "./IA/NN_models/pulse_trace_model_simple_dense.h5"

    ver_densa = visualizador_resultados(t, Δt, N, NUMERO_PULSOS, direccion_archivo, direccion_modelo, conv=False)
    ver_densa.model_summary()
    ver_densa.plot()

    print(f"Error medio en la traza: {np.mean(ver_densa.errores_traza)}")
    print(f"Desviación: {np.std(ver_densa.errores_traza)} (contiene términos infinitos)")

    plt.show()

    """
    Comparamos los resultados con una capa que contiene capas
    convolucionales
    """

    direccion_modelo = "./IA/NN_models/pulse_trace_model_convolucional.h5"

    ver_convolucional = visualizador_resultados(t, Δt, N, NUMERO_PULSOS, direccion_archivo, direccion_modelo, conv=True)
    ver_convolucional.model_summary()
    ver_convolucional.plot()

    print(f"Error medio en la traza: {np.mean(ver_convolucional.errores_traza)}")
    print(f"Desviación: {np.std(ver_convolucional.errores_traza)}")

    plt.show()