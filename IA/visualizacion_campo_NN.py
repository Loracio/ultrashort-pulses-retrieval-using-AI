"""
Este archivo contiene la clase visualizador_resultados,
con la que podremos ver cómoda e interactivamente los 
resultados de la traza obtenidos por la NN y compararlos
con los valores teóricos de las trazas de los pulsos.

Lo que hago es calcular para cada campo obtenido como
resultado de la red neuronal su traza correspondiente,
y pintarla frente a la traza real del pulso para ver
el error.
"""

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
            self.T = self.T.reshape((self.NUMERO_PULSOS, 2*self.N - 1, self.N, 1))
        self.campos_concat_pred = self.model.predict(self.T)
        self.T = self.T.reshape(self.NUMERO_PULSOS, 2*self.N - 1, self.N)
        self.T_pred = np.zeros((self.NUMERO_PULSOS, 2*self.N - 1, self.N))
        self.calcula_trazas_pred()

        self.errores_traza = np.zeros(self.NUMERO_PULSOS)
        self.calcula_errores_traza()

    def calcula_trazas_pred(self):
        for i in range(self.NUMERO_PULSOS):
            pulso = self.campos_concat_pred[i][:self.N] + 1j * self.campos_concat_pred[i][self.N:]
            self.T_pred[i] = traza(pulso, self.t, self.Δt, self.N)

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

        self.ax0 = plt.axes([.25/4, 0.1, 0.25, 0.35])
        self.ax1 = plt.axes([(.5/4 + 0.25), 0.1, 0.25, 0.35])
        self.ax2 = plt.axes([(1 - .25/4 - 0.25), 0.1, 0.25, 0.35])
        self.ax3 = plt.axes([(.5/4 + 0.25 - 0.025), 0.55, 0.25, 0.35])
        self.ax4 = plt.axes([(1 - .25/4 - 0.25), 0.55, 0.25, 0.35])
        self.twin_ax3 = self.ax3.twinx()
        self.twin_ax4 = self.ax4.twinx()

        plt.figtext(0.1, 0.7, 'Control visualización', fontweight='bold')
        self.next_button = Button(plt.axes([0.15, 0.6, 0.075, 0.04]), 'Siguiente', hovercolor='aquamarine', color='lightcyan')
        self.next_button.on_clicked(self.next_result)
        self.prev_button = Button(plt.axes([0.05, 0.6, 0.075, 0.04]), 'Anterior', hovercolor='aquamarine', color='lightcyan')
        self.prev_button.on_clicked(self.prev_result)


        plt.figtext(0.1, 0.9, 'Modo visualización', fontweight='bold')
        self.random_button = Button(plt.axes([0.025, 0.8, 0.075, 0.04]), 'Aleatorio', hovercolor='aquamarine', color='mediumseagreen')
        self.random_button.on_clicked(self.display_random)
        self.best_button = Button(plt.axes([0.125, 0.8, 0.075, 0.04]), 'Mejores', hovercolor='aquamarine', color='lightgrey')
        self.best_button.on_clicked(self.display_best)
        self.worst_button = Button(plt.axes([0.225, 0.8, 0.075, 0.04]), 'Peores', hovercolor='aquamarine', color='lightgrey')
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

        pulso_pred = self.campos_concat_pred[0][:self.N] + 1j * self.campos_concat_pred[0][self.N:]
        I_pulso_pred = np.abs(pulso_pred)**2
        espectro_pred = DFT(pulso_pred, self.t, self.Δt, self.ω, self.Δω)
        I_espectro_pred = np.abs(espectro_pred)**2

        pulso_db = self.campos_concat[0][:self.N] + 1j * self.campos_concat[0][self.N:]
        I_pulso_db = np.abs(pulso_db)**2
        espectro_db = DFT(pulso_db, self.t, self.Δt, self.ω, self.Δω)
        I_espectro_db = np.abs(espectro_db)**2

        fase_campo_pred = np.unwrap(np.angle(pulso_pred)) 
        fase_campo_pred -=  media(fase_campo_pred, I_pulso_pred)
        fase_campo_pred = np.where(I_pulso_db < 1e-10, np.nan, fase_campo_pred)

        fase_espectro_pred = np.unwrap(np.angle(espectro_pred)) 
        fase_espectro_pred -=  media(fase_espectro_pred, I_espectro_pred)
        fase_espectro_pred = np.where(I_espectro_db < 1e-10, np.nan, fase_espectro_pred)

        fase_campo_db = np.unwrap(np.angle(pulso_db)) 
        fase_campo_db -=  media(fase_campo_db, I_pulso_db)
        fase_campo_db = np.where(I_pulso_db < 1e-10, np.nan, fase_campo_db)

        fase_espectro_db = np.unwrap(np.angle(espectro_db)) 
        fase_espectro_db -=  media(fase_espectro_db, I_espectro_db)
        fase_espectro_db = np.where(I_espectro_db < 1e-10, np.nan, fase_espectro_db)


        self.line_I_pulso_db, = self.ax3.plot(self.t,I_pulso_db / np.max(I_pulso_db), color='blue', linewidth=3, alpha=0.5, label='Intensidad campo base de datos')
        self.line_fase_campo_db, = self.twin_ax3.plot(self.t, fase_campo_db, '-.', color='red',alpha=0.5)
        self.ax3.plot(np.nan, '-.', label='Fase', color='red')
        self.line_I_pulso_pred, = self.ax3.plot(self.t, I_pulso_pred / np.max(I_pulso_pred), color='orange', label='Intensidad campo predicho')
        self.line_fase_campo_pred, = self.twin_ax3.plot(self.t, fase_campo_pred, '-.', color='violet')
        self.ax3.plot(np.nan, '-.', label='Fase campo predicho', color='red')
        self.ax3.set_xlabel("Tiempo (ps)")
        self.ax3.set_ylabel("Intensidad (u.a.)")
        self.twin_ax3.set_ylabel("Fase (rad)")
        self.ax3.set_title("Dominio temporal")
        self.ax3.grid()
        self.twin_ax3.set_ylim(-2 * np.pi, 2 * np.pi)

        self.line_I_espectro_db, = self.ax4.plot(self.frecuencias, I_espectro_db / np.max(I_espectro_db), color='blue', linewidth=3, alpha=0.5, label='Intensidad espectral base de datos')
        self.line_fase_espectro_db, = self.twin_ax4.plot(self.frecuencias, fase_espectro_db, '-.', color='red', alpha=0.5)
        self.ax4.plot(np.nan, '-.', label='Fase', color='red')
        self.line_I_espectro_pred, = self.ax4.plot(self.frecuencias, I_espectro_pred / np.max(I_espectro_pred), color='orange', label='Intensidad espectral predicha')
        self.line_fase_espectro_pred, = self.twin_ax4.plot(self.frecuencias, fase_espectro_pred, '-.', color='violet')
        self.ax4.plot(np.nan, '-.', label='Fase espectral predicha', color='red')
        self.ax4.set_xlabel("Frecuencia (1 / ps)")
        self.ax4.set_ylabel("Intensidad (u.a.)")
        self.twin_ax4.set_ylabel("Fase (rad)")
        self.ax4.set_title("Dominio frecuencial")
        self.ax4.grid()
        self.twin_ax4.set_ylim(-2 * np.pi, 2 * np.pi)

        self.fig.legend(*self.ax3.get_legend_handles_labels(), loc='upper right', ncols=4)

        self.pulse_info_text = self.fig.text(0.02, 0.95, f"TBP = {self.TBP[0]:.2f}      Error en la traza = {self.errores_traza[0]:.2E}", fontweight='bold')


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

        self.pulse_info_text.set_text(f"TBP = {self.TBP[i]:.2f}      Error en la traza = {self.errores_traza[i]:.2E}")

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
        self.im2.set_clim(np.min(diff), np.max(diff))

        pulso_pred = self.campos_concat_pred[i][:self.N] + 1j * self.campos_concat_pred[i][self.N:]
        pulso_db = self.campos_concat[i][:self.N] + 1j * self.campos_concat[i][self.N:]

        I_pulso_pred = np.abs(pulso_pred)**2
        espectro_pred = DFT(pulso_pred, self.t, self.Δt, self.ω, self.Δω)
        I_espectro_pred = np.abs(espectro_pred)**2

        I_pulso_db = np.abs(pulso_db)**2
        espectro_db = DFT(pulso_db, self.t, self.Δt, self.ω, self.Δω)
        I_espectro_db = np.abs(espectro_db)**2

        fase_campo_pred = np.unwrap(np.angle(pulso_pred)) 
        fase_campo_pred -=  media(fase_campo_pred, I_pulso_pred)
        fase_campo_pred = np.where(I_pulso_db < 1e-3, np.nan, fase_campo_pred)

        fase_espectro_pred = np.unwrap(np.angle(espectro_pred)) 
        fase_espectro_pred -=  media(fase_espectro_pred, I_espectro_pred)
        fase_espectro_pred = np.where(I_espectro_db < 1e-3, np.nan, fase_espectro_pred)

        fase_campo_db = np.unwrap(np.angle(pulso_db)) 
        fase_campo_db -=  media(fase_campo_db, I_pulso_db)
        fase_campo_db = np.where(I_pulso_db < 1e-3, np.nan, fase_campo_db)

        fase_espectro_db = np.unwrap(np.angle(espectro_db)) 
        fase_espectro_db -=  media(fase_espectro_db, I_espectro_db)
        fase_espectro_db = np.where(I_espectro_db < 1e-3, np.nan, fase_espectro_db)

        self.line_I_pulso_pred.set_ydata(I_pulso_pred / np.max(I_pulso_pred))
        self.line_fase_campo_pred.set_ydata(fase_campo_pred)
        self.line_I_espectro_pred.set_ydata(I_espectro_pred / np.max(I_espectro_pred))
        self.line_fase_espectro_pred.set_ydata(fase_espectro_pred)

        self.line_I_pulso_db.set_ydata(I_pulso_db / np.max(I_pulso_db))
        self.line_fase_campo_db.set_ydata(fase_campo_db)
        self.line_I_espectro_db.set_ydata(I_espectro_db / np.max(I_espectro_db))
        self.line_fase_espectro_db.set_ydata(fase_espectro_db)
        

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

    direccion_modelo = "./IA/NN_models/campo_model_simple_dense.h5"

    ver_densa = visualizador_resultados(t, Δt, N, NUMERO_PULSOS, direccion_archivo, direccion_modelo, conv=False)
    ver_densa.model_summary()
    ver_densa.plot()

    print(f"Error medio en la traza: {np.mean(ver_densa.errores_traza)}")
    print(f"Desviación: {np.std(ver_densa.errores_traza)}")

    plt.show()
    
    """
    Comparamos los resultados con una capa que contiene capas
    convolucionales
    """

    direccion_modelo = "./IA/NN_models/campo_model_convolucional.h5"

    ver_convolucional = visualizador_resultados(t, Δt, N, NUMERO_PULSOS, direccion_archivo, direccion_modelo, conv=True)
    ver_convolucional.model_summary()
    ver_convolucional.plot()

    print(f"Error medio en la traza: {np.mean(ver_convolucional.errores_traza)}")
    print(f"Desviación: {np.std(ver_convolucional.errores_traza)}")

    plt.show()