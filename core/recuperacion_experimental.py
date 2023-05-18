import numpy as np
import matplotlib.pyplot as plt
from .fourier import *
from .unidades import convertir
from .utiles import media

class retrieverExpBase():
    """
    Clase base que inicializa los parámetros para un algoritmo de reconstrucción.
    Genera los arrays de frecuencias, delays, calcula la traza experimental; y demás parámetros
    que serán usados una y otra vez en el algoritmo de recuperación.

    La idea de esta clase es ser usada para inicializar diferentes algoritmos de reconstrucción, creando
    clases 'hijo' a partir de esta.
    """

    def __init__(self, τ, Δτ, ω, Δω, traza_experimental):
        """
        Inicializa los parámetros necesarios para utilizar el algoritmo de reconstrucción.
        """
        # Inicialización de parámetros del pulso
        self.t = τ
        self.Δt = Δτ
        self.N = self.t.size
        self.ω = ω
        self.Δω = Δω
        self.M = self.N
        self.delays = τ
        self.bin_delays = np.array([round(i / self.Δt) for i in self.delays])

        # Calculamos los factores de fase r_n y s_j para agilizar cálculos de transformadas de Fourier
        if self.t[0] == 0.0:
            self.r_n = 1.0
        else:
            self.r_n = np.exp(-1j * np.arange(np.size(self.ω)) * self.t[0] * self.Δω)

        if self.ω[0] == 0.0:
            self.s_j = 1.0
        else:
            self.s_j = np.exp(-1j * self.ω[0] * self.t)

        self.r_n_conj = self.r_n.conjugate()
        self.s_j_conj = self.s_j.conjugate()


        # Parámetros para el algoritmo de reconstrucción
        self.Tmn_medido = traza_experimental
        self.Tmn_medido_max_cuadrado = self.Tmn_medido.max()**2 # Será utilizado muchas veces en el cálculo del error de la traza

    def calcula_Smk(self):
        """
        Calcula el operador señal del pulso, dado por:
            Sₘₖ = E(tₖ)·E(tₖ - τₘ)
        
        Donde m = 0, ... , M - 1 y k = 0, ..., N - 1
        """

        for m, τ in enumerate(self.bin_delays):
            for k in range(self.N):
                if (k - τ) < 0 or (k - τ) > (self.N - 1):
                    self.Smk[m][k] = 0
                else: 
                    self.Smk[m][k] = self.campo[k] * self.campo[k - τ]


    def calcula_Smn(self):
        """
        Calcula la transformada de Fourier del operador señal del pulso. Es decir:
            Sₘₙ = ℱ{Sₘₖ²}
        """
        for τ in range(self.M):
            self.Smn[τ][:] = DFT(self.Smk[τ][:], self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j)

    def calcula_Tmn(self):
        """
        Calcula la traza del pulso candidato de la iteración actual, dada por:
            Tₘₙ = |Sₘₙ|² = ℱ{|Sₘₖ|²}

        Donde Sₘₖ es el operador señal del pulso, dado por: Sₘₖ = E(tₖ)·E(tₖ - τₘ)
        Y Sₘₙ es su transformada de Fourier Sₘₙ = ℱ{Sₘₖ²} 
        """
        for τ in range(self.M):
            self.Tmn[τ][:] = np.abs(self.Smn[τ][:])**2

    def calcula_μ(self):
        """
        Calcula el factor de escala μ, que ha de ser obtenido en cada iteración para 
        calcular el error de la traza. Su expresión es la siguiente:
            μ = ∑ₘₙ (Tₘₙᵐᵉᵃˢ · Tₘₙ) / (∑ₘₙ Tₘₙ²)

        Donde Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²}
        y Tₘₙ es la traza del pulso candidato de la actual iteración Tₘₙ = |Sₘₙ|² = ℱ{|Sₘₖ|²}
        """
        self.μ = np.sum(self.Tmn_medido * self.Tmn) / np.sum(self.Tmn * self.Tmn)

    def calcula_residuos(self):
        """
        Calcula la suma de los cuadrados de los residuos, dada por:
            r = ∑ₘₙ [Tₘₙᵐᵉᵃˢ - μ·Tₘₙ]²

        Donde Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²}
        y Tₘₙ es la traza del pulso candidato de la actual iteración Tₘₙ = |Sₘₙ|² = ℱ{|Sₘₖ|²}
        """
        diferencia = np.ravel(self.Tmn_medido - self.μ * self.Tmn)
        self.r = np.sum(diferencia * diferencia)

    def calcula_R(self):
        """
        Calcula el error de la traza, R, dado por:
            R = r½ / [M·N (maxₘₙ Tₘₙᵐᵉᵃˢ)²]½

        Donde r es la suma de los cuadrados de los residuos,
        Tₘₙᵐᵉᵃˢ es la traza del pulso medido experimentalmente Tₘₙᵐᵉᵃˢ = |Sₘₙᵐᵉᵃˢ|² = ℱ{|Sₘₖᵐᵉᵃˢ|²},
        N es el numero de muestras del pulso
        M = 2·N - 1 es el numero total de retrasos introducidos al pulso
        """
        self.R = np.sqrt(self.r / (self.M * self.N * self.Tmn_medido_max_cuadrado))

    def plot(self, λ0):
        """
        Representa las intensidades temporales y espectrales tanto del pulso original como del pulso solución,
        además de sus correspondientes trazas.

        Devuelve:
            tuple(matplotlib Figure, matplotlib Axis)
        """

        self.fig, self.ax = plt.subplots(2,2)

        twin_ax00 = self.ax[0][0].twinx()
        twin_ax01 = self.ax[0][1].twinx()
        

        self.I_campo_solucion = np.abs(self.campo)**2

        self.I_espectral_solucion = np.abs(DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j))**2

        self.fase_campo_solucion = np.unwrap(np.angle(self.campo)) 
        self.fase_campo_solucion -=  media(self.fase_campo_solucion, self.I_campo_solucion)

        self.fase_espectro_solucion = np.unwrap(np.angle(DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j))) 
        self.fase_espectro_solucion -=  media(self.fase_espectro_solucion, self.I_espectral_solucion)

        self.ax[0][0].plot(self.t, self.I_campo_solucion, color='orange', label='Intensidad campo recuperado')
        twin_ax00.plot(self.t, self.fase_campo_solucion, '-.', color='violet')
        self.ax[0][0].plot(np.nan, '-.', label='Fase campo recuperado', color='violet')
        self.ax[0][0].set_xlabel("Tiempo (fs)")
        self.ax[0][0].set_ylabel("Intensidad (u.a.)")
        twin_ax00.set_ylabel("Fase (rad)")
        self.ax[0][0].set_title("Dominio temporal")
        self.ax[0][0].grid()

        self.ax[0][1].plot(2 * np.pi * 300 / (2 * np.pi * 300 / λ0 + self.ω), self.I_espectral_solucion, color='orange', label='Intensidad espectral recuperada')
        twin_ax01.plot(2 * np.pi * 300 / (2 * np.pi * 300 / λ0 + self.ω), self.fase_espectro_solucion, '-.', color='violet')
        self.ax[0][1].plot(np.nan, '-.', label='Fase espectral recuperada', color='violet')
        self.ax[0][1].set_xlabel("λ (nm)")
        self.ax[0][1].set_ylabel("Intensidad (u.a.)")
        twin_ax01.set_ylabel("Fase (rad)")
        self.ax[0][1].set_title("Dominio frecuencial")
        self.ax[0][1].grid()

        self.fig.legend(*self.ax[0][0].get_legend_handles_labels(), loc='upper center', ncols=4)

        # Plots de la segunda fila: traza del pulso original y del pulso recuperado
        self.calcula_Smk()
        self.calcula_Smn()
        self.calcula_Tmn()

        # Por la manera de coger los delays la traza estaría desplazada si la representamos tal cual, hay que manipularla:
        self.Tmn_medido_normalizado = self.Tmn_medido / np.max(self.Tmn_medido)

        self.Tmn_recuperado_normalizado = self.Tmn / np.max(self.Tmn)

        self.im0 = self.ax[1][0].pcolormesh(2 * np.pi * 300 / λ0 + self.ω, self.t, self.Tmn_medido_normalizado, cmap='YlGnBu_r')
        self.fig.colorbar(self.im0, ax=self.ax[1][0])
        self.ax[1][0].set_xlabel("ω (2π/fs)")
        self.ax[1][0].set_ylabel("τ (fs)")
        self.ax[1][0].set_title("Traza del pulso medido")

        self.im1 = self.ax[1][1].pcolormesh(2 * np.pi * 300 / λ0 + self.ω, self.t, self.Tmn_recuperado_normalizado, cmap='YlGnBu_r')
        self.fig.colorbar(self.im1, ax=self.ax[1][1])
        self.ax[1][1].set_xlabel("ω (2π/fs)")
        self.ax[1][1].set_ylabel("τ (fs)")
        self.ax[1][1].set_title(f"Traza del pulso recuperado, R = {self.solucion_minimo_error[1]:.2E}")

        return self.fig, self.ax

    

class GPA(retrieverExpBase):
    """
    Clase para utilizar el método de proyecciones generalizadas (GPA) para reconstuir un pulso
    a partir de su traza, partiendo de un pulso candidato. 
    
    El método GPA consiste en los siguientes pasos:

        - Paso 1 : proyección sobre Sₘₖ. Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
                        una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
                        la siguiente proyección: S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}

        - Paso 2 : actualización del campo eléctrico, E, mediante un descenso de gradiente. Para ello, definimos Z como la 
                    distancia entre S'ₘₖ y Sₘₖ, es decir, Z = ∑ₘₖ |S'ₘₖ - Sₘₖ|². De esta manera, el descenso de gradiente 
                    vendrá dado por Eⱼ' = Eⱼ - γ·∇Zⱼ ; donde γ es un control del paso del descenso. En el algoritmo GPA 
                    usualmente se realiza una búsqueda lineal para encontrarlo, pero una opción más rápida de encontrar e 
                    igual de válida es tomar γ = Z / ∑ⱼ|∇Zⱼ|². 

        - Paso 3 : calculo de los nuevos parámetros para la siguiente iteración y error en la traza del pulso. Se calculan
                    los nuevos valores del operador señal y la traza del pulso candidato obtenido por descenso de gradiente
                    en el paso 2. Se calcula el error de la traza R, y si se satisface la condición de convergencia o se llega
                    al máximo de iteraciones se para el algoritmo. En caso contrario, se vuelve al paso 1.

    Args:
            t (np.ndarray): array de tiempos equiespaciados Δt
            pulso (np.ndarray[np.complex128]): array con el campo eléctrico del pulso a recuperar
            espectro (np.ndarray[np.complex128], opcional): array con el espectro del pulso a recuperar
    """

    def __init__(self, τ, Δτ, ω, Δω, traza_experimental):
        """
        Llama a la inicialización de los parámetros necesarios para realizar la
        recuperación del pulso: el valor de la traza experimental del pulso,
        el array de frecuencias para realizar saltos entre dominio temporal y frecuencial,
        los factores de fase de las transformadas de Fourier.
        """
        super().__init__(τ, Δτ, ω, Δω, traza_experimental)

    def _inicializa_recuperacion(self, pulso_inicial):
        """
        Inicializa los parámetros necesarios para utilizar el algoritmo de reconstrucción.
        """

        self.campo = np.zeros(self.N, dtype=pulso_inicial.dtype)
        self.campo = pulso_inicial.copy()

        self.Smk = np.zeros((self.M, self.N), dtype=self.campo.dtype)
        self.Smk_siguiente = np.zeros((self.M, self.N), dtype=self.campo.dtype) # Para realizar la primera proyección
        self.calcula_Smk()

        self.Smn = np.zeros((self.M, self.N), dtype=self.campo.dtype)
        self.Smn_siguiente = np.zeros((self.M, self.N), dtype=self.campo.dtype) # Para realizar la primera proyección
        self.calcula_Smn()

        self.Tmn = np.zeros((self.M, self.N), dtype=np.float64)
        self.calcula_Tmn()

        self.μ = None
        self.calcula_μ()

        self.r = None
        self.calcula_residuos()

        self.R = None
        self.calcula_R()

        self.Z = None
        self.gradZ = np.zeros(self.N, dtype=self.campo.dtype)
        self.γ = None

        self.solucion_minimo_error = [np.zeros(self.N, dtype=pulso_inicial.dtype), self.R]


    def recuperacion(self, pulso_inicial, eps, *, max_iter=None):
        """
        Ejecuta el algoritmo de reconstrucción.
        La idea es pasarle un pulso candidato inicial sobre el que inicie la computación del algoritmo.
        Se pasa como argumento la precisión deseada en el error de la traza, y si no, un máximo de iteraciones.

        Dentro de la función se reproduce 'esquematicamente' el algoritmo, desarrollandose internamente en llamadas a funciones,
        tomando el siguiente esquema:

            - Paso 1 : proyección sobre Sₘₖ. Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
                       una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
                       la siguiente proyección: S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}

            - Paso 2 : actualización del campo eléctrico, E, mediante un descenso de gradiente. Para ello, definimos Z como la 
                       distancia entre S'ₘₖ y Sₘₖ, es decir, Z = ∑ₘₖ |S'ₘₖ - Sₘₖ|². De esta manera, el descenso de gradiente 
                       vendrá dado por Eⱼ' = Eⱼ - γ·∇Zⱼ ; donde γ es un control del paso del descenso. En el algoritmo GPA 
                       usualmente se realiza una búsqueda lineal para encontrarlo, pero una opción más rápida de encontrar e 
                       igual de válida es tomar γ = Z / ∑ⱼ|∇Zⱼ|². 

            - Paso 3 : calculo de los nuevos parámetros para la siguiente iteración y error en la traza del pulso. Se calculan
                       los nuevos valores del operador señal y la traza del pulso candidato obtenido por descenso de gradiente
                       en el paso 2. Se calcula el error de la traza R, y si se satisface la condición de convergencia o se llega
                       al máximo de iteraciones se para el algoritmo. En caso contrario, se vuelve al paso 1.

        Args:
            pulso_inicial (np.ndarray[np.complex128]): campo eléctrico del candidato inicial para el algoritmo de recuperación
            eps (float): precisión deseada en el error de la traza del pulso reconstruido
            max_iter (int, opcional): máximo de iteraciones del algoritmo. Por defecto no hay máximo.
        
        Devuelve:
            campo (np.ndarray[np.complex128]): campo eléctrico de la solución obtenida
            espectro (np.ndarray[np.complex128]): espectro de la solución obtenida
        """

        self._inicializa_recuperacion(pulso_inicial)

        if max_iter is None: max_iter = float('inf')
        niter = 0

        while self.R > eps and niter < max_iter:
            
            self.calcula_Smk_siguiente()
            
            self.calcula_Z()
            self.calcula_gradZ_ciclico()
            self.calcula_γ()
            
            self.calcula_campo_siguiente()

            self.calcula_Smk()
            self.calcula_Smn()
            self.calcula_Tmn()
            self.calcula_μ()
            self.calcula_residuos()
            self.calcula_R()
            
            if self.R < self.solucion_minimo_error[1]:
                self.solucion_minimo_error[0] = self.campo.copy()
                self.solucion_minimo_error[1] = self.R
                
            print(f'n={niter+1}, R={self.R}')
            niter += 1

        print("Error final en la traza: ", self.solucion_minimo_error[1])
        self.campo = self.solucion_minimo_error[0].copy()
        

        return self.campo, DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j)

    def calcula_Z(self):
        """
        Calcula Z, dado por la siguiente expresión:
            Z = ∑ₘₖ |S'ₘₖ - Sₘₖ|²

        Donde S'ₘₖ es el operador señal actualizado tras la primera proyección,
        y Sₘₖ es el operador señal del pulso candidato obtenido tras la anterior
        iteración.
        """
        self.Z = np.sum(np.abs(self.Smk_siguiente - self.Smk)**2)

    def calcula_gradZ_ciclico(self):
        """
        Calcula el gradiente de Z, dado por la siguiente expresión:
            ∇Z = -2 ∑ₘ(S'ₘⱼ·E*₍ⱼ₊ₘ₎ - Sₘⱼ·E*₍ⱼ₊ₘ₎) + (S'₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎ - S₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎) =
               = 2 ∑ₘ ΔSₘⱼ·E*₍ⱼ₊ₘ₎ + ΔS₍ⱼ₋ₘ₎ⱼ·E*₍ⱼ₋ₘ₎

        Donde utilizamos un shifteo circular de los índices.
        """
        ΔSmj = self.Smk_siguiente - self.Smk


        #! Implementación del cálculo según PyPret
        # indices = np.array(np.rint(self.t / self.Δt), dtype=np.int32)
        # self.gradZ = np.zeros((self.M, self.N), dtype=np.complex128)

        # for m in range(self.M):
        #     self.gradZ[m, :] = np.roll(ΔSmj[m, :] * self.campo.conj(), -indices[m])

        # espectro = DFT(self.campo, self.t, self.Δt, self.ω, self.Δω)
        # Amk = IDFT(np.exp(-1j * self.delays * self.ω) * espectro, self.t, self.Δt, self.ω, self.Δω, r_n_conj=self.r_n_conj, s_j_conj=self.s_j_conj)
        
        # self.gradZ = -2 * np.sum(self.gradZ + ΔSmj * Amk.conj(), axis=0)

        #! Mi implementación. Asunción de continuación periódica del campo para el cálculo
        for j in range(self.N):
            self.gradZ[j] = 0

            for τ, m in enumerate(self.bin_delays):

                # Primer término del gradiente ΔSₘⱼ·E*₍ⱼ₋ₘ₎
                if (j - m) < 0:
                    self.gradZ[j] += ΔSmj[τ][j] * self.campo[j - m].conj()
                elif (j - m) >= self.N:
                    self.gradZ[j] += ΔSmj[τ][j] * self.campo[(j - m) - self.N].conj()
                else:
                    self.gradZ[j] += ΔSmj[τ][j] * self.campo[j - m].conj()

                # Segundo término del gradiente ΔSₘ₍ⱼ₊ₘ₎·E*₍ⱼ₊ₘ₎
                if (j + m) < 0:
                    self.gradZ[j] += ΔSmj[τ][j + m] * self.campo[j + m].conj()
                elif (j + m) >= self.N:
                    self.gradZ[j] += ΔSmj[τ][(j + m) - self.N] * self.campo[(j + m) - self.N].conj()
                else:
                    self.gradZ[j] += ΔSmj[τ][j + m] * self.campo[j + m].conj()

            self.gradZ[j] *= -2
    
    def calcula_γ(self):
        """
        Calcula el paso del descenso. En el algoritmo GPA usualmente se realiza
        una búsqueda lineal para encontrarlo, pero una opción más rápida de encontrarlo
        e igual de válida es tomar:
            γ = Z / ∑ⱼ|∇Zⱼ|²

        Debe ser calculada a cada paso de descenso de gradiente que se haga.
        """
        self.γ = self.Z / np.sum(np.abs(self.gradZ)**2)

    def calcula_Smk_siguiente(self):
        """
        Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
        una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
        la siguiente proyección: 

            S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
        """

        # Hemos de tener cuidado con valores nulos a la hora de dividir
        # Si el valor absoluto de un elemento es nulo, tomamos el cociente como la unidad
        absSmn = np.abs(self.Smn)
        f = (absSmn > 0.0)
        """
        NOTA: en el artículo original de Trebino sobre el método GPA, no se realiza la división
        entre el factor μ en este caso, como se realiza en COPRA. Si decidimos hacerla, el pulso obtenido
        presentará una ambigüedad extra en el factor de escala del mismo.
        """
        self.Smn_siguiente[~f] = np.sqrt((self.Tmn_medido[~f] + 0.0j))
        self.Smn_siguiente[f] = self.Smn[f] / absSmn[f] * np.sqrt((self.Tmn_medido[f] + 0.0j))

        for τ in range(self.M):
            self.Smk_siguiente[τ][:] = IDFT(self.Smn_siguiente[τ][:], self.t, self.Δt, self.ω, self.Δω, s_j_conj=self.s_j_conj, r_n_conj=self.r_n_conj)

    def calcula_campo_siguiente(self):
        """
        Actualiza el valor del campo eléctrico del pulso candidato realizando
        un descenso de gradiente:
            Eⱼ' = Eⱼ - γ·∇Zⱼ
        """
        for j in range(self.N):
            self.campo[j] -= self.γ * self.gradZ[j]


class PCGPA(retrieverExpBase):
    """
    Clase para utilizar el método de componentes principales de proyecciones generalizadas (PCGPA) para reconstuir un pulso
    a partir de su traza, partiendo de un pulso candidato. 
    
    El método PCGPA consiste en los siguientes pasos:

        - Paso 1 : proyección sobre Sₘₖ. Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
                        una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
                        la siguiente proyección: S'ₘₖ = μ⁻½ ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}

        - Paso 2 : actualización del campo eléctrico, E, utilizando las propiedades de la estructura algebraica de la traza FROG.
                   Para ello, debemos calcular la forma 'de producto exterior' de Sₘₖ, revirtiendo yhaciendo un shifteo circular
                   de k en la columna k-ésima, de manera que Ŝₘₖ = EₘEₖ. Para actualizar E, se realizan las siguientes dos operaciones:
                       Eₖ -> ∑ₖ (Ŝₘₖ)* · Eₖ
                       Eₖ -> ∑ₖ Eₖ* / (∑ₖ|Eₖ|²)½

                   O se puede actualizar realizando una descomposición en valores singulares (SVD) de Ŝₘₖ. 

        - Paso 3 : calculo de los nuevos parámetros para la siguiente iteración y error en la traza del pulso. Se calculan
                    los nuevos valores del operador señal y la traza del pulso candidato obtenido por descenso de gradiente
                    en el paso 2. Se calcula el error de la traza R, y si se satisface la condición de convergencia o se llega
                    al máximo de iteraciones se para el algoritmo. En caso contrario, se vuelve al paso 1.

    Args:
            t (np.ndarray): array de tiempos equiespaciados Δt
            pulso (np.ndarray[np.complex128]): array con el campo eléctrico del pulso a recuperar
            espectro (np.ndarray[np.complex128], opcional): array con el espectro del pulso a recuperar
    """

    def __init__(self, τ, Δτ, ω, Δω, traza_experimental):
        """
        Inicializa los parámetros necesarios para utilizar el algoritmo de reconstrucción.
        """
        super().__init__(τ, Δτ, ω, Δω, traza_experimental)


    def _inicializa_recuperacion(self, pulso_inicial):
        """
        Inicializa los parámetros necesarios para utilizar el algoritmo de reconstrucción.
        """

        self.campo = np.zeros(self.N, dtype=pulso_inicial.dtype)
        self.campo = pulso_inicial.copy()

        self.Smk = np.zeros((self.M, self.N), dtype=self.campo.dtype)
        self.Smk_siguiente = np.zeros((self.M, self.N), dtype=self.campo.dtype) # Para realizar la primera proyección
        self.calcula_Smk()

        self.Smn = np.zeros((self.M, self.N), dtype=self.campo.dtype)
        self.Smn_siguiente = np.zeros((self.M, self.N), dtype=self.campo.dtype) # Para realizar la primera proyección
        self.calcula_Smn()

        self.Tmn = np.zeros((self.M, self.N), dtype=np.float64)
        self.calcula_Tmn()

        self.μ = None
        self.calcula_μ()

        self.r = None
        self.calcula_residuos()

        self.R = None
        self.calcula_R()

        self.solucion_minimo_error = [np.zeros(self.N, dtype=pulso_inicial.dtype), self.R]


    def recuperacion(self, pulso_inicial, eps, *, max_iter=None):
        """
        Ejecuta el algoritmo de reconstrucción.
        La idea es pasarle un pulso candidato inicial sobre el que inicie la computación del algoritmo.
        Se pasa como argumento la precisión deseada en el error de la traza, y si no, un máximo de iteraciones.

        Dentro de la función se reproduce 'esquematicamente' el algoritmo, desarrollandose internamente en llamadas a funciones.
        El método PCGPA consiste en los siguientes pasos:

        - Paso 1 : proyección sobre Sₘₖ. Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
                        una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
                        la siguiente proyección: S'ₘₖ = μ⁻½ ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}

        - Paso 2 : actualización del campo eléctrico, E, utilizando las propiedades de la estructura algebraica de la traza FROG.
                   Para ello, debemos calcular la forma 'de producto exterior' de Sₘₖ, revirtiendo y haciendo un shifteo circular
                   de k en la columna k-ésima, de manera que Ŝₘₖ = EₘEₖ. Para actualizar E, se realizan las siguientes dos operaciones:
                       Eₖ -> ∑ₖ (Ŝₘₖ)* · Eₖ
                       Eₖ -> ∑ₖ Eₖ* / (∑ₖ|Eₖ|²)½

                   O se puede actualizar realizando una descomposición en valores singulares (SVD) de Ŝₘₖ. 

        - Paso 3 : calculo de los nuevos parámetros para la siguiente iteración y error en la traza del pulso. Se calculan
                    los nuevos valores del operador señal y la traza del pulso candidato obtenido por descenso de gradiente
                    en el paso 2. Se calcula el error de la traza R, y si se satisface la condición de convergencia o se llega
                    al máximo de iteraciones se para el algoritmo. En caso contrario, se vuelve al paso 1.

        Args:
            pulso_inicial (np.ndarray[np.complex128]): campo eléctrico del candidato inicial para el algoritmo de recuperación
            eps (float): precisión deseada en el error de la traza del pulso reconstruido
            max_iter (int, opcional): máximo de iteraciones del algoritmo. Por defecto no hay máximo.
        
        Devuelve:
            campo (np.ndarray[np.complex128]): campo eléctrico de la solución obtenida
            espectro (np.ndarray[np.complex128]): espectro de la solución obtenida
        """

        self._inicializa_recuperacion(pulso_inicial)

        if max_iter is None: max_iter = float('inf')
        niter = 0

        while self.R > eps and niter < max_iter:
            
            self.calcula_Smk_siguiente()
            
            self.calcula_campo_siguiente()

            self.calcula_Smk()
            self.calcula_Smn()
            self.calcula_Tmn()
            self.calcula_μ()
            self.calcula_residuos()
            self.calcula_R()
            
            if self.R < self.solucion_minimo_error[1]:
                self.solucion_minimo_error[0] = self.campo.copy()
                self.solucion_minimo_error[1] = self.R
                
            print(f'n={niter+1}, R={self.R}')
            niter += 1

        print("Error final en la traza: ", self.solucion_minimo_error[1])
        self.campo = self.solucion_minimo_error[0].copy()

        return self.campo, DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j)

    
    def calcula_Smk_siguiente(self):
        """
        Se calcula el nuevo valor del operador señal del pulso candidato, S'ₘₖ, realizando
        una proyección sobre el conjunto de pulsos que satisface que Sₘₖ = ℱ⁻¹{√Tₘₙᵐᵉᵃˢ}, por lo que se realiza
        la siguiente proyección: 

            S'ₘₖ = ℱ⁻¹{Sₘₙ / |Sₘₙ|  · √Tₘₙᵐᵉᵃˢ}
        """

        # Hemos de tener cuidado con valores nulos a la hora de dividir
        # Si el valor absoluto de un elemento es nulo, tomamos el cociente como la unidad
        absSmn = np.abs(self.Smn)
        f = (absSmn > 0.0)
        """
        NOTA: en el artículo original de Trebino sobre el método GPA, no se realiza la división
        entre el factor μ en este caso, como se realiza en COPRA. Si decidimos hacerla, el pulso obtenido
        presentará una ambigüedad extra en el factor de escala del mismo.
        """
        self.Smn_siguiente[~f] = np.sqrt((self.Tmn_medido[~f] + 0.0j))
        self.Smn_siguiente[f] = self.Smn[f] / absSmn[f] * np.sqrt((self.Tmn_medido[f] + 0.0j))

        for τ in range(self.M):
            self.Smk_siguiente[τ][:] = IDFT(self.Smn_siguiente[τ][:], self.t, self.Δt, self.ω, self.Δω, s_j_conj=self.s_j_conj, r_n_conj=self.r_n_conj)

    def calcula_campo_siguiente(self):
        """
        Actualiza el valor del campo eléctrico del pulso candidato.
        """
        for n in range(self.N):
            self.Smk_siguiente[:, n] = np.roll(self.Smk_siguiente[::-1, n], n)
        
        #! Realizar el 'power method' presenta ambigüedades en la amplitud del pulso recuperado
        # self.campo = self.campo.conj() / np.sqrt(np.sum(np.abs(self.campo)**2))
        # self.campo = self.Smk_siguiente.conj() @ self.campo

        # Actualización por Descomposición de Valores Singulares (SVD)
        U, s, V = np.linalg.svd(self.Smk_siguiente)
        self.campo = U[:, 0] * np.sqrt(s[0])