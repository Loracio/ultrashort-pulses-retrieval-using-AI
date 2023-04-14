import numpy as np
import matplotlib.pyplot as plt
from .fourier import *
from .unidades import convertir
from .utiles import media

class retrieverBase():
    """
    Clase base que inicializa los parámetros para un algoritmo de reconstrucción.
    Genera los arrays de frecuencias, delays, calcula la traza experimental; y demás parámetros
    que serán usados una y otra vez en el algoritmo de recuperación.

    La idea de esta clase es ser usada para inicializar diferentes algoritmos de reconstrucción, creando
    clases 'hijo' a partir de esta.
    """

    def __init__(self, t, Δt, pulso, *, espectro=None):
        # Inicialización de parámetros del pulso
        self.campo_medido = pulso
        self.t = t
        self.Δt = Δt
        self.N = self.t.size
        self.frecuencias = frecuencias_DFT(self.N, self.Δt)
        self.ω = convertir(self.frecuencias, 'frecuencia', 'frecuencia angular')
        self.Δω = 2 * np.pi / (self.N * self.Δt) #Relación reciprocidad
        self.M = 2 * self.N - 1
        self.bin_delays = np.linspace(-(self.N - 1), (self.N - 1) , num=self.M, dtype=int)

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

        # Espectro del pulso
        if espectro is None:
            self.espectro_medido = DFT(self.campo_medido, self.t, self.Δt, self.ω, self.Δω, s_j=self.s_j, r_n=self.r_n)
        else:
            self.espectro_medido = espectro

        # Parámetros para el algoritmo de reconstrucción
        self.Tmn_medido = np.zeros((self.M, self.N), dtype=np.float64)
        self.calcula_Tmn_medido()
        self.Tmn_medido_max_cuadrado = self.Tmn_medido.max()**2

    def calcula_Tmn_medido(self):
        Smk_medido = np.zeros((self.M, self.N), dtype=self.campo_medido.dtype)

        for τ in range(self.N):
            # Calcula directamente E(tₖ - τₘ)E(tₖ) para todas las k's usando slicing de arrays (hay que pensarlo un poco esto)
            # Simplemente es hacer cero los puntos donde las señales no coinciden y multiplicarlas allá donde sí coincidan
            Smk_medido[τ][:τ + 1] = self.campo_medido[:τ + 1] * self.campo_medido[self.N - τ - 1:]
            Smk_medido[τ][τ + 1:] = 0
            self.Tmn_medido[τ][:] = np.abs(DFT(Smk_medido[τ], self.t, self.Δt, self.ω, self.Δω, s_j=self.s_j, r_n=self.r_n))**2

        for τ in range(self.N - 1):
            # Calcula directamente E(tₖ - τₘ)E(tₖ) para todas las k's usando slicing de arrays (hay que pensarlo un poco esto)
            # Simplemente es hacer cero los puntos donde las señales no coinciden y multiplicarlas allá donde sí coincidan
            Smk_medido[self.N + τ][:τ + 1] = 0
            Smk_medido[self.N + τ][τ + 1:] = self.campo_medido[τ + 1:] * self.campo_medido[: self.N - τ - 1]
            self.Tmn_medido[self.N + τ][:] = np.abs(DFT(Smk_medido[self.N + τ], self.t, self.Δt, self.ω, self.Δω, s_j=self.s_j, r_n=self.r_n))**2
    

class GPA_retriever(retrieverBase):
    """
    Clase para utilizar el método de proyecciones generalizadas (GPA) para reconstuir un pulso
    a partir de su traza.
    """

    def __init__(self, t, Δt, pulso, *, espectro=None):
        super().__init__(t, Δt, pulso, espectro=espectro)

    def _inicializa_recuperacion(self, pulso_inicial):
        """
        Inicializa los parámetros necesarios para utilizar el algoritmo de reconstrucción.

        Args:
            pulso_inicial (np.ndarray[np.complex128]): campo eléctrico del candidato inicial para el algoritmo de recuperación
        """
            
        self.campo = pulso_inicial

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
        self.gradZ = np.zeros(self.N, dtype=np.complex128)
        self.γ = None


    def recuperacion(self, pulso_inicial, eps, *, max_iter=None):
        """
        Core del algoritmo de reconstrucción.
        La idea es pasarle un pulso candidato inicial sobre el que inicie la computación del algoritmo.
        Se pasa como argumento la precisión deseada en el error de la traza, y si no, un máximo de iteraciones.

        Dentro de la función se reproduce 'esquematicamente' el algoritmo, desarrollandose internamente en llamadas a funciones.

        Args:
            pulso_inicial (np.ndarray[np.complex128]): campo eléctrico del candidato inicial para el algoritmo de recuperación
            eps (float): precisión deseada en el error de la traza del pulso reconstruido
            max_iter (int, opcional): máximo de iteraciones del algoritmo. Por defecto no hay máximo.
        """

        self._inicializa_recuperacion(pulso_inicial)

        if max_iter is None: max_iter = float('inf')
        niter = 0

        while self.R > eps and niter < max_iter:
            
            self.calcula_Smk_siguiente()
            
            self.calcula_Z()
            self.calcula_gradZ()
            self.calcula_γ()
            
            self.calcula_campo_siguiente()

            self.calcula_Smk()
            self.calcula_Smn()
            self.calcula_Tmn()
            self.calcula_μ()
            self.calcula_residuos()
            self.calcula_R()
            
            print(self.R)
            print(niter)
            niter += 1

        print(self.R)

        return self.campo, DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j)

    def calcula_Smk(self):

        for τ in range(self.N):
            # Calcula directamente E(tₖ - τₘ)E(tₖ) para todas las k's usando slicing de arrays (hay que pensarlo un poco esto)
            # Simplemente es hacer cero los puntos donde las señales no coinciden y multiplicarlas allá donde sí coincidan
            self.Smk[τ][:τ + 1] = self.campo[:τ + 1] * self.campo[self.N - τ - 1:]
            self.Smk[τ][τ + 1:] = 0

        for τ in range(self.N - 1):
            # Calcula directamente E(tₖ - τₘ)E(tₖ) para todas las k's usando slicing de arrays (hay que pensarlo un poco esto)
            # Simplemente es hacer cero los puntos donde las señales no coinciden y multiplicarlas allá donde sí coincidan
            self.Smk[self.N + τ][:τ + 1] = 0
            self.Smk[self.N + τ][τ + 1:] = self.campo[τ + 1:] * self.campo[: self.N - τ - 1]


    def calcula_Smn(self):
        for τ in range(self.M):
            self.Smn[τ][:] = DFT(self.Smk[τ][:], self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j)

    def calcula_Tmn(self):
        for τ in range(self.M):
            self.Tmn[τ][:] = np.abs(self.Smn[τ][:])**2

    def calcula_μ(self):
        self.μ = np.sum(self.Tmn_medido * self.Tmn) / np.sum(self.Tmn * self.Tmn)

    def calcula_residuos(self):
        diferencia = np.ravel(self.Tmn_medido - self.μ * self.Tmn)
        self.r = np.sum(diferencia * diferencia)

    def calcula_R(self):
        self.R = np.sqrt(self.r / (self.M * self.N * self.Tmn_medido_max_cuadrado))

    def calcula_Z(self):
        self.Z = np.sum(np.abs(self.Smk_siguiente - self.Smk)**2)
    
    def calcula_gradZ(self):
        ΔSmj = self.Smk_siguiente - self.Smk
        self.gradZ *= 0

        for j in range(self.N):
            for m, τ in enumerate(self.bin_delays):
                if 0 <= (j - m) < self.N:
                    self.gradZ[j] += ΔSmj[τ][j] * self.campo[j - m].conj()

                if 0 <= (j + m) < self.N:
                    self.gradZ[j] += ΔSmj[τ + j][j] * self.campo[j + m].conj()

        self.gradZ *= -2
            
    
    def calcula_γ(self):
        self.γ = self.Z / np.sum(np.abs(self.gradZ)**2)

    def calcula_Smk_siguiente(self):
        # Hemos de tener cuidado con valores nulos a la hora de dividir
        # Si el valor absoluto de un elemento es nulo, tomamos el cociente como la unidad
        absSmn = np.abs(self.Smn)
        f = (absSmn > 0.0)
        self.Smn_siguiente[~f] = np.sqrt(self.Tmn_medido[~f] + 0.0j)
        self.Smn_siguiente[f] = self.Smn[f] / absSmn[f] * np.sqrt(self.Tmn_medido[f] + 0.0j)

        for τ in range(self.M):
            self.Smk_siguiente[τ][:] = np.sqrt(1/self.μ) * IDFT(self.Smn_siguiente[τ][:], self.t, self.Δt, self.ω, self.Δω, s_j_conj=self.s_j_conj, r_n_conj=self.r_n_conj)
        

    def calcula_campo_siguiente(self):
        for j in range(self.N):
            self.campo[j] -= self.γ * self.gradZ[j]

    def plot(self):
        """
        Representa las intensidades temporales y espectrales tanto del pulso original como del pulso solución,
        además de sus correspondientes trazas.
        
        Probablemente esta función necesite retoques para organizarla por 'subfunciones' para que sea menos confusa de leer.
        """
        self.fig, self.ax = plt.subplots(2,2)

        twin_ax00 = self.ax[0][0].twinx()
        twin_ax01 = self.ax[0][1].twinx()
        
        # Plots de la primera fila: intensidad e intensidad espectral del pulso y solución
        self.I_campo_medido = np.abs(self.campo_medido)**2
        self.I_espectral_medido = np.abs(self.espectro_medido)**2

        self.I_campo_solucion = np.abs(self.campo)**2
        self.I_espectral_solucion = np.abs(DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j))**2

        self.fase_campo_medido = np.unwrap(np.angle(self.campo_medido)) 
        self.fase_campo_medido -=  media(self.fase_campo_medido, self.I_campo_medido)

        self.fase_campo_solucion = np.unwrap(np.angle(self.campo)) 
        self.fase_campo_solucion -=  media(self.fase_campo_solucion, self.I_campo_solucion)

        self.fase_espectro_medido = np.unwrap(np.angle(self.espectro_medido)) 
        self.fase_espectro_medido -=  media(self.fase_espectro_medido, self.I_espectral_medido)

        self.fase_espectro_solucion = np.unwrap(np.angle(DFT(self.campo, self.t, self.Δt, self.ω, self.Δω, r_n=self.r_n, s_j=self.s_j))) 
        self.fase_espectro_solucion -=  media(self.fase_espectro_solucion, self.I_espectral_solucion)

        self.ax[0][0].plot(self.t,self. I_campo_medido, color='blue', label='Intensidad campo medido')
        twin_ax00.plot(self.t, self.fase_campo_medido, '-.', color='red')
        self.ax[0][0].plot(np.nan, '-.', label='Fase', color='red')
        self.ax[0][0].plot(self.t,self. I_campo_solucion, color='orange', label='Intensidad campo recuperado')
        twin_ax00.plot(self.t, self.fase_campo_solucion, '-.', color='violet')
        self.ax[0][0].plot(np.nan, '-.', label='Fase campo recuperado', color='violet')
        self.ax[0][0].set_xlabel("Tiempo (ps)")
        self.ax[0][0].set_ylabel("Intensidad (u.a.)")
        twin_ax00.set_ylabel("Fase (rad)")
        self.ax[0][0].set_title("Dominio temporal")
        self.ax[0][0].grid()
        self.ax[0][0].legend()

        self.ax[0][1].plot(self.frecuencias, self.I_espectral_medido, color='blue', label='Intensidad espectral medida')
        twin_ax01.plot(self.frecuencias, self.fase_espectro_medido, '-.', color='red')
        self.ax[0][1].plot(np.nan, '-.', label='Fase', color='red')
        self.ax[0][1].plot(self.frecuencias, self.I_espectral_solucion, color='orange', label='Intensidad espectral medida')
        twin_ax01.plot(self.frecuencias, self.fase_espectro_solucion, '-.', color='violet')
        self.ax[0][1].plot(np.nan, '-.', label='Fase espectral recuperada', color='violet')
        self.ax[0][1].set_xlabel("Frecuencia (1 / ps)")
        self.ax[0][1].set_ylabel("Intensidad (u.a.)")
        twin_ax01.set_ylabel("Fase (rad)")
        self.ax[0][1].set_title("Dominio frecuencial")
        self.ax[0][1].grid()
        self.ax[0][1].legend()

        # Plots de la segunda fila: traza del pulso original y del pulso recuperado
        self.calcula_Tmn()
        self.Tmn_recuperado_normalizado = self.Tmn / np.max(self.Tmn)
        self.Tmn_medido_normalizado = self.Tmn_medido / np.max(self.Tmn_medido)

        self.im0 = self.ax[1][0].pcolormesh(self.frecuencias, self.bin_delays * self.Δt, self.Tmn_medido_normalizado, cmap='inferno')
        self.fig.colorbar(self.im0, ax=self.ax[1][0])
        self.ax[1][0].set_xlabel("Frecuencia (1/ps)")
        self.ax[1][0].set_ylabel("Retraso (ps)")
        self.ax[1][0].set_title("Traza del pulso medido")

        self.im1 = self.ax[1][1].pcolormesh(self.frecuencias, self.bin_delays * self.Δt, self.Tmn, cmap='inferno')
        self.fig.colorbar(self.im1, ax=self.ax[1][1])
        self.ax[1][1].set_xlabel("Frecuencia (1/ps)")
        self.ax[1][1].set_ylabel("Retraso (ps)")
        self.ax[1][1].set_title("Traza del pulso recuperado")

        return self.fig, self.ax
