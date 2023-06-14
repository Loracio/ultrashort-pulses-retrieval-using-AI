try:
    __CORE_MODULE_IMPORTED__
except NameError:
    __CORE_MODULE_IMPORTED__= False

if not __CORE_MODULE_IMPORTED__:
    from .autocorrelaciones import trapecio, autocorrelacion_2orden, autocorrelacion_3orden, autocorrelacion_interferometrica, traza
    from .fourier import DFT, IDFT, DFT_clasica, IDFT_clasica, fft, ifft, transformada_bluestein, convolucion, frecuencias_DFT
    from .graficos import plot_real_imag, plot_intensidad, plot_traza
    from .pulso_aleatorio import pulso_aleatorio
    from .recuperacion_experimental import GPA, PCGPA
    from .recuperacion import GPA_retriever, PCGPA_retriever
    from .unidades import convertir
    from .utiles import pulso_gaussiano, transformada_pulso_gaussiano, media, desviacion_estandar, FWHM

__CORE_MODULE_IMPORTED__ = True