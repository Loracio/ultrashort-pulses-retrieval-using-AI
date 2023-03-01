import numpy as np
import scipy.optimize as opt
from .utiles import media, desviacion_estandar
from .fourier import frecuencias_DFT, DFT, IDFT
from .unidades import convertir


def pulso_aleatorio(t, Δt, N,  TBP, valor_extremos=None, max_iteraciones=10):
    """
    Función que genera un pulso aleatorio especificado por su TBP (producto tiempo-ancho de banda).
    Esta función nos permite crear varios pulsos con el mismo parámetro en común pero de diferente forma,
    ya que dos pulsos con mismo valor de TBP pueden tener distintos valores de anchura temporal y anchura espectral.

    El pulso se crea sobre el array de tiempos especificado, con unos valores extremos en el dominio
    de frecuencias especificado por el argumento 'valor_extremos', que por defecto se evalua como una
    estimación del error numérico que se tiene al realizar los saltos entre dominios con la transformada de Fourier.

    Para ello, se crea en primer lugar una gaussiana en el espacio de frecuencias que cumpla que su
    anchura espectral sea la que haga que en los extremos se llegue al valor especificado ('valor_extremos').

    Generamos N numeros complejos aleatorios, a los que aplicamos este filtro frecuencial (la gaussiana), que
    tiene el TBP especificado. El resultado de esto no nos dará un pulso con el TBP especificado (es decir, si
    realizamos la IDFT y obtenemos la desviación estándar en el dominio temporal y en el frecuencial y las
    multiplicamos, no obtenemos el TBP especificado). 

    Para obtenerlo, deberemos realizar una búsqueda de cierto factor por el que tendremos que multiplicar el TBP
    especificado al aplicar el filtro gaussiano en el espacio temporal para que sí se satisfazca que el producto de las 
    desviaciones estándar en los dos dominios sea el TBP especificado.

    Esta búsqueda se realiza probando con una serie de valores del factor (desde 0.5 a 1.5), dando como resultado el
    siguiente procedimiento:

        - Partimos de los números complejos aleatorios multiplicados por el filtro gaussiano
        en el espacio de frecuencias, calculamos su IDFT y aplicamos el filtro temporal al pulso
        obtenido. Este filtro temporal será una gaussiana con una anchura que satisface el tener
        el TBP dado, multiplicada por el factor adecuado tentativo para que al aplicar el filtro
        a los números aleatorios se satisfazca la condición del TBP.

        - Una vez hemos aplicado el filtro con el valor tentativo, calculamos la DFT obteniendo el
        espectro de este pulso, y calculamos la desviación estándar en los dos dominios, obteniendo
        el TBP como el producto de estas desviaciones. Comprobamos si este TBP tentativo es igual
        al indicado, viendo si la diferencia de ellos es cero. Si no lo es, repetimos el proceso:

        - Partimos de los números complejos aleatorios multiplicados por el filtro en frecuencias,
        hallamos sus valores en el dominio temporal, aplicamos el filtro temporal corregido con el
        nuevo valor tentativo. Después nuevamente su espectro en frecuencias, y calculamos la desviación
        estándar en ambos dominios para ver si satisfacen el TBP especificado.

    Para hallar este factor utilizamos la función brentq de scipy.optimize, que aplica el algoritmo de Brent
    para encontrar los ceros de una función.

    Finalmente, aplicamos el filtro temporal con el factor optimizado que nos da el TBP sobre el 
    pulso del que partíamos (números complejos aleatorios multiplicados por el filtro gaussiano en
    el dominio de frecuencias que tiene el TBP especificado).

    Así, obtenemos el valor del pulso y su espectro tal que tienen el TBP especificado.

    Notar que el mínimo valor de TBP posible es 1/2 (caso limitado por la transformada) y en ese caso
    el espectro resultado es simplemente los números complejos aleatorios multiplicados por el filtro 
    gaussiano original.


    Args:
        t (np.array): vector de tiempos donde crear el pulso
        Δt (float): espaciado del vector de tiempos
        N (int): número de elementos del vector de tiempos
        TBP (float): producto tiempo-ancho de banda
        valor_extremos (float, optional): valor del pulso en los extremos en el dominio frecuencial.
                                          Por defecto es el error estimado al realizar el salto entre 
                                          dominios con la transformada de Fourier.
        max_iteraciones (int, optional): Número máximo de iteraciones para encontrar un pulso tentativo. Por defecto es 10.

    Devuelve:
        (np.array), (np.array): vector con el campo complejo del pulso, vector con los coeficientes de su DFT
    """
    # Creamos array de frecuencias para saltar entre dominios
    ω = convertir(frecuencias_DFT(N, Δt), 'f', 'ω')
    Δω = 2 * np.pi / (N  * Δt)
    # Para centrar el pulso resultante
    t_0 = 0.5 * (t[0] + t[-1])
    ω_0 = 0.5 * (ω[0] + ω[-1])

    """
    Si no se especifica el valor que queramos que tenga el 
    pulso en sus extremos, tomamos el valor del error que se
    tiene al saltar entre dominios con la fft, que viene dado
    por el producto del número de puntos y la precisión en los
    números de tipo double.
    """

    if valor_extremos is None: # Si no se espeficica valor, usamos el por defecto
        valor_extremos = N * np.finfo(np.double).eps

    log_ve = np.log(valor_extremos)

    """
    Calculamos el valor de anchura espectral que tendrá una Gaussiana
    que decaiga en los extremos al valor fijado. Esto viene dado por:
    Δf = sqrt(-0.125 * (ω_fin - ω_ini)² / ln(valor_extremos))
    """

    anchura_espectral = np.sqrt(-0.125 * (ω[0] - ω[-1])**2 / log_ve)
    # Obtenemos la anchura temporal por la relación de incertidumbre
    anchura_temporal = 2.0 * TBP / anchura_espectral #! No veo el por qué de esto

    # Dado esto, como máximo, la anchura temporal podrá ser:
    anchura_temporal_maxima = np.sqrt(-0.125 * (t[0] - t[-1])**2 / log_ve)

    if anchura_temporal > anchura_temporal_maxima:
        raise ValueError("No se puede alcanzar el producto ancho de banda - tiempo especificado\nReduce el valor en los extremos o aumenta el número de muestras.")

    # Creamos una Gaussiana con la anchura espectral dada por el TBP y el valor en los extremos
    filtro_espectral = np.exp(- (ω-ω_0)**2 / (2 * anchura_espectral * anchura_espectral))

    """
    El algoritmo funciona filtrando iterativamente en el dominio frecuencial
    y temporal. Sin embargo, las funciones escogidas para el filtro no dan el
    TBP correcto. Para obtener el resultado exacto, necesitamos escalar el ancho
    de banda del filtro temporal por un factor y realizar una búsqueda que nos de
    que la diferencia con el valor especificado es mínimo.

    Para eso se utiliza el algoritmo de scipy.optimize llamado brentq, que utiliza
    el método de Brent para hallar las raíces de una función.

    Por lo tanto, creamos una función que nos devuelva el TBP calculado 'a mano'
    de los valores del array y lo restamos al valor dado como argumento. Cuando
    la diferencia sea nula, habremos encontrado un pulso que tenga el TBP especificado.
    """


    if TBP == 0.5:
        # Caso pulso limitado por la transformada
        #? Por qué es 0.5 y no 0.441? En pypret dice que es porque con esta definición un pulso Gaussiano limitado tiene un TBP de 0.5
        fase = np.exp(2j * np.pi * np.random.rand(N))
        espectro = filtro_espectral * fase
        return  IDFT(espectro, t, Δt, ω, Δω), espectro 


    # Amplitud y fase aleatorias multiplicadas por una gaussiana con la anchura espectral determinada por el TBP y valores en los extremos
    espectro_tentativo = np.random.rand(N) * np.exp(2j * np.pi * np.random.rand(N)) * filtro_espectral
    # Guardamos el valor de la transformada inversa para aplicarle el filtro corregido con el factor
    pulso_tentativo = IDFT(espectro_tentativo, t, Δt, ω, Δω)
    # Array que contendrá el pulso temporal con el filtro aplicado con el factor a optimizar
    pulso = np.zeros(N, dtype=pulso_tentativo.dtype)
    # Array que contendrá la transformada del pulso temporal para calcular el TBP y ver si es el especificado
    espectro = np.zeros(N, dtype=espectro_tentativo.dtype)


    # Valor tentativo para el rango donde estará el valor óptimo del factor
    factor_min, factor_max = 0.5, 1.5

    def crea_pulso(factor):
        """
        Aplica el filtro temporal al pulso tentativo y calcula su espectro, guardándolos en los
        arrays 'pulso' y 'espectro' para poder calcular el valor del TBP en la función de optimización
        """
        filtro_temporal = np.exp(- (t-t_0)*(t-t_0) / (2 * anchura_temporal * anchura_temporal * factor * factor))

        pulso[:] = pulso_tentativo * filtro_temporal
        espectro[:] = DFT(pulso, t, Δt, ω, Δω)

    def objetivo(factor):
        """
        Función objetivo, deberemos encontrar su raíz con el método de Brent
        """
        crea_pulso(factor)
        TBP_tentativo = desviacion_estandar(t, np.abs(pulso)**2) * desviacion_estandar(ω, np.abs(espectro)**2)

        return TBP - TBP_tentativo

    iteracion = 0

    while np.sign(objetivo(factor_min)) == np.sign(objetivo(factor_max)):
        espectro_tentativo = np.random.rand(N) * np.exp(2j * np.pi * np.random.rand(N)) * filtro_espectral
        pulso_tentativo = IDFT(espectro_tentativo, t, Δt, ω, Δω)

        iteracion +=1 

        if iteracion == max_iteraciones:
            raise ValueError("Máximo de iteraciones alcanzado: no se ha podido crear un pulso para estos parámetros")

    # Calculo de la optimizacion del factor
    factor = opt.brentq(objetivo, factor_min, factor_max)

    # Calculamos el pulso y su espectro con ese valor
    crea_pulso(factor)

    return pulso, espectro