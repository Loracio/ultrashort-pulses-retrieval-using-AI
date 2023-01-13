"""
A la hora de trabajar con pulsos electromagnéticos, hay varias magnitudes
relacionadas con tiempo/frecuencia que pueden ser empleadas.

Este módulo nos otorga funciones para convertir entre las siguientes unidades:

    - λ: longitud de onda [m]
    - ω: frecuencia angular [rad/s]
    - f: frecuencia [1/s]
    - k: vector de ondas [rad/m]

Para simplificar el uso del módulo, en vez de recordar los nombres de las 12 funciones
para pasar de una unidad a otra, utilizamos una única función a la que introducimos el
array a convertir y dos cadenas de carácteres que indican de qué unidad a qué unidad 
queremos convertir. Un ejemplo de esto sería:

>>> convertir(x, 'λ', 'ω')

La implementación de esta función se basa en definir cada una de las 12 funciones de
conversión, y después guardar en un diccionario ('selector') cuándo debe ser usada cada
función dependiendo de la conversión deseada.
"""
import numpy as np
import scipy.constants as constants

def longitudOnda_frecuenciaAngular(λ):
    return 2 * np.pi * constants.c / λ

def longitudOnda_frecuencia(λ):
    return constans.c / λ

def longitudOnda_vectorOndas(λ):
    return 2 * np.pi / λ

def frecuenciaAngular_longitudOnda(ω):
    return 2 * np.pi * constants.c / ω

def frecuenciaAngular_frecuencia(ω):
    return ω / (2 * np.pi)

def frecuenciaAngular_vectorOndas(ω):
    return ω / constants.c

def frecuencia_longitudOnda(f):
    return constants.c / f

def frecuencia_frecuenciaAngular(f):
    return 2 * np.pi * f

def frecuencia_vectorOndas(f):
    return 2*np.pi * f / constants.c

def vectorOndas_longitudOnda(k):
    return 2 * np.pi / k

def vectorOndas_frecuenciaAngular(k):
    return k * constants.c

def vectorOndas_frecuencia(k):
    return k * constants.c / (2 * np.pi)

alias = {
    'λ': 'λ',
    'lambda': 'λ',
    'longitud onda': 'λ',
    'ω': 'ω', 
    'omega' : 'ω',
    'frecuencia angular' : 'ω',
    'f': 'f',
    'frecuencia': 'f',
    'k': 'k',
    'vector ondas': 'k'
}

selector = {
    'λ' : {
        'λ': lambda x: copy(x),
        'ω': longitudOnda_frecuenciaAngular,
        'f': longitudOnda_frecuencia,
        'k': longitudOnda_vectorOndas
    },

    'ω': {
        'λ': frecuenciaAngular_longitudOnda,
        'ω': lambda x: copy(x),
        'f': frecuenciaAngular_frecuencia,
        'k': frecuenciaAngular_vectorOndas
    },

    'f': {
        'λ': frecuencia_longitudOnda,
        'ω': frecuencia_frecuenciaAngular,
        'f': lambda x: copy(x),
        'k': frecuencia_vectorOndas
    },

    'k': {
        'λ': vectorOndas_longitudOnda,
        'ω': vectorOndas_frecuenciaAngular,
        'f': vectorOndas_frecuencia,
        'k': lambda x: copy(x)
    },
}

def convertir(x: np.ndarray, unidades_entrada: str, unidades_salida: str):
    """
    Convierte un array dado en ciertas unidades (unidades_entrada) a otras
    unidades (unidades_salida). Pueden utilizarse los siguientes 'alias' para
    las distintas unidades:

        'λ', 'lambda', 'longitud onda' para la longitud de onda
        'ω', 'omega', 'frecuencia angular' para la frecuencia angular
        'f', 'frecuencia' para la frecuencia
        'k', 'vector ondas' para el vector de ondas

    Args:
        x (np.ndarray): array con los valores numéricos en la unidad de entrada
        unidades_entrada (string): unidad de los datos de entrada
        unidades_salida (string): unidad a la que se quieren convertir los datos

    Devuelve:
        (np.ndarray): array con los valores numéricos en la unidad a la que se quería convertir.
                      Siempre será una copia del array de entrada y nunca lo sobreescribirá.
    """

    return selector[alias[unidades_entrada]][alias[unidades_salida]](x)
