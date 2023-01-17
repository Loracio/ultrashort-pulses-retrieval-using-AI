try:
    __CORE_MODULE_IMPORTED__
except NameError:
    __CORE_MODULE_IMPORTED__= False

if not __CORE_MODULE_IMPORTED__:
    from .fourier import *
    from .graficos import *
    from .unidades import *
    from .utiles import *
__CORE_MODULE_IMPORTED__ = True