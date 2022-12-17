try:
    __CORE_MODULE_IMPORTED__
except NameError:
    __CORE_MODULE_IMPORTED__= False

if not __CORE_MODULE_IMPORTED__:
    from .fourier import *
    from .funciones_plots import *
    from .utils import *
__CORE_MODULE_IMPORTED__ = True