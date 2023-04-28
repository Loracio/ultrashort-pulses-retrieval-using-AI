"""
Pequeño script que ayuda a buscar el módulo de core dentro de la carpeta del repositorio
"""

import importlib
spec = importlib.util.find_spec("core")
if spec is None or spec.origin == "namespace":
    import sys
    from pathlib import Path
    core_folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(core_folder))