"""Material descriptors and YAML-backed database.

Modules:
- substrate.py — Substrate dataclass (name + optional acoustic props)
- database.py — Material dataclass, load_material, list_materials
- data/ — material YAMLs: Al, Nb, NbN, Sn, Ta, TiN
"""

from qpsim.materials.database import Material, list_materials, load_material
from qpsim.materials.substrate import Substrate

__all__ = ["Material", "Substrate", "list_materials", "load_material"]
