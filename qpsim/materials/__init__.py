"""Material descriptors and YAML-backed database.

Ported (Gate 2 task 11):
- substrate.py — Substrate dataclass (name + optional acoustic props)
- database.py — Material dataclass, load_material, list_materials
- data/ — initial YAMLs: Al.yaml, Nb.yaml, TiN.yaml
"""

from qpsim.materials.database import Material, list_materials, load_material
from qpsim.materials.substrate import Substrate

__all__ = ["Material", "Substrate", "list_materials", "load_material"]
