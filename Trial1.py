
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from ase.build import molecule
from ase.visualize import view


# === 1️⃣ Cargar modelo preentrenado ===
mlip = get_predict_unit("esen-sm-direct-all-omol")

# === 2️⃣ Crear el calculador FAIRChem ===
calc = FAIRChemCalculator(mlip)

# === 3️⃣ Definir la molécula (Etanol) ===
atoms = molecule("H2O")
atoms.calc = calc

# === 4️⃣ Obtener energía y fuerzas ===
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energía total: {energy:.4f} eV")
print("Fuerzas (primeros 3 átomos):")
print(forces[:3])

# === 5️⃣ Visualización 3D interactiva ===
atoms.new_array("forces", forces)
view(atoms)

