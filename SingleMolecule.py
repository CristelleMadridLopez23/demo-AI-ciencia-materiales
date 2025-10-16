
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
atoms = molecule("SiH4")  # silano, precursor semiconductor
atoms.calc = calc

atoms= {
    
}

# === 4️⃣ Informacion sobre la molecula  ===
# Datos básicos
print("Fórmula:", atoms.get_chemical_formula())
print("Número de átomos:", atoms.get_number_of_atoms())

# Propiedades globales (requiere calculador)
print("Energía total:", atoms.get_potential_energy(), "eV")
print("Centro de masa:", atoms.get_center_of_mass())

forces = atoms.get_forces()
print("Fuerzas (primeros 3 átomos):")
print(forces[:3])

# === 5️⃣ Visualización 3D interactiva ===
atoms.new_array("forces", forces)
view(atoms)

