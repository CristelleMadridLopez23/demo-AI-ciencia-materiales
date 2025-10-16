import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from tqdm import tqdm
from ase import Atoms
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.datasets import AseDBDataset

# === 1️⃣ Cargar modelo preentrenado ===
mlip = get_predict_unit("esen-sm-direct-all-omol")
calc = FAIRChemCalculator(mlip)

# === 2️⃣ Cargar dataset ===
dataset_path = "/Users/cristellemadrid/Tesis_PlayZone/trials/test"
dataset = AseDBDataset({"src": dataset_path})
print(f"✅ Dataset cargado con {len(dataset)} estructuras")

# === 3️⃣ Procesar muestras del dataset ===
data = []

for i in tqdm(range(0, 20)):  # muestreo cada 50k
    try:
        atomic_data = dataset[i]  # <- objeto AtomicData
        # Convertir a ASE
        atomic_numbers = atomic_data.atomic_numbers.cpu().numpy()
        positions = atomic_data.pos.cpu().numpy()
        
        # Convertir símbolos
        from ase.data import chemical_symbols
        symbols = [chemical_symbols[int(z)] for z in atomic_numbers]

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc

        # === Calcular propiedades ===
        formula = atoms.get_chemical_formula()
        n_atoms = len(atoms)
        energy = atoms.get_potential_energy()
        center = atoms.get_center_of_mass()
        forces = atoms.get_forces()

        data.append({
            "id": i,
            "formula": formula,
            "num_atomos": n_atoms,
            "energia_eV": energy,
            "centro_masa_x": center[0],
            "centro_masa_y": center[1],
            "centro_masa_z": center[2],
            "fuerza_promedio": float(forces.mean())
        })

        print(f"✅ Estructura {i} procesada correctamente ({formula})")

    except Exception as e:
        print(f"⚠️ Error en muestra {i}: {e}")

# === 4️⃣ Convertir a DataFrame y guardar ===
df = pd.DataFrame(data)
print("\n=== Vista previa de resultados ===")
print(df.head())

df.to_csv("propiedades_dataset_atomicdata.csv", index=False)
print("💾 Archivo guardado: propiedades_dataset_atomicdata.csv")
