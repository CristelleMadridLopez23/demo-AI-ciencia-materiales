import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from tqdm import tqdm
from ase import Atoms
from ase.data import chemical_symbols

# === FAIRChem imports ===
from fairchem.core.common.registry import registry
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.datasets import AseDBDataset

# 🚀 1️⃣ Forzar registro de modelos
from fairchem.core.models import *   # <- esta línea es clave
print("📦 Módulo 'esen' cargado en el registro.")

# === 2️⃣ Cargar checkpoint antiguo (.pt) ===
checkpoint_path = "/Users/cristellemadrid/Tesis_PlayZone/trials/esen_sm_direct_all.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
print(f"✅ Checkpoint cargado: {type(checkpoint)}")

# === 3️⃣ Extraer los pesos ===
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# === 4️⃣ Obtener la clase del modelo ===
print("Reconstruyendo arquitectura ESEN...")
model_class = registry.get_model_class("esen")
model = model_class()
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"⚙️ Pesos cargados con {len(missing)} faltantes y {len(unexpected)} no coincidentes.")
model.eval()

# === 5️⃣ Envolver en FAIRChemCalculator ===
calc = FAIRChemCalculator(model)
print("✅ Modelo ESEN reconstruido y envuelto como calculador ASE")

# === 6️⃣ Cargar dataset ===
dataset_path = "/Users/cristellemadrid/Tesis_PlayZone/trials/test"
dataset = AseDBDataset({"src": dataset_path})
print(f"✅ Dataset cargado con {len(dataset)} estructuras")

# === 7️⃣ Procesar primeras 20 estructuras ===
data = []

for i in tqdm(range(0, 20)):
    try:
        atomic_data = dataset[i]
        atomic_numbers = atomic_data.atomic_numbers.cpu().numpy()
        positions = atomic_data.pos.cpu().numpy()

        symbols = [chemical_symbols[int(z)] for z in atomic_numbers]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc

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

# === 8️⃣ Guardar resultados ===
df = pd.DataFrame(data)
print("\n=== Vista previa de resultados ===")
print(df.head())

df.to_csv("propiedades_dataset_rebuild_esen_final.csv", index=False)
print("💾 Archivo guardado: propiedades_dataset_rebuild_esen_final.csv")
