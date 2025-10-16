import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from ase.build import molecule

# === 1️⃣ Modelo preentrenado ===
mlip = get_predict_unit("esen-sm-direct-all-omol")
calc = FAIRChemCalculator(mlip)

# === 2️⃣ Diccionario de moléculas paramagnéticas o radicales ===
MOLECULAS = {
    "O2": "Paramagnética",
    "NO": "Paramagnética",
    "CN": "Paramagnética",
    "OH": "Radical",
    "NH": "Radical",
    "CH": "Radical",
    "CH2_s3B1d": "Birradical",
    "ClO": "Radical",
    "SO": "Radical",
    "CH3": "Radical",
    "O3": "Paramagnética leve",
    "HCO": "Radical",
    "SiH2_s3B1d": "Birradical"
}

# === 3️⃣ Recorrer y guardar propiedades ===
data = []

for nombre, tipo in MOLECULAS.items():
    try:
        atoms = molecule(nombre)
        atoms.calc = calc

        formula = atoms.get_chemical_formula()
        n_atoms = len(atoms)
        energy = atoms.get_potential_energy()
        center = atoms.get_center_of_mass()
        forces = atoms.get_forces()

        data.append({
            "nombre": nombre,
            "tipo_magnetismo": tipo,
            "formula": formula,
            "num_atomos": n_atoms,
            "energia_eV": energy,
            "centro_masa_x": center[0],
            "centro_masa_y": center[1],
            "centro_masa_z": center[2],
            "fuerza_promedio": float(forces.mean())
        })

        print(f"✅ {nombre} procesada correctamente ({formula})")

    except Exception as e:
        print(f"⚠️ Error con {nombre}: {e}")

# === 4️⃣ Convertir a DataFrame ===
df = pd.DataFrame(data)
print("\n=== Vista previa de resultados ===")
print(df)

# Guardar
df.to_csv("moleculas_paramagneticas.csv", index=False)
