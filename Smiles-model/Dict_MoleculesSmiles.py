import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
from tqdm import tqdm
from ase.visualize import view
from ase.io import read
from io import StringIO
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# === 🧠 1️⃣ Cargar modelo preentrenado ===
mlip = get_predict_unit("esen-sm-direct-all-omol")
calc = FAIRChemCalculator(mlip)

# === 🧫 2️⃣ Lista de SMILES ===
smiles_list = [
    # === Hierro (Fe) ===
    "[Fe]",  # átomo libre
    "[Fe+2]", "[Fe+3]",
    "C1=CC=[C-]C=C1.[Fe+2]",  # ferroceno aniónico
    "C1=CC=CC=C1[Fe]C1=CC=CC=C1",  # ferroceno
    "OC(=O)[Fe]OC(=O)O",  # oxalato férrico
    "[Fe(H2O)6]3+",  # complejo acuoso
    "[Fe(CN)6]4-",  # ferrocianuro
    "[Fe(CO)5]",  # pentacarbonil hierro
    "CC(=O)O[Fe]OOC(C)=O",  # acetato férrico

    # === Níquel (Ni) ===
    "[Ni]", "[Ni+2]",
    "[Ni(CO)4]",  # tetracarbonil níquel
    "[Ni(H2O)6]2+",  # hexaaquaníquel
    "[NiCl4]2-",  # tetrachloroníquelato
    "O=[Ni]=O",  # óxido de níquel simplificado
    "C1=CC=CC=C1[Ni]C1=CC=CC=C1",  # bis(benceno)níquel
    "CC(=O)O[Ni]OOC(C)=O",  # acetato de níquel
    "[Ni(NH3)6]2+",  # hexamminenickel(II)

    # === Cobalto (Co) ===
    "[Co]", "[Co+2]", "[Co+3]",
    "[Co(NH3)6]3+",  # hexaammincobalto(III)
    "[Co(CO)4]-",  # tetracarbonilcobalto aniónico
    "[Co(CN)6]3-",  # hexacianocobaltato
    "[CoCl4]2-",  # tetraclorocobaltato
    "O=[Co]=O",  # óxido de cobalto (simplificado)
    "CC(=O)O[Co]OOC(C)=O",  # acetato de cobalto
    "[Co(H2O)6]2+",  # hexaaquacobalto(II)

    # === Manganeso (Mn) ===
    "[Mn]", "[Mn+2]", "[Mn+3]",
    "[Mn(CO)5]−",  # pentacarbonilmanganato
    "[Mn(H2O)6]2+",
    "[MnCl4]2-",  # tetrachloromanganato
    "O=[Mn]=O",  # óxido de manganeso
    "CC(=O)O[Mn]OOC(C)=O",  # acetato de manganeso
    "[Mn(CN)6]3-",  # hexacianomanganato

    # === Cromo (Cr) ===
    "[Cr]", "[Cr+2]", "[Cr+3]",
    "[Cr(H2O)6]3+",
    "[Cr(CO)6]",  # hexacarbonilcromo
    "[Cr(CN)6]3-",  # hexacianocromato
    "[CrCl4]2-",  # tetraclorocromato
    "CC(=O)O[Cr]OOC(C)=O"  # acetato de cromo
]


# === 📂 3️⃣ Crear carpeta de salida ===
output_dir = "outputs_moleculas"
os.makedirs(output_dir, exist_ok=True)

data = []

# === ⚙️ 4️⃣ Procesar cada SMILES ===
for i, smi in enumerate(tqdm(smiles_list)):
    try:
        # --- Crear molécula RDKit con hidrógenos y geometría optimizada ---
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        # --- Convertir RDKit Mol a MOLBlock (contiene enlaces) ---
        mol_block = Chem.MolToMolBlock(mol)

        # --- Cargar en ASE conservando bonds ---
        atoms = read(StringIO(mol_block), format='mol')
        atoms.calc = calc

        # --- Calcular propiedades con FairChem ---
        formula = atoms.get_chemical_formula()
        n_atoms = len(atoms)
        energy = atoms.get_potential_energy()
        center = atoms.get_center_of_mass()
        forces = atoms.get_forces()

        # --- Guardar imagen 2D (opcional) ---
        image_path = os.path.join(output_dir, f"mol_{i}_{formula}.png")
        Draw.MolToFile(mol, image_path, size=(400, 400))

        # --- Visualización 3D interactiva (con enlaces visibles) ---
        print(f"\n🧬 Visualizando {formula} ({smi})")
        view(atoms)

        # --- Guardar resultados ---
        data.append({
            "id": i,
            "smiles": smi,
            "formula": formula,
            "num_atomos": n_atoms,
            "energia_eV": energy,
            "centro_masa_x": center[0],
            "centro_masa_y": center[1],
            "centro_masa_z": center[2],
            "fuerza_promedio": float(forces.mean()),
            "imagen": image_path
        })

        print(f"✅ Procesado: {formula} | Imagen: {image_path}")

    except Exception as e:
        print(f"⚠️ Error con {smi}: {e}")

# === 📊 5️⃣ Exportar resultados ===
df = pd.DataFrame(data)
print("\n=== Vista previa de resultados ===")
print(df.head())

output_csv = os.path.join(output_dir, "propiedades_smiles_fairchem.csv")
df.to_csv(output_csv, index=False)
print(f"💾 Archivo guardado en: {output_csv}")
