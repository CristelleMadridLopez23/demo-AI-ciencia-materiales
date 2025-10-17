import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ase.visualize import view
from ase.io import read
from io import StringIO
from fairchem.core.calculate.pretrained_mlip import get_predict_unit
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, DataStructs
from sklearn.manifold import TSNE

def properties_generator():
    # directorio y rutas de salida
    output_dir = "/Users/cristellemadrid/Desktop/demo-AI-ciencia-materiales/visual_embeddings/visualizacion_molecular/data"
    base_csv = os.path.join(output_dir, "propiedades_smiles_fairchem.csv")
    cache_csv = base_csv.replace(".csv", "_with_emb.csv")

    # si ya existe cache con embeddings, devolverla inmediatamente
    if os.path.exists(cache_csv):
        return os.path.abspath(cache_csv)

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


    data = []
    # === ⚙️ 4️⃣ Procesar cada SMILES ===
    for i, smi in enumerate(tqdm(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)

            mol_block = Chem.MolToMolBlock(mol)
            atoms = read(StringIO(mol_block), format='mol')
            atoms.calc = calc

            formula = atoms.get_chemical_formula()
            n_atoms = len(atoms)
            energy = atoms.get_potential_energy()
            center = atoms.get_center_of_mass()
            forces = atoms.get_forces()

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
            })

        except Exception as e:
            print(f"⚠️ Error con {smi}: {e}")

    # === 📊 5️⃣ Crear DataFrame y guardar base CSV (sin embeddings todavía) ===
    df = pd.DataFrame(data)
    print(f"💾 Archivo base sin embdx y emby {df}")

    # === 🔢 6️⃣ Calcular embeddings (fingerprints -> t-SNE) ===
    smiles = df.get("smiles", pd.Series([""] * len(df))).fillna("").tolist()
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        arr = np.zeros((256,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_idx.append(i)

    if len(fps) > 1:
        fps_arr = np.array(fps)
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(fps_arr)//4)), random_state=42, init='pca')
        emb = tsne.fit_transform(fps_arr)
        emb_x = [float("nan")] * len(df)
        emb_y = [float("nan")] * len(df)
        for j, idx in enumerate(valid_idx):
            emb_x[idx] = float(emb[j, 0])
            emb_y[idx] = float(emb[j, 1])
        df["emb_x"] = emb_x
        df["emb_y"] = emb_y
    else:
        df["emb_x"] = [float("nan")] * len(df)
        df["emb_y"] = [float("nan")] * len(df)

    # asegurar columna 'nombre' para la plantilla
    if "nombre" not in df.columns:
        if "formula" in df.columns:
            df["nombre"] = df["formula"]
        else:
            df["nombre"] = df.index.astype(str)

    # guardar cache con embeddings y devolver ruta absoluta
    df.to_csv(cache_csv, index=False)
    print(f"💾 Archivo con embeddings guardado en: {cache_csv}")
    
    return os.path.abspath(cache_csv)