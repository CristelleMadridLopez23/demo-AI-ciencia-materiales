from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 1️⃣ Leer tu dataset con SMILES ===
df = pd.read_csv("/Users/cristellemadrid/Desktop/demo-AI-ciencia-materiales/outputs_moleculas/propiedades_smiles_fairchem.csv")

# Asegúrate de tener una columna 'smiles' o ajusta el nombre
smiles_list = df["smiles"].tolist()

# === 2️⃣ Convertir SMILES → vectores fingerprint ===
fingerprints = []
mol_names = []

for smi, name in zip(smiles_list, df["formula"]):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
        mol_names.append(name)

fingerprints = np.array(fingerprints)

# === 3️⃣ Reducir dimensionalidad con t-SNE ===
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
emb = tsne.fit_transform(fingerprints)

# === 4️⃣ Visualizar ===
plt.figure(figsize=(9,7))
plt.scatter(emb[:,0], emb[:,1], s=60, color="teal", alpha=0.7)

for i, name in enumerate(mol_names):
    plt.text(emb[i,0]+0.5, emb[i,1], name, fontsize=8)

plt.title("Mapa de similitud molecular (t-SNE sobre SMILES)")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.show()
