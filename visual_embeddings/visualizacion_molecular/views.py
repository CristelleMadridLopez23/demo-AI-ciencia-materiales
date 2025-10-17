import os
import pandas as pd
from django.shortcuts import render
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.manifold import TSNE
import numpy as np
from .utils import properties_generator

def index(request):
    file_path = properties_generator()
    cache_path = file_path.replace(".csv", "_with_emb.csv")
    # si ya existe cache con embeddings, usarla
    load_path = cache_path if os.path.exists(cache_path) else file_path
    df = pd.read_csv(load_path)
    df.columns = [c.strip() for c in df.columns]

    # si faltan embeddings, calcular y guardar cache
    if "emb_x" not in df.columns or "emb_y" not in df.columns:
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
            df.to_csv(cache_path, index=False)

    # garantizar campo 'nombre' usado en la plantilla
    if "nombre" not in df.columns:
        if "formula" in df.columns:
            df["nombre"] = df["formula"]
        else:
            df["nombre"] = df.index.astype(str)

    columnas_esperadas = [
        "id", "smiles", "formula", "num_atomos", "energia_eV",
        "centro_masa_x", "centro_masa_y", "centro_masa_z",
        "fuerza_promedio", "imagen", "emb_x", "emb_y", "nombre"
    ]
    columnas_disponibles = [c for c in columnas_esperadas if c in df.columns]
    if not columnas_disponibles:
        columnas_disponibles = df.columns.tolist()

    data = df[columnas_disponibles].to_dict(orient="records")
    return render(request, "index.html", {"data": data})