from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/Users/cristellemadrid/Desktop/demo-AI-ciencia-materiales/esen-sm-direct-all-omol/propiedades_dataset_atomicdata.csv")

X = df[["energia_eV", "fuerza_promedio", "num_atomos"]].values

# 🔽 Ajustar perplexity al tamaño del dataset
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
emb = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(emb[:,0], emb[:,1], c='steelblue')

# Etiquetas de las moléculas
for i, name in enumerate(df["formula"]):
    plt.text(emb[i,0]+0.1, emb[i,1], name, fontsize=9)

plt.grid(False)
plt.show()
