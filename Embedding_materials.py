from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/Users/cristellemadrid/Desktop/demo-AI-ciencia-materiales/moleculas_paramagneticas.csv")

X = df[["energia_eV", "fuerza_promedio", "num_atomos"]].values

# 🔽 Ajustar perplexity al tamaño del dataset
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
emb = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(emb[:,0], emb[:,1], c='steelblue')

# Etiquetas de las moléculas
for i, name in enumerate(df["nombre"]):
    plt.text(emb[i,0]+0.1, emb[i,1], name, fontsize=9)

plt.title("Embeddings de moléculas paramagnéticas según propiedades FAIRChem")
plt.xlabel("Componente 1 (t-SNE)")
plt.ylabel("Componente 2 (t-SNE)")
plt.grid(True)
plt.show()
