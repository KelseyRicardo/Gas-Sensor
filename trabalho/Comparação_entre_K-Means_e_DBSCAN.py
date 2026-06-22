from dataset.pre_processamento import *
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=10)
nbrs = neighbors.fit(X_ss)
distances, indices = nbrs.kneighbors(X_ss)
distances = np.sort(distances[:,9])

plt.plot(distances)
plt.xlabel('Pontos ordenados')
plt.ylabel('Distância ao 10º vizinho')
plt.axhline(y=3, color="#E74C3C", linestyle="--", linewidth=1.5,
           label=f"eps escolhido = {3}")
plt.show()

dbscan = DBSCAN(eps=3, min_samples=10)
labels = dbscan.fit_predict(X_ss)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print("Clusters:", n_clusters)
print("Ruído:", n_noise)

plt.scatter(
    all_batch['feature_9'], all_batch['feature_100'],
    c=labels, alpha=0.7
)
plt.show()

ct = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct = pd.crosstab(ct['gas'],ct['cluster'],margins=True)

print(ct)