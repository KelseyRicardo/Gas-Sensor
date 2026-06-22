from dataset.pre_processamento import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X)
labels = kmeans.predict(X)

clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    plt.scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
plt.xlabel('feature_9')
plt.ylabel('feature_100')
plt.title('K-Means (K=6)')
plt.legend()
plt.show()


kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_ss)
labels = kmeans.predict(X_ss)

clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    plt.scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
plt.xlabel('feature_9')
plt.ylabel('feature_100')
plt.title('Standard Scaler')
plt.legend()
plt.show()

ct1 = pd.DataFrame({'label':labels,
                    'gases': y['gas'].values})
ct1 = pd.crosstab(ct1['gases'],ct1['label'], margins=True)
print(ct1)


kmeans = KMeans(n_clusters=6, random_state=42, n_init=10, max_iter=300)
kmeans.fit(X_mm)
labels = kmeans.predict(X_mm)

clusters = np.unique(labels)
for c in clusters:
    idx = labels == c
    plt.scatter(
        all_batch.loc[idx, 'feature_9'],all_batch.loc[idx, 'feature_100'],
        label=f'Cluster {c}',alpha=0.7
    )
plt.xlabel('feature_9')
plt.ylabel('feature_100')
plt.title('MinMax Scaler')
plt.legend()
plt.show()

ct2 = pd.DataFrame({'label':labels,
                    'gases': y['gas'].values})
ct2 = pd.crosstab(ct2['gases'],ct2['label'], margins=True)
print(ct2)

sns.heatmap(ct1,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Cluster K-Means')
plt.ylabel('Gás Real')
plt.title('Standard Scaler')
plt.show()

sns.heatmap(ct2,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Cluster K-Means')
plt.ylabel('Gás Real')
plt.title('MinMax Scaler')
plt.show()