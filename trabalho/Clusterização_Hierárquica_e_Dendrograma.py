from matplotlib import pyplot as plt
from dataset.pre_processamento import *
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

merging = linkage(X_ss, method='complete')
dendrogram(merging)
plt.title('Dendrograma: complete')
#plt.show()
labels = fcluster(merging,t=6,criterion='maxclust')
ct = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct = pd.crosstab(ct['gas'],ct['cluster'],margins=True)
print(ct)

merging = linkage(X_ss, method='average')
dendrogram(merging)
plt.title('Dendrograma: average')
#plt.show()
labels = fcluster(merging,t=6,criterion='maxclust')
ct = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct = pd.crosstab(ct['gas'],ct['cluster'],margins=True)
print(ct)

merging = linkage(X_ss, method='single')
dendrogram(merging)
plt.title('Dendrograma: single')
#plt.show()
labels = fcluster(merging,t=6,criterion='maxclust')
ct = pd.DataFrame({'cluster': labels,'gas': y['gas'].values})
ct = pd.crosstab(ct['gas'],ct['cluster'],margins=True)
print(ct)