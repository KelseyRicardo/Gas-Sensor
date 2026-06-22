import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from pre_processamento import X_ss, X_mm, all_batch_numbers

# Escolha qual normalização deseja usar
X = X_ss          # Z-Score
# X = X_mm        # Min-Max

y = all_batch_numbers['gas']


# TREINO E TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ==========================
# MODELO SEM PCA
# ==========================

knn_original = KNeighborsClassifier(
    n_neighbors=5
)

knn_original.fit(X_train, y_train)

y_pred_original = knn_original.predict(X_test)

# ==========================
# PCA
# ==========================

pca = PCA(
    n_components=0.95
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\n===== PCA =====")
print("Atributos originais:", X_train.shape[1])
print("Componentes PCA:", X_train_pca.shape[1])

print(
    "Variância explicada:",
    round(
        pca.explained_variance_ratio_.sum()*100,
        2
    ),
    "%"
)

# ==========================
# MODELO COM PCA
# ==========================

knn_pca = KNeighborsClassifier(
    n_neighbors=5
)

knn_pca.fit(
    X_train_pca,
    y_train
)

y_pred_pca = knn_pca.predict(
    X_test_pca
)

# ==========================
# FUNÇÃO DE AVALIAÇÃO
# ==========================

def avaliar(y_true, y_pred):

    return {
        "Acurácia":
            accuracy_score(y_true, y_pred),

        "Precisão":
            precision_score(
                y_true,
                y_pred,
                average='weighted'
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                average='weighted'
            ),

        "F1-score":
            f1_score(
                y_true,
                y_pred,
                average='weighted'
            )
    }


# RESULTADOS

resultado_original = avaliar(
    y_test,
    y_pred_original
)

resultado_pca = avaliar(
    y_test,
    y_pred_pca
)

comparacao = pd.DataFrame({
    "Sem PCA": resultado_original,
    "Com PCA": resultado_pca
})

print("\n===== COMPARAÇÃO =====")
print(comparacao)

# MATRIZ DE CONFUSÃO

fig, ax = plt.subplots(
    figsize=(8,6)
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_original,
    ax=ax
)

plt.title("KNN Sem PCA")
plt.show()

fig, ax = plt.subplots(
    figsize=(8,6)
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_pca,
    ax=ax
)

plt.title("KNN Com PCA")
plt.show()

# SALVAR RESULTADOS

comparacao.to_csv(
    "comparacao_pca.csv",
    index=True
)

print(
    "\nTabela salva em comparacao_pca.csv"
)