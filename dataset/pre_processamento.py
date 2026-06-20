import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

batch1 = pd.read_csv('../dataset/batch1.dat', sep=' ', header=None)
batch2 = pd.read_csv('../dataset/batch2.dat', sep=' ', header=None)
batch3 = pd.read_csv('../dataset/batch3.dat', sep=' ', header=None)
batch4 = pd.read_csv('../dataset/batch4.dat', sep=' ', header=None)
batch5 = pd.read_csv('../dataset/batch5.dat', sep=' ', header=None)
batch6 = pd.read_csv('../dataset/batch6.dat', sep=' ', header=None)
batch7 = pd.read_csv('../dataset/batch7.dat', sep=' ', header=None)
batch8 = pd.read_csv('../dataset/batch8.dat', sep=' ', header=None)
batch9 = pd.read_csv('../dataset/batch9.dat', sep=' ', header=None)
batch10 = pd.read_csv('../dataset/batch10.dat', sep=' ', header=None)
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)

def transformar_data(linhas):
    gas, concentracao = linhas[0].split(';')
    concentracao = float(concentracao)
    gas = int(gas)

    features = {}

    for i in range(1, len(linhas)):
        if pd.notna(linhas[i]):
            index, valor = linhas[i].split(':')
            features[f"feature_{int(index)}"] = float(valor)

    return {"gas": gas, "concentracao": concentracao, **features}
batch1 = batch1.apply(transformar_data, axis=1, result_type='expand')
batch2 = batch2.apply(transformar_data, axis=1, result_type='expand')
batch3 = batch3.apply(transformar_data, axis=1, result_type='expand')
batch4 = batch4.apply(transformar_data, axis=1, result_type='expand')
batch5 = batch5.apply(transformar_data, axis=1, result_type='expand')
batch6 = batch6.apply(transformar_data, axis=1, result_type='expand')
batch7 = batch7.apply(transformar_data, axis=1, result_type='expand')
batch8 = batch8.apply(transformar_data, axis=1, result_type='expand')
batch9 = batch9.apply(transformar_data, axis=1, result_type='expand')
batch10 = batch10.apply(transformar_data, axis=1, result_type='expand')
all_batch = pd.concat([batch1, batch2, batch3, batch4,batch5,batch6,batch7,batch8,batch9,batch10], ignore_index=True)
#print(batch1)
