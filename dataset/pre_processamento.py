import pandas as pd
data = pd.read_csv('../dataset/batch1.dat', sep=' ', header=None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

def transformar_data(linhas):
    gas_type, concentration = linhas[0].split(';')
    concentration = float(concentration)
    gas_type = int(gas_type)

    features = {}

    for i in range(1, len(linhas)):
        if pd.notna(linhas[i]):
            feature_index, feature_value = linhas[i].split(':')
            features[f"feature_{int(feature_index)}"] = float(feature_value)

    # Return a dictionary with the gas type, concentration, and features
    return {"gas_type": gas_type, "concentration": concentration, **features}
data = data.apply(transformar_data, axis=1, result_type='expand')
#print(data)
print(data['gas_type'].value_counts().sort_index())