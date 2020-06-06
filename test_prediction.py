
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# Explorar datos
# ============================================================

# Cargar los datos
dataset = pd.read_csv("testfiles/train.csv", index_col=0)

# 
dims = dataset.shape
describe = dataset.describe()

# Eliminar valores nulos
nulos = dataset.isnull()
conteo_nulos = nulos["NU_NOTA_MT"].value_counts()
dataset = dataset.dropna(subset=["NU_NOTA_MT"]).reset_index(drop=True)

# Seleccionar columnas categoricas
categorical_columns =\
    dataset.select_dtypes(include=['object']).columns.to_list()
categorical_columns.remove("NU_INSCRICAO")

# Contar valores nan en las columnas y seleccionar aquellas que tienen más de 
# un 75 % de los datos
null_columns =\
    pd.DataFrame(dataset.isnull().sum(axis=0)).reset_index(drop=False)
null_columns.columns = ['columnas', 'Conteo valores nan']

total_valores = dataset.shape[0]
null_columns['Percent nan'] =\
    (1 - (null_columns['Conteo valores nan'] / total_valores)) * 100

null_columns_filtered = null_columns[null_columns['Percent nan'] > 50]
columns_correct = null_columns_filtered['columnas'].to_list()
dataset_filtered = dataset[columns_correct]


dataset_filtered = dataset_filtered.dropna()




# enc = OneHotEncoder(handle_unknown='ignore')
# enc_dataset =\
#     pd.DataFrame(enc.fit_transform(dataset[categorical_columns]).toarray())

# dataset = dataset.join(enc_dataset)



# sns.distplot(dataset['NU_NOTA_MT']);



# #correlation matrix
# corrmat = dataset.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);





# #scatterplot
# sns.set()
# cols = ['NU_NOTA_CN', 'NU_NOTA_MT', 'NU_NOTA_COMP1', 'NU_NOTA_COMP4']
# sns.pairplot(dataset[cols], size = 2.5)
# plt.show();





