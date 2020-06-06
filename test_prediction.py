import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

from sklearn.decomposition import PCA

from codenation_library import *

# ============================================================
# Explorar datos
# ============================================================

# Load_data
dataset = pd.read_csv("testfiles/train.csv", index_col=0)

# Describe
dims = dataset.shape
describe = dataset.describe()

# ============================================================
# Eliminete NaN
# ============================================================

# Drop rows with nan values in the NU_NOTA_MT column
nulos = dataset.isnull()["NU_NOTA_MT"].value_counts()
dataset = dataset.dropna(subset=["NU_NOTA_MT"]).reset_index(drop=True)

# Count nan values by columns
null_columns =\
    pd.DataFrame(dataset.isnull().sum(axis=0)).reset_index(drop=False)
null_columns.columns = ['columns', 'count_nan_values']

# Calculated percentage of NON null values 
sum_values = dataset.shape[0]
null_columns['percentage_non_nan'] =\
    (1 - (null_columns['count_nan_values'] / sum_values)) * 100

# Select only the columns with less than 50 % of nan's values
null_columns_filtered = null_columns[null_columns['percentage_non_nan'] > 50]
columns_correct = null_columns_filtered['columns'].to_list()

# Filter dataset
dataset_filtered = dataset[columns_correct]

# Filtered dataset with the columns selected before 
dataset = dataset_filtered.dropna()

# ============================================================
# Transform categorical to numeric
# ============================================================

# Target: inscription number and score math test
target = dataset[["NU_INSCRICAO","NU_NOTA_MT"]]

dataset = dataset.drop(columns=["NU_INSCRICAO","NU_NOTA_MT"])


# Seleccionar columnas categoricas
categorical_columns =\
    dataset.select_dtypes(include=['object']).columns.to_list()

# Apply laber encoder to categorical columns
dataset = LaberEncoder(dataset, categorical_columns)


# ============================================================
# Basic linear regression
# ============================================================

X = dataset
y = target["NU_NOTA_MT"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X.asfloat) 


print_model = model.summary()
print(print_model)










# # Aplicar PCA para encontrar componentes principales
# pca = PCA(n_components=100)
# principalComponents = pca.fit_transform(x)




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





