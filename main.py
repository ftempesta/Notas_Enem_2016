import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np


from codenation_library import *

# ============================================================
# Explorar datos
# ============================================================

# Load_data
dataset = pd.read_csv('testfiles/train.csv', index_col=0)

# Describe
dims = dataset.shape
describe = dataset.describe()

# ============================================================
# Eliminete NaN's in NU_NOTA_MT and irrelevats columns
# ============================================================

# Drop rows with nan values in the NU_NOTA_MT column
nulos = dataset.isnull()['NU_NOTA_MT'].value_counts()
dataset = dataset.dropna(subset=['NU_NOTA_MT']).reset_index(drop=True)


# Delete irrelevant columns 
columns_dataset = dataset.columns.to_list()

columns_to_delete = ['NU_ANO', 'CO_MUNICIPIO_RESIDENCIA',
                     'SG_UF_RESIDENCIA', 'CO_MUNICIPIO_NASCIMENTO',
                     'SG_UF_NASCIMENTO', 'CO_ESCOLA', 'CO_MUNICIPIO_ESC', 
                     'SG_UF_ESC', 'SG_UF_ENTIDADE_CERTIFICACAO',
                     'CO_MUNICIPIO_PROVA', 'SG_UF_PROVA', 'CO_PROVA_CN',
                     'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
                     'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC',
                     'TX_RESPOSTAS_MT', 'TX_GABARITO_CN', 'TX_GABARITO_CH',
                     'TX_GABARITO_LC', 'TX_GABARITO_MT', 'TP_STATUS_REDACAO',
                     'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
                     'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
                     'NO_MUNICIPIO_NASCIMENTO']

dataset = dataset.drop(columns=columns_to_delete)

# ============================================================
# Eliminete columns with less than 50 % of the values
# ============================================================

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

# Target: inscription number and score math testv
target = dataset[['NU_INSCRICAO','NU_NOTA_MT']]
dataset = dataset.drop(columns=['NU_INSCRICAO','NU_NOTA_MT'])

# ============================================================
# Transform categorical to numeric
# ============================================================

#label_encoder_columns = ['TP_SEXO', 'NO_MUNICIPIO_RESIDENCIA', 
#'NO_MUNICIPIO_PROVA']
categorical_columns =\
     dataset.select_dtypes(include=['object']).columns.to_list()
dataset = labeL_encoder(dataset, categorical_columns)

# # Select categorical columns
# categorical_columns =\
#     dataset.select_dtypes(include=['object']).columns.to_list()
# # Convert categorical columns into numerical column
# dataset = one_hot_encoder(dataset, categorical_columns)

# ============================================================
# Data preparation
# ============================================================

X = dataset
# Columns 
columns_dataset = X.columns
y = target["NU_NOTA_MT"]


# Normalize features and split
sc_X = MinMaxScaler(feature_range = (0,1))
X = sc_X.fit_transform(X)

sc_y = MinMaxScaler(feature_range = (0,1))
y = y.to_numpy()
y = y.reshape(-1,1)
y = sc_y.fit_transform(y)

# Min and max to rescale the data
min_y = sc_y.data_min_
min_y = min_y[0]
max_y = sc_y.data_max_
max_y = max_y[0]

# Split datasets
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=42)


# XGboost hyperparameters
early_stopping_rounds = 200
# Neural net hyperparameters
batch_size = 256
epochs = 40
patience = 100
min_delta = 0.5

# ============================================================
# Algoritm selection
# ============================================================

df_prediction, df_mae_score =\
    master_regression_algorithm(X_train, X_test, y_train, y_test,
                                early_stopping_rounds, batch_size,
                                epochs, patience, min_delta,
                                min_y, max_y, columns_dataset)

y_test_rescaled = y_test*(max_y-min_y)+min_y






# # Select the 30° most important columns

# feature_importance = feature_importance.iloc[:30]

# important_columns = (feature_importance["feature"]).to_list()

# # Calculate correlation just in features importante
# dataset_important = dataset[important_columns]

# correlation_feature_important =\
#     dataset_important.corr().unstack().\
#         sort_values().drop_duplicates().reset_index(drop=False)

# correlation_feature_important.columns = ["var1", "var2", "correlation"]

# correlation_feature_important =\
#     correlation_feature_important.sort_values(by=["correlation"],
#                                                       ascending=False).\
#     iloc[1:].reset_index(drop=True)

# # Dataframe of correlation between columns (complete dataset)
# correlation_columns =\
#     dataset.corr().unstack().sort_values().\
#         drop_duplicates().reset_index(drop=False)
# correlation_columns.columns = ["var1", "var2", "correlation"]
# correlation_columns = correlation_columns.sort_values(by=["correlation"],
#                                                       ascending=False).\
#     iloc[1:].reset_index(drop=True)


# # High correlation columns
# high_correlated_columns =\
#     correlation_columns[(correlation_columns['correlation'] > 0.25) |
#                         (correlation_columns['correlation'] < -0.25)]







# correlation_importance_selection1 = pd.merge(feature_importance,
#                                             high_correlated_columns,
#                                             how='outer',
#                                             left_on=['feature'],
#                                             right_on =['var1']).dropna()

# correlation_importance_selection2 = pd.merge(feature_importance,
#                                             high_correlated_columns,
#                                             how='outer',
#                                             left_on=['feature'],
#                                             right_on =['var2']).dropna()

# correlation_importance_selection = pd.merge(correlation_importance_selection1,
#                                             correlation_importance_selection2,
#                                             how='outer',
#                                             left_on=['feature'],
#                                             right_on =['feature']).dropna()


 

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





