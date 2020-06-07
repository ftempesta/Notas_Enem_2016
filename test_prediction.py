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

X = dataset_important
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


# ============================================================
# Data preparation
# ============================================================

# Predictions of the linear regression
y_pred_regression =  linear_regression_processing(X_train, X_test,
                                                  y_train, y_test)
# MAE of regression
mae_regression =\
    mae_error_models(y_pred_regression, y_test, min_y, max_y)
# Standart deviation of error
std_mae_regression =\
    std_mae_error(y_pred_regression, y_test, min_y, max_y)

model_name = "Regression"
plot_results(y_test, y_pred_regression,
             model_name, min_y, max_y)


# ============================================================
# XGBoost model
# ============================================================

# Number of steps on the boosting algorithm
early_stopping_rounds = 200

# Mean absolute error and std of mae error of xgboost algorithm for y:test
y_pred_xgboost, feature_importance =\
    xgboost_processing(X_train, X_test, y_train,
                       y_test, early_stopping_rounds,
                       columns_dataset)

# MAE of xgboost algorithm
mae_xgboost =\
    mae_error_models(y_pred_xgboost, y_test, min_y, max_y)
# Standart deviation of error
std_mae_xgboost =\
    std_mae_error(y_pred_xgboost, y_test, min_y, max_y)

# Plot results XGBoost
model_name = "XGboost"
plot_results(y_test, y_pred_xgboost, model_name, min_y, max_y)


# ============================================================
# Support vector regressor
# ============================================================

y_pred_svr =\
    support_vector_machines_processing(X_train, X_test, y_train, y_test)

# MAE of SVR algorithm
mae_svr =\
    mae_error_models(y_pred_svr, y_test, min_y, max_y)
    
# Standard deviation of mae
std_mae_svr =\
    std_mae_error(y_pred_svr, y_test, min_y, max_y)

# Plot results SVR
model_name = "Support vector regressor"
plot_results(y_test, y_pred_svr, model_name, min_y, max_y)


# ============================================================
# Neural networks
# ============================================================

def neural_net_processing():
    
    





# Select the 30° most important columns

feature_importance = feature_importance.iloc[:30]

important_columns = (feature_importance["feature"]).to_list()

# Calculate correlation just in features importante
dataset_important = dataset[important_columns]

correlation_feature_important =\
    dataset_important.corr().unstack().\
        sort_values().drop_duplicates().reset_index(drop=False)

correlation_feature_important.columns = ["var1", "var2", "correlation"]

correlation_feature_important =\
    correlation_feature_important.sort_values(by=["correlation"],
                                                      ascending=False).\
    iloc[1:].reset_index(drop=True)










# Dataframe of correlation between columns (complete dataset)
correlation_columns =\
    dataset.corr().unstack().sort_values().\
        drop_duplicates().reset_index(drop=False)
correlation_columns.columns = ["var1", "var2", "correlation"]
correlation_columns = correlation_columns.sort_values(by=["correlation"],
                                                      ascending=False).\
    iloc[1:].reset_index(drop=True)


# High correlation columns
high_correlated_columns =\
    correlation_columns[(correlation_columns['correlation'] > 0.25) |
                        (correlation_columns['correlation'] < -0.25)]







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


# for ii in range(correlation_importance_selection.shape[0]):
    
#     var2 = correlation_importance_selection["var2"].iloc[ii]
    
#     for jj in range(len(important_columns)):
#         features =(important_columns[jj]

#         if features==var:
#             print("variable repetida")
 

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





