import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost
import numpy as np
import pickle 
from statsmodels.regression.linear_model import OLSResults


from codenation_library_matheus import *

# ============================================================
# Explorar datos
# ============================================================

# Load_data
dataset = pd.read_csv('testfiles/train.csv', index_col=0)
dataset_test = pd.read_csv('testfiles/test.csv')

columns_test = dataset_test.columns.to_list()
columns_test.append('NU_NOTA_MT')

dataset = dataset[columns_test]


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

# columns_to_delete = ['NU_ANO', 'CO_MUNICIPIO_RESIDENCIA',
#                      'SG_UF_RESIDENCIA', 'CO_MUNICIPIO_NASCIMENTO',
#                      'SG_UF_NASCIMENTO', 'CO_ESCOLA', 'CO_MUNICIPIO_ESC', 
#                      'SG_UF_ESC', 'SG_UF_ENTIDADE_CERTIFICACAO',
#                      'CO_MUNICIPIO_PROVA', 'SG_UF_PROVA', 'CO_PROVA_CN',
#                      'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
#                      'TX_RESPOSTAS_CN', 'TX_RESPOSTAS_CH', 'TX_RESPOSTAS_LC',
#                      'TX_RESPOSTAS_MT', 'TX_GABARITO_CN', 'TX_GABARITO_CH',
#                      'TX_GABARITO_LC', 'TX_GABARITO_MT', 'TP_STATUS_REDACAO',
#                      'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
#                      'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
#                      'NO_MUNICIPIO_NASCIMENTO']

# dataset = dataset.drop(columns=columns_to_delete)

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

categorical_columns =\
     dataset.select_dtypes(include=['object']).columns.to_list()
dataset = labeL_encoder(dataset, categorical_columns)

# Apply label encoder
dataset_filtered = labeL_encoder(dataset_filtered, categorical_columns)
total_columns = dataset.columns.to_list()


# ============================================================
# Data preparation
# ============================================================
dataset = dataset.reset_index(drop=True)
dataset_filtered = dataset_filtered.reset_index(drop=True)

print(dataset_filtered.shape)
print(dataset.shape)


indices_list = dataset.index.to_list()

train_indices, test_indices = train_test_split(indices_list, test_size=0.2)

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

min_x = sc_X.data_min_
min_x = min_x[0]
max_x = sc_X.data_max_
max_x = max_x[0]

y_train = y[train_indices,:]
y_test = y[test_indices,:]

X_train = X[train_indices,:]
X_test = X[test_indices,:]


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

df_prediction, df_mae_score, regression, xgboost, svr, nn =\
    master_regression_algorithm(X_train, X_test, y_train, y_test,
                                early_stopping_rounds, batch_size,
                                epochs, patience, min_delta,
                                min_y, max_y, columns_dataset)


X_test = pd.DataFrame(X_test, columns=columns_dataset)
X_test = X_test.apply(lambda x: x*(max_x-min_x)+min_x)


y_test_rescaled = y_test*(max_y-min_y)+min_y
y_test_rescaled = pd.DataFrame(y_test_rescaled, columns=["Resultados test"])


inscription_number = dataset_filtered['NU_INSCRICAO']
inscription_number = pd.DataFrame(inscription_number.iloc[test_indices]).reset_index()

result = pd.concat([X_test, y_test_rescaled,inscription_number], axis=1)


# ============================================================
# Apply to test set
# ============================================================

columns_test = X_test.columns.to_list()

dataset_final_test = dataset_test[columns_test].reset_index(drop=True)
inscription_number_test = pd.DataFrame(dataset_test['NU_INSCRICAO'])


pkl_file = open('label_encoding.pkl', 'rb')
label_encoder = pickle.load(pkl_file) 
pkl_file.close()


for column in categorical_columns:
    dataset_final_test[column] =\
        label_encoder.fit_transform(dataset_final_test[column])

dataset_final_test = dataset_final_test.reset_index(drop=True)
dataset_final_test = dataset_final_test.fillna(dataset_final_test.mean())

# Regression

X_final_test = dataset_final_test
# Normalize features 
X_final_test = sc_X.fit_transform(X_final_test)


# XGBoost
xgboost = pickle.load(open("xgboost.pickle.dat", "rb"))
# make predictions for test data
predictions_xgboost = xgboost.predict(X_final_test)
predictions_xgboost = predictions_xgboost*(max_y-min_y)+min_y
predictions_xgboost = pd.DataFrame(predictions_xgboost)
# SVR
predictions_svr = svr.predict(X_final_test)
predictions_svr = predictions_svr*(max_y-min_y)+min_y
predictions_svr = pd.DataFrame(predictions_svr)

# Neural net
prediction_nn = nn.predict(X_final_test)
prediction_nn = prediction_nn*(max_y-min_y)+min_y
prediction_nn = pd.DataFrame(prediction_nn)


resultados_finales =\
    pd.concat([inscription_number_test, 
               predictions_xgboost, predictions_svr, prediction_nn], axis=1)

resultados_finales.columns = ["NU_INSCRICAO", "xgboost", "svr", "nn"]


csv1 = resultados_finales[["NU_INSCRICAO", "xgboost"]]
csv1.columns = ["NU_INSCRICAO", "NU_NOTA_MT"]

csv = csv1[csv1['NU_NOTA_MT']<=1000].mean()[0]


for value in range(csv1.shape[0]):
    
    nota_i = csv1['NU_NOTA_MT'].iloc[value]
    
    if nota_i >= 1000:
        
        csv1['NU_NOTA_MT'].iloc[value] = csv
        



csv1.to_csv('answer.csv', index=False)












# prediction_regression = regression.predict(X_final_test)
# prediction_regression = prediction_regression*(max_y-min_y)+min_y








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





