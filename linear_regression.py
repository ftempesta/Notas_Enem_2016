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
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE

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
# Select only notes
y = dataset["NU_NOTA_MT"]

num_1 = pd.DataFrame(dataset['NU_INSCRICAO']).reset_index(drop=True)
dataset = dataset.drop(columns=['NU_INSCRICAO', 'NU_NOTA_MT']).reset_index(drop=True)

num_2 = pd.DataFrame(dataset_test['NU_INSCRICAO']).reset_index(drop=True)
dataset_test = dataset_test.drop(columns='NU_INSCRICAO').reset_index(drop=True)


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
dataset = dataset[columns_correct]
dataset_test = dataset_test[columns_correct]

dataset = dataset.reset_index(drop=True)
dataset_test = dataset_test.reset_index(drop=True)


dataset_total = pd.concat([dataset, dataset_test], axis=0)
    
dataset_total = dataset_total.fillna(dataset_total.mean())
dataset_total = dataset_total.reset_index(drop=True)


# ============================================================
# Transform categorical to numeric
# ============================================================

categorical_columns =\
     dataset_total.select_dtypes(include=['object']).columns.to_list()
    

dataset_total = label_encoder(dataset_total, categorical_columns)

# ============================================================
# Data preparation
# ============================================================

train_indices = np.arange(0,dataset.shape[0]).tolist()
test_indices = np.arange(dataset.shape[0], dataset_total.shape[0]).tolist()

X = dataset_total
columns_dataset = X.columns.to_list()

# X_train = X.iloc[train_indices]
# X_test = X.iloc[test_indices]

# y_train = y

# Normalize features and split
sc_X = MinMaxScaler(feature_range = (0,1))
X = sc_X.fit_transform(X)

# Min and max to rescale the data
min_x = sc_X.data_min_
min_x = min_x[0]
max_x = sc_X.data_max_
max_x = max_x[0]

X_train  = X[0:dataset.shape[0]]
X_test  = X[dataset.shape[0]:]

X = X_train


#no of features
nof_list=np.arange(1,X.shape[1])            
high_score=0


# list_scores_models = []
# for j in range(2):

#     #Variable to store the optimum features
#     nof=0           
#     score_list =[]
    
#     for n in range(len(nof_list)):
#         X_train, X_test, y_train, y_test = train_test_split(X,y,
#                                                             test_size = 0.3,
#                                                             random_state=j)
#         model = LinearRegression()
#         rfe = RFE(model,nof_list[n])
#         X_train_rfe = rfe.fit_transform(X_train,y_train)
#         X_test_rfe = rfe.transform(X_test)
#         model.fit(X_train_rfe,y_train)
#         score = model.score(X_test_rfe,y_test)
#         score_list.append(score)
#         if(score>high_score):
#             high_score = score
#             nof = nof_list[n]
            
#     list_scores_models.append([nof, high_score])
    
# df_scores = pd.DataFrame(list_scores_models)
# df_scores.columns = ["number features", "score"]

# df_scores.to_csv("nof_scores_hh.csv")


# df_scores = df_scores[df_scores["number features"]>0]
# nof_min = df_scores['number features'].min()
# nof_max = df_scores['number features'].max()
# nof_mean = df_scores['number features'].mean()





cols = list(dataset_total.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 27)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index.to_list()
print(selected_features_rfe)


dataset_total = dataset_total[selected_features_rfe]

X = dataset_total.to_numpy()

X_train  = X[0:dataset.shape[0]]
X_test  = X[dataset.shape[0]:]

X = X_train




model = LinearRegression()

model.fit(X, y)
predictions = model.predict(X_test)
predictions = pd.DataFrame(predictions)


resultados_finales =\
    pd.concat([num_2, predictions], axis=1)


resultados_finales.columns = ['NU_INSCRICAO', 'NU_NOTA_MT']

resultados_finales.to_csv('answer.csv', index=False)
