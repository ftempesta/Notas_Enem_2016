# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Stats models
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Machine learning
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Deep learning
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from sklearn.metrics import mean_squared_error,mean_absolute_error
import keras.backend as K

# Visualization
import matplotlib.pyplot as plt



def labeL_encoder(dataset, categorical_columns):
    """
    Function to convert categorical columns to numerical
    applying label encoder

    Parameters
    ----------
    dataset : DataFrame
        
    categorical_columns : List
        Columns to convert to numerical 

    Returns
    -------
    Dataset with only numeric values applying label encoder
    
    """
    labelencoder = LabelEncoder()
    
    # Assigning numerical values and storing in another column
    
    for column in categorical_columns:
        dataset[column] =\
            labelencoder.fit_transform(dataset[column])
            
    return dataset

    
def one_hot_encoder(dataset, categorical_columns):
    """
    Convert categorical columns to a new columns with only 0 o 1 values
    
    Parameters
    ----------
    dataset : DataFrame
        
    categorical_columns : List
        Columns to convert to numerical 
    Returns
    -------
    Dataset with only numeric values applying one hot encoder
    """
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc_dataset =\
        pd.DataFrame(enc.fit_transform(dataset).toarray())
    dataset = dataset.join(enc_dataset)
    dataset =\
        dataset.drop(columns=categorical_columns).reset_index(drop=True)
        
    return dataset




def mae_error_models(y_pred, y_test, min_y, max_y):
    """
    Function to get the Mean Absolute error of wholes models    

    Parameters
    ----------
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    min_y : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.

    Returns
    -------
    mae_error : TYPE
        DESCRIPTION.

    """
    y_pred_scaled = y_pred*(max_y-min_y) + min_y
    y_test = y_test*(max_y-min_y)+min_y
    
    mae_error = abs((y_pred_scaled - y_test).mean())
    
    return mae_error

def std_mae_error(y_pred, y_test, min_y, max_y):
    """
    Function to analyse standar distribution of mae error

    Parameters
    ----------
    y_pred : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    min_y : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.

    Returns
    -------
    std_mae : TYPE
        DESCRIPTION.

    """
    
    y_pred_scaled = y_pred*(max_y-min_y) + min_y
    y_test = y_test*(max_y-min_y)+min_y
    
    std_mae = (y_pred_scaled - y_test).std()
    
    return std_mae


def linear_regression_processing(X_train, X_test, y_train, y_test):
    """
    Create a linear regression to this data to establish a base line of
    error on the next models

    Parameters
    ----------
    X_train : Dataframe
        Train features
    X_train : Dataframe
        Test feature     
    y_train : Series
        Train target
    y_test : Series
        Test target

    Returns
    -------
    return the error of the model

    """    
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    
    model = sm.OLS(y_train, X_train).fit()    
    predictions = model.predict(X_test) 

    return predictions



def xgboost_processing(X_train, X_test, y_train,
                       y_test, early_stopping_rounds,
                       columns_dataset):
    """
    

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    early_stopping_rounds : TYPE
        DESCRIPTION.
    columns_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    y_pred : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    """
    
    columns_X = columns_dataset
    
    # Reordenar en la forma (algo,)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # Conjunto de testeo
    eval_set = [(X_test, y_test)]
    
    # XGBRegressor MODEL
    regressor = xgboost.XGBRegressor()
    
    # Entrenar el modelo
    regressor.fit(X_train, y_train, eval_set = [(X_test, y_test)],
                  early_stopping_rounds = 20)
    
    # Generar predicciones
    y_pred = regressor.predict(X_test)
    
    feature_importance =\
        regressor.get_booster().get_score(importance_type="weight")
    
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())

    data = pd.DataFrame(data=values,
                        index=keys,
                        columns=["score"]).sort_values(by = "score",
                                                       ascending=False)

    list_feature_importance = []  

    # Get the feature importance on a dataframe by column name
    for col,score in zip(columns_X,regressor.feature_importances_):
        list_feature_importance.append([col, score])
        
    data =pd.DataFrame(list_feature_importance,
                     columns=["feature", "importance_score"])
    data = data.sort_values(by="importance_score",
                            ascending=False).reset_index(drop=True)
    
    
    # Plot importance
    plt.figure(figsize=(40,20))
    plot_importance(regressor)
    pyplot.show()
    plt.show()

    return  y_pred, data


def support_vector_machines_processing(X_train, X_test, y_train, y_test):
    """
    

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    Returns
    -------
    y_pred : TYPE
        DESCRIPTION.

    """

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # Support vector machines regressor
    regressor = SVR(kernel='rbf')
    
    # Train model
    regressor.fit(X_train,y_train)
    
    # Generate predictions
    y_pred = regressor.predict(X_test)

    return y_pred



def plot_results(y_test, y_pred, model_name, min_y, max_y):
    """
    

    Parameters
    ----------
    y_test : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.
    model_name : TYPE
        DESCRIPTION.
    min_y : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    y_pred_scaled = y_pred*(max_y-min_y) + min_y
    y_test = y_test*(max_y-min_y)+min_y

    # Plot of real values vs predicted
    plt.figure(figsize=(20,10))
    plt.scatter(y_test, y_pred_scaled, color = 'blue')
    plt.scatter(y_test, y_test, color = 'red')
    plt.title('Model :'+ " "+model_name, fontsize=30)
    plt.xlabel('Math test values', fontsize=30)
    plt.ylabel('Predictions', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    return plt.show()
