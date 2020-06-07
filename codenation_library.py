from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import pandas as pd
import numpy as np
import xgboost
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def LaberEncoder(dataset, categorical_columns):
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

    
def OneHotEncoder(dataset, categorical_columns):
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
        
    return dataset


def LinearRegression(X,y):
    """
    Create a linear regression to this data to establish a base line of
    error on the next models

    Parameters
    ----------
    X : Dataframe
        Features.
    y : Series
        Targets.

    Returns
    -------
    return the error of the model

    """
    
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X) 
    
    error =  model.rsquared

    return error



def xgboost_processing(X,y, early_stopping_rounds):
    
    """
    Analyse feature importance with xgboost

    Parameters
    ----------
    X : Dataframe
        Features.
    y : Series
        Targets.
    early_stopping_rounds: int
        number of round in the boosting algorithm 
        
        

    Returns
    -------
    return the error of the model
    
    """

    # Normalizar carácterirsticas y etiquetas
    sc_X = MinMaxScaler(feature_range = (0,1))
    X = sc_X.fit_transform(X)
    
    sc_y = MinMaxScaler(feature_range = (0,1))
    y = y.to_numpy()
    y = y.reshape(-1,1)
    y = sc_y.fit_transform(y)
    
    # Dividir los conjuntos de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    min_y = sc_y.data_min_
    min_y = min_y[0]
    max_y = sc_y.data_max_
    max_y = max_y[0]
    
    # Re escalar los datos
    y_pred_scaled = y_pred*(max_y-min_y) + min_y
    y_test = y_test*(max_y-min_y)+min_y
    
    # Obtener el MAE Acumulado
    mae_acum = abs(y_pred-y_test)
    
    # Promedio y desviación estandar de los mae's
    total_mae = mae_acum.mean()
    std = mae_acum.std()
    
    plt.figure(figsize=(40,20))
    plt.scatter(y_test, y_pred_scaled, color = 'blue')
    plt.scatter(y_test, y_test, color = 'red')
    plt.title('Modelo "XGBRegressor"')
    plt.xlabel('Cantidades reales')
    plt.ylabel('Predicción XGBRegressor de cantidades')
    plt.show()
    

    plt.figure(figsize=(40,20))
    plot_importance(regressor)
    pyplot.show()
    plt.show()

    return total_mae, std, mae_acum






