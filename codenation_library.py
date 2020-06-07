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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
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
    
    
    # # Plot importance
    # plt.figure(figsize=(40,20))
    # plot_importance(regressor)
    # pyplot.show()
    # plt.show()

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


def neural_net_processing(X_train, X_test, y_train, y_test,
                          batch_size, epochs, patience, min_delta):
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
    batch_size : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.
    patience : TYPE
        DESCRIPTION.
    min_delta : TYPE
        DESCRIPTION.

    Returns
    -------
    y_pred : TYPE
        DESCRIPTION.

    """
    
    # Model implementation
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='relu'))
    model.summary()

    # Compile optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')

    keras.callbacks.Callback()

    stop_condition = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode ='min',
                                              patience=patience,
                                              verbose=1,
                                              min_delta=min_delta,
                                              restore_best_weights=True)

    learning_rate_schedule = ReduceLROnPlateau(monitor="val_loss",
                                         factor=0.5,
                                         patience=100,
                                         verbose=1,
                                         mode="auto",
                                         cooldown=0,
                                         min_lr=5E-3)

    callbacks = [stop_condition, learning_rate_schedule]


    history = model.fit(X_train, y_train,validation_split=0.1,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=False,
                        verbose=1,
                        callbacks=callbacks)

    # Hist training 
    fig,ax = plt.subplots(1,figsize=(16, 8))
    ax.plot(history.history['loss'],'k', linewidth=2)
    ax.plot(history.history['val_loss'],'r', linewidth=2)
    ax.set_xlabel('Epochs', fontname="Arial", fontsize=14)
    ax.set_ylabel('Mean squared error', fontname="Arial", fontsize=14)
    ax.legend(['Training', 'Validation'], loc='upper left',prop={'size': 14})

    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    plt.show()
    
    fig.savefig("training_results.png")

    # Score del modelo entrenado
    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Mean squared error, Test:', scores)


    # Predictions on y_test
    y_pred = model.predict(X_test)
    
    # Save the model as .h5
    model.save("model_checkpoint.h5")

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



def master_regression_algorithm(X_train, X_test, y_train, y_test,
                                early_stopping_rounds,
                                batch_size,
                                epochs,
                                patience,
                                min_delta,
                                min_y, max_y,
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
    batch_size : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.
    patience : TYPE
        DESCRIPTION.
    min_delta : TYPE
        DESCRIPTION.
    min_y : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.
    columns_dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    df_prediction : TYPE
        DESCRIPTION.
    df_mae_score : TYPE
        DESCRIPTION.

    """
    

    df_prediction = pd.DataFrame()
    
    mae_score = []
    
    # ============================================================
    # Regression algorithm 
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
    
    df_prediction['regression_prediction'] = y_pred_regression

    mae_score.append(['Regression',mae_regression])
    
    # ============================================================
    # XGBoost model
    # ============================================================
    
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
    
    df_prediction['xgboost_prediction'] = y_pred_xgboost

    mae_score.append(['XGboost',mae_xgboost])
    
    
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
    
    df_prediction['svr_prediction'] = y_pred_svr

    mae_score.append(['Support'+'\n'+'vector'+'\n'+'regressor',mae_svr])
    
    # ============================================================
    # Neural networks
    # ============================================================
    
    y_pred_nn = neural_net_processing(X_train, X_test, y_train, y_test,
                                      batch_size,
                                      epochs,
                                      patience,
                                      min_delta)
    
    # MAE of Neural network
    mae_nn =\
        mae_error_models(y_pred_nn, y_test, min_y, max_y)
        
    # Standard deviation of mae
    std_mae_nn =\
        std_mae_error(y_pred_nn, y_test, min_y, max_y)
    
    # Plot results Neural networks
    model_name = "Neural network"
    plot_results(y_test, y_pred_nn, model_name, min_y, max_y)

    df_prediction['nn_prediction'] = y_pred_nn
    mae_score.append(['Neural'+'\n'+'network',mae_nn])
    

    # Dataframe with mean absolute errors
    df_mae_score = pd.DataFrame(mae_score, columns=['Algorithm',
                                                    'Mean absolute error'])

    df_mae_score.plot.bar(x='Algorithm', y='Mean absolute error',
                          rot=0, color='r')
    
    # Re-scale the predictions
    df_prediction = df_prediction.apply(lambda x: x*(max_y-min_y)+min_y)
    
    return df_prediction, df_mae_score


