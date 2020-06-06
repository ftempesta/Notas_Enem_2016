from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

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

