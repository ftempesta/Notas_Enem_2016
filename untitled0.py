from sklearn.preprocessing import LabelEncoder

def LaberEncoder(dataset), categorical_columns:
    
    """
    Function to convert categorical columns to numerical
    applying label encoder
    
    """

    # Label enconding of categorical values 
    labelencoder = LabelEncoder()
    
    # Assigning numerical values and storing in another column
    
    for column in categorical_columns:
        dataset[column] =\
            labelencoder.fit_transform(dataset[column])
            
    return dataset
