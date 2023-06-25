import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(df, column_name):
    """
    One-hot encoding for categorical variables
    :param df: dataframe
    :param column_name: column name
    :return: dataframe with one-hot encoding
    """

    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    return pd.concat([df, dummies], axis=1)

def one_hot_encoding_sklearn(data_frame, column_name, sparse=False, drop_column=True):
    """
    One-hot encoding for categorical variables
    :param df: dataframe
    :param column_name: column name
    :return: dataframe with one-hot encoding
    """

    one_hot_encoder = OneHotEncoder(sparse=False)
    df_encoded = one_hot_encoder.fit_transform(data_frame[[column_name]])
    df_encoded = pd.DataFrame(df_encoded, columns=one_hot_encoder.categories_[0])
    data_frame = pd.concat([data_frame, df_encoded], axis=1)
    if drop_column:
        data_frame = data_frame.drop([column_name], axis=1)
    return data_frame