import pandas as pd
import numpy as np
from acquire import get_iris_data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

df_i = get_iris_data()
def prep_iris(df_i):
    df_i.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    df_i.rename(columns = {'species_name' : 'species'}, inplace = True)
    int_encoder = LabelEncoder()
    int_encoder.fit(df_i.species)
    df_i.species = int_encoder.transform(df_i.species)
    species_array = np.array(df_i.species)
    return df_i
prep_iris(df_i)