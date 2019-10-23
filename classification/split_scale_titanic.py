# Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe.

import pandas as pd
from env import host, user, password

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import wrangle
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler




#min_max_scaler()
def min_max_scaler(df_titanic):
    mm_df_titanic_age = MinMaxScaler(copy=True, feature_range=(0,80)).fit(df_titanic.age)
    mm_df_titanic_fare = MinMaxScaler(copy=True, feature_range=(0,512)).fit(df_titanic.age)
    
    return mm_df_titanic_age, mm_df_titanic_fare
