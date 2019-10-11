# Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe.

import pandas as pd
from env import host, user, password

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import wrangle
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#1. split_my_data(X, y)
def split_my_data():
    df = wrangle.wrangle_telco()
    X = df.drop(columns=['total_charges'])
    y = df.total_charges
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8, random_state = 123)
    return x_train, x_test, y_train, y_test




def standard_scaler(x_train, x_test, y_train, y_test):
    scaler_x_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(x_train[['monthly_charges', 'tenure']])
    scaler_x_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(x_test[['monthly_charges', 'tenure']])
    scaler_y_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(pd.DataFrame(y_train))
    scaler_y_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(pd.DataFrame(y_test))
    
#scaler.transform
    train_x_scaled_data = scaler_x_train.transform(x_train[['monthly_charges', 'tenure']])
    test_x_scaled_data = scaler_x_test.transform(x_test[['monthly_charges', 'tenure']])
    train_y_scaled_data = scaler_y_train.transform(pd.DataFrame(y_train))
    test_y_scaled_data = scaler_y_test.transform(pd.DataFrame(y_test))
    return train_x_scaled_data, test_x_scaled_data, train_y_scaled_data, test_y_scaled_data



#scale_inverse()
def scale_inverse(x_train, x_test, y_train, y_test):
    inverse_scaler_x_train = train_x_scaled_data.inverse_transform(x_train[['monthly_charges', 'tenure']])
    inverse_scaler_x_test = test_x_scaled_data.inverse_transform(x_test[['monthly_charges', 'tenure']])
    inverse_scaler_y_train = train_y_scaled_data.inverse_transform(pd.DataFrame(y_train))
    inverse_scaler_y_test = test_y_scaled_data.inverse_transform(pd.DataFrame(y_test))
    return inverse_scaler_x_train, inverse_scaler_x_test, inverse_scaler_y_train, inverse_scaler_y_test  
    

#uniform_scaler()
def uniform_scaler(x_train, x_test, y_train, y_test):
    u_scaler_x_train = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(x_train[['monthly_charges', 'tenure']])
    u_scaler_x_test = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(x_test[['monthly_charges', 'tenure']])
    u_scaler_y_train = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(pd.DataFrame(y_train))
    u_scaler_y_test = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(pd.DataFrame(y_test))

    u_train_x_scaled = pd.DataFrame(u_scaler_x_train.transform(x_train[['monthly_charges', 'tenure']]))
    u_test_x_scaled = pd.DataFrame(u_scaler_x_test.transform(x_test[['monthly_charges', 'tenure']]))
    u_train_y_scaled = pd.DataFrame(u_scaler_y_train.transform(pd.DataFrame(y_train)))
    u_test_y_scaled = pd.DataFrame(u_scaler_y_test.transform(pd.DataFrame(y_test)))
    return u_train_x_scaled, u_test_x_scaled, u_train_y_scaled, u_test_y_scaled




#gaussian_scaler()
def gaussian_scaler(x_train, x_test, y_train, y_test):
    g_x_train_scaler =PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(x_train[['monthly_charges', 'tenure']])
    g_x_test_scaler =PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(x_test[['monthly_charges', 'tenure']])
    g_y_train_scaler =PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(pd.DataFrame(y_train))
    g_y_test_scaler =PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(pd.DataFrame(y_test))

    g_x_train_scaled = pd.DataFrame(g_x_train_scaler.transform(x_train[['monthly_charges', 'tenure']]))
    g_x_test_scaled = pd.DataFrame(g_x_test_scaler.transform(x_test[['monthly_charges', 'tenure']]))
    g_y_train_scaled = pd.DataFrame(g_y_train_scaler.transform(pd.DataFrame(y_train)))
    g_y_test_scaled = pd.DataFrame(g_y_test_scaler.transform(pd.DataFrame(y_test)))
    return g_x_train_scaled, g_x_test_scaled, g_y_train_scaled, g_y_test_scaled




#min_max_scaler()
def min_max_scaler(x_train, x_test, y_train, y_test):
    mm_x_train_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(x_train[['monthly_charges', 'tenure']])
    mm_x_test_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(x_test[['monthly_charges', 'tenure']])
    mm_y_train_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(pd.DataFrame(y_test))
    mm_y_test_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(pd.DataFrame(y_test))

    mm_x_train_scaled = pd.DataFrame(mm_x_train_scaler.transform(x_train[['monthly_charges', 'tenure']]))
    mm_x_test_scaled = pd.DataFrame(mm_x_test_scaler.transform(x_test[['monthly_charges', 'tenure']]))
    mm_y_train_scaled = pd.DataFrame(mm_y_train_scaler.transform(pd.DataFrame(y_train)))
    mm_y_test_scaled = pd.DataFrame(mm_y_test_scaler.transform(pd.DataFrame(y_test)))
    return mm_x_train_scaled, mm_x_test_scaled, mm_y_train_scaled, mm_y_test_scaled



#iqr_robust_scaler()
def iqr_robust_scaler(x_train, x_test, y_train, y_test):
    iqr_x_train_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(x_train[['monthly_charges', 'tenure']])
    iqr_x_test_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(x_test[['monthly_charges', 'tenure']])
    iqr_y_train_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(pd.DataFrame(y_test))
    iqr_y_test_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(pd.DataFrame(y_test))

    iqr_x_train_scaled = pd.DataFrame(iqr_x_train_scaler.transform(x_train[['monthly_charges', 'tenure']]))
    iqr_x_test_scaled = pd.DataFrame(iqr_x_test_scaler.transform(x_test[['monthly_charges', 'tenure']]))
    iqr_y_train_scaled = pd.DataFrame(iqr_y_train_scaler.transform(pd.DataFrame(y_train)))
    iqr_y_test_scaled = pd.DataFrame(iqr_y_test_scaler.transform(pd.DataFrame(y_test)))
    return iqr_x_train_scaled, iqr_x_test_scaled, iqr_y_train_scaled, iqr_y_test_scaled

