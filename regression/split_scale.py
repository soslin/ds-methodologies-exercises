# Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe.

import pandas as pd
from env import host, user, password

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import wrangle
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#QuantileTransformer is exclusive
#StandardScaler is z-score

df = wrangle.wrangle_telco()
X = df.drop(columns=['total_charges', 'customer_id'])
y = pd.DataFrame(df.total_charges)

#1. split_my_data(X, y)
def split_my_data(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = .8, random_state = 123)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split_my_data(X,y)



def standard_scaler(x_train, x_test):
    scaler_x_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(x_train[['monthly_charges', 'tenure']])
    scaler_x_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
                .fit(x_test[['monthly_charges', 'tenure']])
#scaler.transform
    train_x_scaled_data = pd.DataFrame(scaler_x_train.transform(x_train), columns = x_train.columns.values).set_index([x_train.index.values])
    
    test_x_scaled_data = pd.DataFrame(scaler_x_test.transform(x_test), columns = x_test.columns.values).set_index([x_test.index.values])
    
    return train_x_scaled_data, test_x_scaled_data, scaler_x_train, scaler_x_test
    
train_x_scaled_data, test_x_scaled_data = standard_scaler(x_train,x_test)



#scale_inverse()
def scale_inverse(train_x_scaled_data, test_x_scaled_data, scaler_x_train, scaler_x_test):
    inverse_scaler_x_train = pd.DataFrame(scaler_x_train.inverse_transform(train_x_scaled_data), columns = x_train.columns.values).set_index([x_train.index.values])
    
    inverse_scaler_x_test = pd.DataFrame(scaler_x_train.inverse_transform(test_x_scaled_data), columns = x_test.columns.values).set_index([x_test.index.values])

    return inverse_scaler_x_train, inverse_scaler_x_test

inverse_scaler_x_train, inverse_scaler_x_test = scale_inverse(train_x_scaled_data, test_x_scaled_data, scaler_x_train, scaler_x_test)
    



#uniform_scaler()
def uniform_scaler(x_train, x_test):
    u_scaler_x_train = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(x_train[['monthly_charges', 'tenure']])
   
    u_train_x_scaled = pd.DataFrame(u_scaler_x_train.transform(x_train), columns = x_train.columns.values).set_index([x_train.index.values])
    u_test_x_scaled = pd.DataFrame(u_scaler_x_train.transform(x_test), columns = x_test.columns.values).set_index([x_test.index.values])
    
    return u_train_x_scaled, u_test_x_scaled

u_train_x_scaled, u_test_x_scaled = uniform_scaler(x_train, x_test)



#gaussian_scaler()
def gaussian_scaler(x_train, x_test):
    g_x_train_scaler =PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(x_train[['monthly_charges', 'tenure']])
    
    g_x_train_scaled = pd.DataFrame(g_x_train_scaler.transform(x_train), columns = x_train.columns.values).set_index([x_train.index.values])
    g_x_test_scaled = pd.DataFrame(g_x_train_scaler.transform(x_test), columns = x_test.columns.values).set_index([x_test.index.values])
    
    return g_x_train_scaled, g_x_test_scaled

g_x_train_scaled, g_x_test_scaled = gaussian_scaler(x_train, x_test)




#min_max_scaler()
def min_max_scaler(x_train, x_test):
    mm_x_train_scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(x_train[['monthly_charges', 'tenure']])
    
    mm_x_train_scaled = pd.DataFrame(mm_x_train_scaler.transform(x_train), columns = x_train.columns.values).set_index([x_train.index.values])
    mm_x_test_scaled = pd.DataFrame(mm_x_train_scaler.transform(x_test), columns = x_test.columns.values).set_index([x_test.index.values])
    return mm_x_train_scaled, mm_x_test_scaled

mm_x_train_scaled, mm_x_test_scaled = min_max_scaler(x_train, x_test)




#iqr_robust_scaler()
def iqr_robust_scaler (x_train, x_test):
    iqr_x_train_scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(x_train[['monthly_charges', 'tenure']])

    iqr_x_train_scaled = pd.DataFrame(iqr_x_train_scaler.transform(x_train), columns = x_train.columns.values).set_index([x_train.index.values])
    iqr_x_test_scaled = pd.DataFrame(iqr_x_train_scaler.transform(x_test), columns = x_test.columns.values).set_index([x_test.index.values])
    
    return iqr_x_train_scaled, iqr_x_test_scaled

iqr_x_train_scaled, iqr_x_test_scaled = iqr_robust_scaler (x_train, x_test)