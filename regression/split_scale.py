# Each scaler function should create the object, fit and transform both train and test. They should return the scaler, train dataframe scaled, test dataframe scaled. Be sure your indices represent the original indices from train/test, as those represent the indices from the original dataframe.

import pandas as pd
from env import host, user, password

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import wrangle

df = wrangle.wrangle_telco()
df

tlc_chrn_minus_total_charges = df.drop(columns=['total_charges'])
tlc_chrn_only_total_charges = df.total_charges

#1. split_my_data(X, y, train_pct)
x_train, x_test, y_train, y_test = train_test_split(tlc_chrn_minus_total_charges, tlc_chrn_only_total_charges, train_size = .80, random_state = 123)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

plt.figure(figsize = (16,3)) #histogram train X data
for i, col in enumerate(['monthly_charges', 'tenure']):
    plot_number = i + 1 # i starts at 0, but plot nos should start at 1
    series = x_train[col]
    plt.subplot(1,3,plot_number)
    plt.title(col)
    series.hist(bins=5)

plt.figure(figsize = (16,3)) #histogram test X data
for i, col in enumerate(['monthly_charges', 'tenure']):
    plot_number = i + 1 # i starts at 0, but plot nos should start at 1
    series = x_test[col]
    plt.subplot(1,3,plot_number)
    plt.title(col)
    series.hist(bins=5)


from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#2. standard_scaler()
x_train.info()
scaler_x_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(x_train[['monthly_charges', 'tenure']])
scaler_x_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(x_test[['monthly_charges', 'tenure']])
scaler_y_train = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(pd.DataFrame(y_train))
scaler_y_test = StandardScaler(copy=True, with_mean=True, with_std=True)\
            .fit(pd.DataFrame(y_test))


scaler.transform

train_scaled_data = scaler.transform(train)
test_scaled_data = scaler.transform(test)


scale_inverse()


uniform_scaler()



gaussian_scaler()


min_max_scaler()




iqr_robust_scaler()