import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

import env
import wrangle
import split_scale


# 1. Write a function, plot_variable_pairs(dataframe) that plots all of the pairwise relationships along with the regression line for each pair.


df_train_x = pd.DataFrame(train_x_scaled_data)
df_train_y = pd.DataFrame(train_y_scaled_data)
df_test_x = pd.DataFrame(test_x_scaled_data)
df_test_y  = pd.DataFrame(test_y_scaled_data)

df_train_x['target'] = train_y_scaled_data

g = sns.PairGrid(df_train_x)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)

df_test_x['target'] = test_y_scaled_data

g = sns.PairGrid(df_test_x)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)

def plot_variable_pairs(df_train_x):
    g = sns.PairGrid(df_train_x)
    return  g.map_diag(plt.hist), g.map_offdiag(plt.scatter)
plot_variable_pairs(df_train_x)
     

# Write a function, months_to_years(tenure_months, df) that returns your dataframe with a new feature tenure_years, in complete years as a customer.

def months_to_years(df):
    df['tenure_years'] = round(df.tenure//12)/astype('category')