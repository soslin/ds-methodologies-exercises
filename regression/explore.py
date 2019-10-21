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

split_scale.standard_scaler()
df_train_x = pd.DataFrame(train_x_scaled_data)
df_train_y = pd.DataFrame(train_y_scaled_data)
df_test_x = pd.DataFrame(test_x_scaled_data)
df_test_y  = pd.DataFrame(test_y_scaled_data)

df_train_x['target'] = df_train_y
df_test_x['target'] = df_test_y 
df_train_x.head()
df_test_x.head()

df_train_x.rename(columns={0: "monthly_charges", 1: "tenure"})
df_test_x.rename(columns={0: "monthly_charges", 1: "tenure"})

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

def months_to_years(tenure, df):
    df['tenure_years'] = round(df.tenure//12).astype('category')
    return tenure, df
df_train_x.head(10)



# Write a function, plot_categorical_and_continous_vars(categorical_var, continuous_var, df), that outputs 3 different plots for plotting a categorical variable with a continuous variable, e.g. tenure_years with total_charges. For ideas on effective ways to visualize categorical with continuous: https://datavizcatalogue.com/. You can then look into seaborn and matplotlib documentation for ways to create plots.

