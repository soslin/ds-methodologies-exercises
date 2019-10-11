import pandas as pd
from env import host, user, password
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

from pydataset import data
df = data('tips')

# 2. Fit a linear regression model (ordinary least squares) and compute yhat, predictions of tip using total_bill. You may follow these steps to do that:

# import the method from statsmodels: from statsmodels.formula.api import ols
# fit the model to your data, where x = total_bill and y = tip: regr = ols('y ~ x', data=df).fit()
# compute yhat, the predictions of tip using total_bill: df['yhat'] = regr.predict(df.x)

#Descriptive
df.head()
df.columns.values
df.shape
df.describe()
df.info()
print(df.isnull().sum())
df.total_bill.value_counts(ascending = True)
df.tip.value_counts(ascending = True)

#Variables
x = df.total_bill 
y = df.tip


ols_model = ols('y ~ x', data=df).fit()
df['yhat'] = ols_model.predict(pd.DataFrame(x))
df.head()




# 4. Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way to do this!)

