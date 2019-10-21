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




ols_model = ols('tip ~ total_bill', data=df).fit() #'y ~ x' special code to indicate building a model based on x to predict y
ols_model
df['yhat'] = ols_model.predict(pd.DataFrame(x))
df.head()




# 4. Write a function, plot_residuals(x, y, dataframe) that takes the feature, the target, and the dataframe as input and returns a residual plot. (hint: seaborn has an easy way to do this!)

def plot_residuals(x,y,dataframe):
    g = sns.residplot(x, y, data=df, color='firebrick')
    return g
plot_residuals(x,y,df)





# 5. Write a function, regression_errors(y, yhat), that takes in y and yhat, returns the sum of squared errors (SSE), explained sum of squares (ESS), total sum of squares (TSS), mean squared error (MSE) and root mean squared error (RMSE).

df['residual'] = df['yhat'] - df['tip']
df.head()

yhat = df['yhat']

df['residual^2'] = df.residual ** 2
df.head()

def regression_errors(y,yhat):
    SSE = sum((df['yhat'] - df['tip'])**2)
    ESS = sum((df.yhat - df['tip'].mean())**2) #ESS (Explained Sum of Squares) is the difference between the predicted and the mean.
    TSS = SSE + ESS #TSS (Total Sum of Squares) is the difference between the actual and the mean. Also the total of ESS and SSE.
    MSE = SSE/len(df) 
    RMSE = sqrt(mean_squared_error(df['tip'], df.yhat))
    return (SSE, ESS, TSS, MSE, RMSE)
regression_errors(df['tip'], df['yhat'])





#6. Write a function, baseline_mean_errors(y), that takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).

def baseline_mean_errors(y):
    mean_y = df["tip"].mean()
    SSE_baseline = sum((mean_y - df['tip'])**2)
    MSE_baseline = SSE_baseline/len(df)
    RMSE_baseline = sqrt(MSE_baseline)
    return SSE_baseline, MSE_baseline, RMSE_baseline
baseline_mean_errors(df['tip'])


#calculating R2 - 2 ways
#1
SSE = sum((df['yhat'] - df['tip'])**2)
ESS = sum((df.yhat - df['tip'].mean())**2)
TSS = SSE + ESS 
MSE = SSE/len(df) 
RMSE = sqrt(mean_squared_error(df['tip'], df.yhat))

R2 = ESS/TSS
print('R-squared = ',round(R2,3))
print("Percent of variance in y explained by x = ", round(R2*100,1), "%")

#2
r2 = ols_model.rsquared
print('R-squared = ', round(r2,3))

#F statistic/p-value
f_pval = ols_model.f_pvalue
print("p-value for model significance = ", round(f_pval,8))





#7. Write a function, better_than_baseline(SSE), that returns true if your model performs better than the baseline, otherwise false.

def better_than_baseline(MSE, MSE_baseline):
    MSE_baseline = mean_squared_error(df_baseline.y, df_baseline.yhat)
    if MSE < MSE_baseline:
        return MSE < MSE_baseline
better_than_baseline(MSE, MSE_baseline)
print("model is better?", MSE < MSE_baseline)

MSE_baseline = mean_squared_error(df_baseline['tip'], df_baseline.yhat)


#8. Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model (r^2), and the value telling you whether the correlation between the model and the tip value are statistically significant (F stat, value).

def model_significance(ols_model):
    r2 = ESS/TSS
    print('R-squared = ',round(r2,3))
    print("Percent of variance in y explained by x = ", round(R2*100,1), "%")
    return model_significance
model_significance(ols_model)

r2 = ols_model.rsquared
print('r-squared = ', round(r2,3))

    #F statistic/p-value
f_pval = ols_model.f_pvalue

print("p-value for model significance = ", f_pval)