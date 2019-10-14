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
    SSE = (df['yhat'] - df['tip'])**2
    ESS = sum((df.yhat - df['tip'].mean())**2) #ESS (Explained Sum of Squares) is the difference between the predicted and the mean.
    TSS = SSE + ESS #TSS (Total Sum of Squares) is the difference between the actual and the mean. Also the total of ESS and SSE.
    MSE = SSE/len(df) 
    RMSE = sqrt(mean_squared_error(df['tip'], df.yhat))
    return (SSE, ESS, TSS, MSE, RMSE) #How do i run this?





#6. Write a function, baseline_mean_errors(y), that takes in your target, y, computes the SSE, MSE & RMSE when yhat is equal to the mean of all y, and returns the error values (SSE, MSE, and RMSE).

def baseline_mean_errors(y):

ss = pd.DataFrame(np.array(['SSE','ESS','TSS']), columns=['metric'])
ss['model_values'] = np.array([SSE, ESS, TSS])

df_baseline = df[['total_bill','tip']]
df_baseline['yhat'] = df_baseline['tip'].mean()

df_eval = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
df_eval['model_error'] = np.array([SSE, MSE, RMSE])
df_eval

df_baseline['residual'] = df_baseline['yhat'] - df_baseline['tip']
# square that delta
df_baseline['residual^2'] = df_baseline['residual'] ** 2

df_eval['baseline_error'] = np.array([SSE, MSE, RMSE])
df_eval['error_delta'] = df_eval.model_error - df_eval.baseline_error
df_eval

ESS_baseline = sum((df.yhat - df['tip'].mean())**2)
SSE_baseline = SSE['baseline_error'] #why is this erring?
TSS_baseline = ESS_baseline + SSE_baseline
MSE_baseline = SSE['baseline_error']/len(df)
RMSE_baseline = sqrt(MSE_baseline)



#calculating R2 - 2 ways
#1
SSE = (df['yhat'] - df['tip'])**2
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

def better_than_baseline(SSE, SSE_baseline):
    MSE_baseline = mean_squared_error(df_baseline.y, df_baseline.yhat)
    if MSE < MSE_baseline:
        return True
MSE_baseline = mean_squared_error(df_baseline['tip'], df_baseline.yhat)
print("model is better?", MSE < MSE_baseline)


#8. Write a function, model_significance(ols_model), that takes the ols model as input and returns the amount of variance explained in your model (r^2), and the value telling you whether the correlation between the model and the tip value are statistically significant (F stat, value).




def model_significance(ols_model):

r2 = ESS/TSS
print('R-squared = ',round(r2,3))
print("Percent of variance in y explained by x = ", round(R2*100,1), "%")

r2 = ols_model.rsquared
print('r-squared = ', round(r2,3))

#F statistic/p-value
f_pval = ols_model.f_pvalue

print("p-value for model significance = ", round(f_pval,8))