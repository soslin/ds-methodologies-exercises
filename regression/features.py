# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

# ## 1. Write a function, select_kbest_freg() that takes X_train, y_train and k as input (X_train and y_train should not be scaled!) and returns a list of the top k features.


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
from sklearn.feature_selection import SelectKBest, f_regression



df = wrangle.wrangle_telco()
X = df.drop(columns=['total_charges', 'customer_id'])
y = pd.DataFrame(df.total_charges)


x_train,x_test,y_train,y_test=split_scale.split_my_data(X,y)
#X_train


def select_kbest_freg(x_train, y_train, k):

    f_selector = SelectKBest(f_regression, k)

    f_selector.fit(x_train, y_train)

    f_support = f_selector.get_support()
    f_feature = x_train.loc[:,f_support].columns.tolist()
    
    return f_feature


select_kbest_freg(x_train,y_train, 2)
    #print(str(len(f_feature)), 'selected features')
    #print(f_feature)
  


plt.figure(figsize=(6,5))
cor = x_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()



# ## Write a function, select_kbest_freg() that takes X_train, y_train (scaled) and k as input and returns a list of the top k features.

train_x_scaled_data, test_x_scaled_data,scaler_x_train, scaler_x_test = split_scale.standard_scaler(x_train,x_test)




def select_kbest_freg(train_x_scaled_data, k):
    f_selector = SelectKBest(f_regression, k)

    f_selector.fit(x_train, y_train)

    f_support = f_selector.get_support()
    f_feature = x_train.loc[:,f_support].columns.tolist()
    
    return f_feature


select_kbest_freg(train_x_scaled_data, 2)


plt.figure(figsize=(6,5))
cor = train_x_scaled_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()



# ## Write a function, ols_backware_elimination() that takes X_train and y_train (scaled) as input and returns selected features based on the ols backwards elimination method.


import statsmodels.api as sm

def ols_backware_elimination(train_x_scaled_data):
    cols = list(train_x_scaled_data.columns)
    pmax = 1
    while (len(cols)>0):
        p= []
        X_1 = train_x_scaled_data[cols]
        model = sm.OLS(y_train,X_1).fit()
        p = pd.Series(model.pvalues.values[:],index = cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    return selected_features_BE


ols_backware_elimination (train_x_scaled_data)






# ## Write a function, lasso_cv_coef() that takes X_train and y_train as input and returns the coefficients for each feature, along with a plot of the features and their weights.

from sklearn.linear_model import LassoCV



reg = LassoCV()
def lasso_cv_coef(train_x_scaled_data):
    reg.fit(train_x_scaled_data, y_train)
    coef = pd.Series(reg.coef_, index = train_x_scaled_data.columns)
    return coef
df=lasso_cv_coef (train_x_scaled_data)
df = pd.DataFrame(df).reset_index()
df.rename(columns = {'index': 'variable', 0 : 'coef'}, inplace = True)
df.head()


plt.figure(figsize=(6,5))
sns.barplot(x = 'variable', y = 'coef', data = df)
plt.show()


#%%


# ## Write 3 functions, the first computes the number of optimum features (n) using rfe, the second takes n as input and returns the top n features, and the third takes the list of the top n features as input and returns a new X_train and X_test dataframe with those top features.

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def num_optimum_features(x_train, y_train):
    number_of_features_list=np.arange(1,3)
    high_score=0
    number_of_features=0           
    score_list =[]
    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        x_train_rfe = rfe.fit_transform(x_train, y_train)
        x_test_rfe = rfe.transform(x_test)
        model.fit(x_train_rfe,y_train)
        score = model.score(x_test_rfe, y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features
number_of_features = num_optimum_features (x_train, y_train)

print("Optimum number of features: %d" %number_of_features)


def top_features(x_train, y_train):
    cols = list(x_train.columns)
    model = LinearRegression()

    #Initializing RFE model
    rfe = RFE(model, 2)

    #Transforming data using RFE
    x_rfe = rfe.fit_transform(x_train,y_train)  

    #Fitting the data to model
    model.fit(x_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    return selected_features_rfe

selected_features_rfe = top_features(x_train, y_train)

print(selected_features_rfe)


def new_dfs(x_train, x_test, selected_features_rfe):
    x_train = x_train['monthly_charges', 'tenure']
    x_test = x_test['monthly_charges', 'tenure']
    return x_train, x_test
print(x_train)


