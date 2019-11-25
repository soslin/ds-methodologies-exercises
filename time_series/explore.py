


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math
from pprint import pprint
import requests

from datetime import datetime
import itertools
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm


# In[2]:


store = pd.read_csv('swo_grocery.csv')


# In[3]:


store['sale_date'] = pd.to_datetime(store['sale_date'])
store.head()


# In[4]:


store=store.set_index('sale_date')
store.head()


# Split your data into train and test using the sklearn.model_selection.TimeSeriesSplit method.

# In[5]:


type(store.sale_amount)


# In[ ]:


store2 = store.resample('D').sum().reset_index()


# In[9]:


store2.head()


# In[10]:


X = store2.sale_date
y = store2.sale_amount


# In[21]:


X


# In[22]:


tss = TimeSeriesSplit(n_splits=5, max_train_size=None)
for train_index, test_index in tss.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[26]:


print(X_train.tail())
print(X_test.head())


# Validate your splits by plotting X_train and y_train.

# In[27]:


plt.plot(X_train, y_train)
plt.plot(X_test, y_test)


# In[14]:


X_train = pd.DataFrame(X_train)


# In[28]:


aggregation = 'sum'

train = store[:'2017-03-02 00:00:00+00:00'].sale_amount.resample('D').agg(aggregation)
test = store['2017-03-03 00:00:00+00:00':].sale_amount.resample('D').agg(aggregation)

print('Observations: %d' % (len(train.values) + len(test.values)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))


# In[ ]:





# In[32]:


train.plot()


# Plot the weekly average & the 7-day moving average. Compare the 2 plots.

# In[31]:


train.resample('W').mean().plot(figsize=(12, 4))
plt.show()


# In[35]:


train.rolling(7).mean().plot(figsize= (12,4))


# Plot the daily difference. Observe whether usage seems to vary drastically from day to day or has more of a smooth transition

# In[38]:


train.resample('D').mean().diff(periods = 1).plot(figsize = (12,4))


# In[42]:


train.diff(periods = 7).plot(figsize = (12,4), linewidth = .5)
plt.show()


# Plot a time series decomposition.

# In[46]:


decomposition = sm.tsa.seasonal_decompose(train.resample('W'). mean(), model = 'additive')
fig = decomposition.plot()
plt.show


# In[ ]:





# Create a lag plot (day over day).

# In[48]:


pd.plotting.lag_plot(train.resample('W').mean(), lag = 1)


# In[ ]:





# Run a lag correlation.

# In[51]:


df_corr = pd.concat([train.shift(1), train], axis = 1)
df_corr.columns = ['t-1', 't+1']
result = df_corr.corr()
print(result)


# Split your data into train and test using the percent cutoff method.

# In[ ]:


power = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')


# In[ ]:


power.info()


# In[ ]:


power['Date'] = pd.to_datetime(power['Date'])
power.head()


# In[ ]:


power=power.set_index('Date')
power.head()


# In[ ]:


ts_data = power.resample('D').agg(sum).reset_index()
ts_data.set_index('Date')


# In[ ]:


X = ts_data.drop('Consumption', axis = 1)
y = ts_data.Consumption


# In[ ]:


X.set_index('Date')


# In[ ]:


X.head()


# In[ ]:


y.head()


# In[ ]:


train_size = int(len(ts_data) * 0.66)
train1, test1 = ts_data[0:train_size], ts_data[train_size:len(ts_data)]
print('Observations: %d' % (len(ts_data)))
print('Training Observations: %d' % (len(train1)))
print('Testing Observations: %d' % (len(test1)))


# In[ ]:





# Validate your splits by plotting X_train and y_train.

# In[65]:


plt.figure(figsize=(8, 4))
train1.plot()
test1.plot()
plt.show()


# Plot the weekly average & the 7-day moving average. Compare the 2 plots.
# 
# 
# Plot a time series decomposition. Takeaways?

# In[66]:


train1.resample('W').mean().plot(figsize=(12, 4))
plt.show()


# In[68]:


train.resample('W').mean().diff(periods=7).plot(figsize=(12,4))


# Plot the daily difference. Observe whether usage seems to vary drastically from day to day or has more of a smooth transition.

# In[70]:


train1.resample('D').mean().plot(figsize=(12, 4))
plt.show()


# Group the electricity consumption time series by month of year, to explore annual seasonality.

# In[67]:


train1.resample('M').mean().plot(figsize=(12, 4))
plt.show()


# In[71]:


decomposition = sm.tsa.seasonal_decompose(train1.resample('W').mean(), model='additive')

fig = decomposition.plot()
plt.show()


# In[ ]:




