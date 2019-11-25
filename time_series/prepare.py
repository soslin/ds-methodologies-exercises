


from vega_datasets import data
import numpy as np
df = data.sf_temps()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math
from pprint import pprint
import requests


# Convert date column to datetime format.

# In[2]:


grocery = pd.read_csv('oslin_merge_total.csv')


# In[3]:


grocery.sale_date.dtype


# In[4]:


grocery.sale_date.head()


# In[5]:


grocery['sale_date'] = pd.to_datetime(grocery['sale_date'])
grocery.head()


# In[ ]:





# Plot the distribution of sale_amount, item_price and sale_date.

# In[6]:


plt.hist(grocery.sale_amount)
plt.show()


# In[7]:


plt.hist(grocery.item_price)
plt.show()


# In[8]:


plt.hist(grocery.index)
plt.show()


# Set the index to be the datetime variable.

# In[9]:


grocery = grocery.sort_values('sale_date').set_index('sale_date')
grocery.head()


# Add a 'month' and 'day of week' column to your dataframe, derived from the index using the keywords for those date parts.

# In[10]:


grocery['month'] = grocery.index.month
grocery['Weekday Name'] = grocery.index.weekday_name
grocery.head()


# Add a column to your dataframe, sales_total, which is a derived from sale_amount (total items) and item_price.

# In[11]:


grocery['sales_total'] = grocery.sale_amount * grocery.item_price
grocery.head()


# Using pandas.DataFrame.diff() function, create a new column that is the result of the current sales - the previous days sales.

# In[26]:


grocery['diff_from_last_day'] = grocery.sales_total.diff()
grocery.head()


# In[27]:


grocery.head()


# In[25]:


grocery.to_csv('swo_grocery.csv')


# Make sure all the work that you have done above is reproducible. That is, you should put the code above into separate functions and be able to re-run the functions and get the same results.

# In[ ]:





# In[ ]:





# In[14]:


power = pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')


# Convert date column to datetime format.

# In[15]:


power.head()


# In[16]:


power['Date'] = pd.to_datetime(power['Date'])


# In[ ]:





# Plot the distribution of each of your variables.

# In[17]:


plt.hist(power.Consumption)
plt.show()


# In[18]:


plt.hist(power.Wind)
plt.show()


# In[19]:


plt.hist(power.Solar)
plt.show()


# In[20]:


plt.hist(power['Wind+Solar'])
plt.show()


# Set the index to be the datetime variable.

# In[21]:


power = power.sort_values('Date').set_index('Date')
power.head()


# Add a month and a year column to your dataframe.

# In[22]:


power['month'] = power.index.month
power['year'] = power.index.year
power.head()


# In[23]:


power.to_csv('power.csv')


# Make sure all the work that you have done above is reproducible. That is, you should put the code above into separate functions and be able to re-run the functions and get the same results.

# In[ ]:




