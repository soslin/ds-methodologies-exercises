# As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.

# Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

import pandas as pd
from env import host, user, password

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
df = pd.read_sql('SELECT customer_id, monthly_charges, tenure, total_charges FROM customers', url)

# Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.
df.head()
df.tail(10)
df.describe()
df.shape
df.info() #total_charges are an object, but need to be float
df.isnull().sum() # can't eval whether there are nulls because of total_charges is the wrong type
df.total_charges.value_counts()
df.replace(r'^\s*$', np.nan, regex=True, inplace=True) #replace missing values with nan
df.total_charges.value_counts()
df = df.dropna() #drop non-numbers
df.info()
df.total_charges = df.total_charges.astype('float') #save series as 
df.info()

plt.figure(figsize = (16,3))
for i, col in enumerate(['monthly_charges', 'tenure', 'total_charges']):
    plot_number = i + 1 # i starts at 0, but plot nos should start at 1
    series = df[col]
    plt.subplot(1,3,plot_number)
    plt.title(col)
    series.hist(bins=5)

plt.figure(figsize = (8,4))
sns.boxplot(data = df.drop(columns = ['customer_id', 'total_charges']))

