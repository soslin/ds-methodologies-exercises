from vega_datasets import data
import numpy as np
df = data.sf_temps()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math


# Resample by the day and take the average temperature. Visualize the average temperature over time.

df = df.set_index('date')
df.describe()
df.isnull().sum()
df.asfreq('D')
df.resample('D').mean()


# Write the code necessary to visualize the minimum temperature over time.
df.plot()
plt.show()


df.resample("D").min().plot()
plt.show()


# Write the code necessary to visualize the maximum temperature over time.

df.resample("D").max()

df.resample("D").max().plot()
plt.show()


# Which month is the coldest, on average?

df.resample("M").min().sort_values(by = 'temp').head(1)


df.resample("M").min().plot()
plt.show()


# #### Which month has the highest average temperature?

df.resample("M").mean().sort_values(by = 'temp').head(1)

df.resample("M").mean().plot()
plt.show()


# #### Resample by the day and calculate the min and max temp for the day (Hint: .agg(['min', 'max'])). Use this resampled dataframe to calculate the change in temperature for the day. Which month has the highest daily temperature variability?

min_max_temps = df.resample('D').agg(['min', 'max'])
min_max_temps.head()

temp_diff = min_max_temps.temp['max'] - min_max_temps.temp['min']
# tri_montly_cofee_consumption = df.coffee_consumption.resample('3M').agg(['min', 'max'])
# tri_montly_cofee_consumption['max'] - tri_montly_cofee_consumption['min']


min_max_temps.plot()
plt.show()


# #### Bonus: Visualize the daily min, average, and max temperature over time on a single line plot, i.e. the min, average, and maximum temperature should be 3 seperate lines.

min_avg_max_temps = df.temp.resample('D').agg(['min', 'mean', 'max'])
min_avg_max_temps


min_avg_max_temps.plot()
plt.show()

from vega_datasets import data
df = data.seattle_weather()
df.head()


df.index


# #### Which year and month combination has the highest amount of precipitation?

df.groupby([df.date.dt.year, df.date.dt.month]).sum().precipitation.idxmax()


# #### Visualize the amount of monthly precipitation over time.

df['date'] = pd.to_datetime(df['date'])


df = df.sort_values('date').set_index('date')
df.head(1)


df.precipitation.resample("M").sum().plot()
plt.show()


# #### Visualize the amount of wind over time. Choose a time interval you think is appropriate.

# In[26]:


df.wind.resample("M").sum().plot()
plt.show()


# #### Which year-month combination is the windiest?

# In[27]:


df.groupby([df.index.year, df.index.month]).sum().wind.idxmax()


# #### What's the sunniest year? (Hint: which day has the highest number of days where weather == sun?)

# In[28]:


df2 = df.assign(sunny = df.weather == 'sun')
df2.info()


# In[29]:


df2.resample('Y').sunny.sum()


# #### In which month does it rain the most?

# In[30]:


df3 = df.assign(rainy = df.weather == 'rain')
df3.head()


# In[31]:


df3.resample('Y').rainy.sum()


# #### Which month has the most number of days with a non-zero amount of precipitation?

# In[32]:


df3.weather.value_counts()


# In[33]:


df3['precipitation'] = np.where((df3.weather == 'snow') | (df3.weather == 'rain') | (df3.weather == 'drizzle') ,1,0)


# In[34]:


df3.resample('M').precipitation.sum().max()


# In[35]:


df3.groupby([df3.index.year, df3.index.month]).sum().precipitation.idxmax()


# In[36]:


df3.precipitation.resample('M').sum().plot()
plt.show()


# In[ ]:





# In[37]:


df4 = data.flights_20k()


# Convert any negative delays to 0.

# In[38]:


df4['non_neg_delay'] = np.where((df4.delay < 0), 0, df4.delay)


# Which hour of the day has the highest average delay?

# In[39]:


df4['date'] = pd.to_datetime(df4['date'])


# In[40]:


df4 = df4.sort_values('date').set_index('date')


# In[ ]:





# In[41]:


df4['hour'] = df4.index.hour
df4.head()


# In[42]:


df4.groupby('hour').delay.mean().sort_values(ascending = False)


# Does the day of the week make a difference in the delay amount?

# In[43]:


df4['day-of-week'] = df4.index.strftime('%w' '%a')
df4.head()


# In[44]:


df4.groupby('day-of-week').delay.mean().sort_values(ascending = False)


# In[ ]:





# In[ ]:





# Does the month make a difference in the delay amount?

# In[45]:


df4['month'] = df4.index.strftime('%m %b')
df4.head()


# In[46]:


df4.groupby('month').delay.mean().plot.bar(figsize=(9, 5), color='red', edgecolor='black')

mu = df4.delay.mean()
se = 2.58 * (df4.delay.std()/math.sqrt(df4.shape[0]))
ub, lb = mu + se, mu - se

plt.hlines(mu, -.5, 2.5, ls = '--')
plt.hlines([lb, ub], -.5, 2.5,ls =':')
plt.xticks(rotation = 0)

plt.show()


# In[47]:


from vega_datasets import data
iowa = data.iowa_electricity()
iowa.head()


# In[48]:


iowa.shape


# In[49]:


iowa.year.value_counts()


# In[50]:


sns.lineplot(data=iowa.reset_index(), x='year', y='net_generation', hue='source')
plt.show()


# In[51]:


iowa.pivot_table('net_generation', 'year', 'source')


# In[52]:


iowa.pivot_table('net_generation', 'year', 'source').plot()


# For each row, calculate the percentage of the year's total that energy source provided.

# In[53]:


iowa = iowa.sort_values('year').set_index('year')
iowa.head()


# In[54]:


iowa['just_year'] = iowa.index.strftime('%Y')
iowa.head()


# In[55]:


iowa.groupby(['just_year', 'source']).mean()/iowa.net_generation.sum()


# In[56]:


iowa['yearly_total'] = iowa.groupby('year').net_generation.transform('sum')
iowa.head()


# In[57]:


iowa.resample('Y').sum().net_generation.plot()
plt.show()


# Lineplot of generation over time, color by source

# In[58]:


iowa.groupby('source').resample('Y').sum().net_generation.plot()
plt.show()


# Display the data as table where years are columns, and energy source is rows (Hint: df.pivot_table)

# In[59]:


iowa.pivot_table('net_generation', 'year', 'source')


# Make a line plot that shows the amount of energy generated over time. Each source should be a separate line?
# Is the total generation increasing over time?

# In[60]:


iowa.pivot_table('net_generation', 'year', 'source').plot.area(figsize=(12, 8))


# 
# Create pretty labels for time plots
# Visualize the number of days of each month that fall into each bin by year (e.g. x=month, y=n_days, hue=temp_bin) or st similar

# In[61]:


from vega_datasets import data
sf = data.sf_temps()
sf.info()


# Create 4 categories for temperature, cold, cool, warm, hot (hint: use pd.cut or pd.qcut for this)

# In[62]:


temp2 = pd.cut(sf.temp, 4, labels=["cold", "cool", "warm", 'hot'])


# In[63]:


sf['temp2'] = pd.DataFrame(temp2)
sf.head()


# In[64]:


sf.temp2.value_counts()


# How does the occurances of these 4 categories change month over month? i.e. how many days have each distinction? 

# In[65]:


sf.head()


# In[66]:


sf = sf.sort_values('date').set_index('date')
sf.head()


# In[67]:


sf['month'] = sf.index.strftime('%m %b')
sf.head()


# In[69]:


sf.groupby(['month','temp2']).count()


# Visualize this and give the visual appropriate colors for each category.

# In[75]:


sf.pivot_table('temp', 'month', 'temp2')


# In[77]:


sf.pivot_table('temp', 'month', 'temp2').plot()


# In[ ]:




