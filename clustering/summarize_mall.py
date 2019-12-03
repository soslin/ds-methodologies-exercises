import numpy as np
import pandas as pd
from env import host, user, password
import acquire

df3 = acquire.acquire_mallcustomer_data()

def summarize(df3):
    df3.head()
    df3.tail(5)
    df3.sample(5)
    df3.describe()
    df3.shape
    df3.isnull().sum()
    df3.info()
    return df3
summarize_mall = summarize(df3)