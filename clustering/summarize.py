import numpy as np
import pandas as pd
from env import host, user, password
from acquire import df


def stats(df):
    print(df.head())
    print(df.tail())
    print(df.sample(10))
    print(df.shape)
    print(df.describe())
    print(df.info())
    print(df.isnull())
stats(df)