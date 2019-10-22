from env import host, user, password
import pandas as pd

def get_titanic_data():
    url = f'mysql+pymysql://{user}:{password}@{host}/titanic_db'
    df_t = pd.read_sql('''SELECT *
    FROM passengers;''', url)
    return df_t

