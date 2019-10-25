from env import host, user, password
import pandas as pd

def get_titanic_data():
    url = f'mysql+pymysql://{user}:{password}@{host}/titanic_db'
    df_t = pd.read_sql('''SELECT *
    FROM passengers;''', url)
    return df_t

def get_iris_data():
    url = f'mysql+pymysql://{user}:{password}@{host}/iris_db'
    df_i = pd.read_sql('''SELECT * FROM measurements as m
    JOIN species as s ON s.species_id = m.species_id;''', url)
    return df_i

df_i = acquire.get_iris_data()
def prep_iris(df_i):
    df_i.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    df_i.rename(columns = {'species_name' : 'species'}, inplace = True)
    int_encoder = LabelEncoder()
    int_encoder.fit(df_i.species)
    df_i.species = int_encoder.transform(df_i.species)
    species_array = np.array(df_i.species)
    return df_i
prep_iris(df_i)

def titanic_prep():
    titanic_df = pd.read_csv('https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/export?format=csv&gid=341089357')
    titanic_df.drop(columns = ['deck'], inplace = True)
    titanic_df.fillna(np.nan, inplace=True)
    train, test = train_test_split(df_t, train_size=.8, random_state=123)
    imp_mode = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    imp_mode.fit(df_titanic[['embarked', 'embark_town']])
    df_titanic[['embarked', 'embark_town']] = imp_mode.transform(df_titanic[['embarked', 'embark_town']])
    
    embarked_array = embarked_array.reshape(len(embarked_array), 1)
    return titanic_prep