import numpy as np
import pandas as pd
import acquire


def remove_dup_col(df):
    df = df.loc[:,~df.columns.duplicated()]
    return df

def new_df(df):
    num_rows_missing = df.isna().sum()
    pct_rows_missing = num_rows_missing/len(df)*100
    df_sum = pd.DataFrame()
    df_sum['num_rows_missing'] = num_rows_missing
    df_sum['pct_rows_missing'] = pct_rows_missing
    return df_sum

def nulls_by_row(df):
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

# Check if correct
def drop_dups(df):
    df.drop_duplicates('parcelid', keep='last',inplace=True) 
    return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def data_prep(df, cols_to_remove=[], prop_required_column=.6, prop_required_row=.75):
    df.drop(columns = cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


def drop_col(df):
    df = df.drop(columns = ['buildingqualitytypeid', 'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid', 'lotsizesquarefeet', 'propertyzoningdesc', 'regionidcity', 'unitcnt', 'yearbuilt', 'censustractandblock', 'heatingorsystemdesc'])
      
    return df

def impute_values(df):
    sqfeet = df.calculatedfinishedsquarefeet.median()
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.fillna(sqfeet)
    regionzip = df.regionidzip.median()
    df.regionidzip = df.regionidzip.fillna(regionzip)
    structuretaxvalue = df.structuretaxvaluedollarcnt.median()
    df.structuretaxvaluedollarcnt = df.structuretaxvaluedollarcnt.fillna(structuretaxvalue)
    taxvalue = df.taxvaluedollarcnt.median()
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.fillna(taxvalue)
    landtaxvalue = df.landtaxvaluedollarcnt.median()
    df.landtaxvaluedollarcnt = df.landtaxvaluedollarcnt.fillna(landtaxvalue)
    tax = df.taxamount.median()
    df.taxamount = df.taxamount.fillna(tax)
    return df

# def get_upper_outliers(s, k = 1.5):
#     '''
#     Given a series and a cutoff value, k, returns the upper outliers for the series.
#     The values returned will be either 0 (if the point is not an outlier), or a
#     number that indicates how far away from the upper bound the observation is.
#     '''
#     q1 = s.quantile([.25])
#     q3 = s.quantile([.75])
#     iqr = q3 - q1
#     upper_bound = q3 + k * iqr
#     return s.apply(lambda x: max([x - upper_bound, 0]))


# def add_upper_outlier_columns(df, k):
    
#     #Add a column with the suffix _outliers for all the numeric columns in the given dataframe.
    
#     # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k) for col in df.select_dtypes('number')}
#     # return df.assign(**outlier_cols)

#     for col in df.select_dtypes('number'):
#         df[col + '_outliers'] = get_upper_outliers(df[col], k)
#         return df



def print_outliers(df):
    outlier_cols = [col for col in df if col.endswith('_outliers')]
    for col in outlier_cols:
        print('~~~\n' + col)
        data = df[col][df[col] > 0]
        print(data.describe())
    