# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns

import os
from env import host, username, password

##### DB CONNECTION #####
def get_db_url(db, username=username, host=host, password=password):
    
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'

##### ACQUIRE ZILLOW #####
def new_zillow_data():
    '''
    gets zillow information from CodeUp db using SQL query
    and creates a dataframe
    '''

    # SQL query
    zillow_query = '''SELECT *
                        FROM properties_2017 prop
                        INNER JOIN (SELECT parcelid, logerror, max(transactiondate) transactiondate
                                    FROM predictions_2017
                                    GROUP BY parcelid, logerror) pred USING (parcelid)
                        LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                        LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                        LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                        LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                        LEFT JOIN propertylandusetype land USING (propertylandusetypeid)
                        LEFT JOIN storytype story USING (storytypeid)
                        LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                        LEFT JOIN unique_properties special USING (parcelid)
                        WHERE prop.latitude IS NOT NULL 
                        AND prop.longitude IS NOT NULL
                        AND transactiondate LIKE '2017%'
                    '''
    
    # reads SQL query into a DataFrame            
    df = pd.read_sql(zillow_query, get_db_url('zillow'))
    
    return df

def get_zillow_data():
    '''
    checks for existing csv file
    loads csv file if present
    if there is no csv file, calls new_zillow_data
    '''
    
    if os.path.isfile('zillow.csv'):
        
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        df = new_zillow_data()
        
        df.to_csv('zillow.csv')
    
    return df

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    cols_missing = cols_missing.sort_values(by='percent_rows_missing', ascending=False)
    return cols_missing

def cols_missing(df):
    num_missing = df.isnull().sum(axis=1)
    percent_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_missing})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    rows_missing = rows_missing.sort_values(by='percent_cols_missing', ascending=False)
    return rows_missing

def remove_outliers(df, col_list, k=1.5):
    for col in col_list:
        
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
    
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
    
        df = df[df[col] > lower_bound]
        df = df[df[col] < upper_bound]
    
    return df

def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def impute(df, method, col_list):
    imputer = SimpleImputer(strategy=method)
    df[col_list] = imputer.fit_transform(df[col_list])
    return df

def split_zillow(df, target):
    '''
    this function takes in the zillow dataframe
    splits into train, validate and test subsets
    then splits for X (features) and y (target)
    '''
    
    # split df into 20% test, 80% train_validate
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    
    # split train_validate into 30% validate, 70% train
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    
    # Split with X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

def prepare_zillow(df):
    
    # filter by propertylandusetypeid
    df = df[df.propertylandusetypeid.isin([261, 262, 263, 264, 266, 268, 273, 276])]
    
    # remove outliers
    df = remove_outliers(df, ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet'], k=1.5)
 
    # drop columns and rows with missing values
    df = handle_missing_values(df)
    
    # drop redundant and unneeded columns
    df = df.drop(columns=['propertylandusetypeid', 'heatingorsystemtypeid', 'id', \
                          'calculatedbathnbr',  'finishedsquarefeet12', 'fullbathcnt', \
                          'propertycountylandusecode', 'propertyzoningdesc', 'censustractandblock'])
    
    # going to drop buildingqualitytypeid and heatingorsystemdesc - changing the value would cause huge impact
    df = df.drop(columns=['buildingqualitytypeid', 'heatingorsystemdesc'])

    # impute list to replace with 'median' values
    df = impute(df, 'median', ['lotsizesquarefeet', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', \
                               'landtaxvaluedollarcnt', 'taxamount'])
    
    # impute list to replace with 'most_frequent' values
    df = impute(df, 'most_frequent', ['regionidcity', 'regionidzip', 'yearbuilt'])
    
    return df