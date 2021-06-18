# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Exploring
import scipy.stats as stats

# Visualizing
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format

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
    print(type(num_missing))
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

def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_column * len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def wrangle_zillow(df):
    
    # filter by propertylandusetypeid
    df = df[df.propertylandusetypeid.isin([261, 262, 263, 264, 266, 268, 273, 276])]
    
    # going to remove properties with no bathrooms, no bedrooms, less than 500 sqft
    df = df[(df.bathroomcnt > 0) & (df.bedroomcnt > 0) & (df.calculatedfinishedsquarefeet > 500)]
    
    # drop columns and rows with missing values
    df = handle_missing_values(df)
    
    # drop redundant and unneeded columns
    df = df.drop(columns=['propertylandusetypeid', 'heatingorsystemtypeid', 'id', 'calculatedbathnbr', 'finishedsquarefeet12',\
                      'fullbathcnt', 'propertycountylandusecode', 'propertyzoningdesc', 'censustractandblock'])
    
    return df