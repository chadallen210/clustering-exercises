##### IMPORTS #####
import pandas as pd
import os

from env import host, username, password

def get_db_url(db, username=username, host=host, password=password):
    
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    
def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_db_url('mall_customers'))
    return df.set_index('customer_id')

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    print(type(num_missing))
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    return cols_missing
    
def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    percent_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': percent_missing})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing

def summarize(df):
    '''
    summarize 
    '''
    print('==============================================')
    print('DataFrame head: ')
    print(df.head(3))
    print('==============================================')
    print('DataFrame info: ')
    print(df.info())
    print('==============================================')
    print('DataFrame description: ')
    print(df.describe())
    num_col = [col for col in df.columns if df[col].dtype!='O']
    cat_col = [col for col in df.columns if col not in num_col]
    print('==============================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_col:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('==============================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('==============================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('==============================================')
    df.hist()
    plt.tight_layout()
    
    return plt.show()

def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column=0.5, prop_required_row=0.75):
    threshold = int(round(prop_required_column * len(df.index),0))
    df =  df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns),0))
    df =  df.dropna(axis=0, thresh=threshold)
    return df

def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df

def get_lower_outliers(s, k=1.5):
    q1, q3 = s.quartile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - lower_bound, 0]))

def add_lower_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_lower_outliers'] = get_upper_outliers(df[col], k)
    return df

def remove_outliers(df, col_list, k=1.5):
    for col in col_list:
        
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
    
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
    
        df = df[df[col] > lower_bound]
        df = df[df[col] < upper_bound]
        
        outlier_cols = [col for col in df.columns if col.endswith('_outliers')]
        df = df.drop(columns=outlier_cols)
    
    return df

def mall_split(df):
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=1234)
    
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=1234)
    
    return train, validate, test

def mall_encoder(df, col):
    
    df = pd.get_dummies(df, columns=col, drop_first=True)

    return df

def min_max_scale(train, validate, test, numeric_cols):
    """
    this function takes in 3 dataframes with the same columns,
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler.
    it returns 3 dataframes with the same column names and scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).
    scaler = MinMaxScaler(copy=True).fit(train[numeric_cols])
    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    train_scaled_array = scaler.transform(train[numeric_cols])
    validate_scaled_array = scaler.transform(validate[numeric_cols])
    test_scaled_array = scaler.transform(test[numeric_cols])
    # convert arrays to dataframes
    train_scaled = pd.DataFrame(train_scaled_array, columns=numeric_cols).set_index(
        [train.index.values]
    )
    validate_scaled = pd.DataFrame(
        validate_scaled_array, columns=numeric_cols
    ).set_index([validate.index.values])
    test_scaled = pd.DataFrame(test_scaled_array, columns=numeric_cols).set_index(
        [test.index.values]
    )
    return train_scaled, validate_scaled, test_scaled

def prepare_mall(df):
    
    