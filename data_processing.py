# from project import *
import requests
import pandas as pd
import time
import yfinance as yf
import os

apikey = 'O6LFU5LE4ZVYXL1H'

def convert_js_to_df(data):
    df = pd.json_normalize(data['data'])
    # Convert the 'date' column into a datetime object
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Convert the 'value' column into a float
    df['value'] = pd.to_numeric(df['value'])
    return df

def generate_economic_indicators(api=apikey):
    end = '&apikey='+api
    
    # monthly and semiannual CPI
    CPI_month_url = 'https://www.alphavantage.co/query?function=CPI'+'&interval=monthly'+end
    CPI_semiannual_url = 'https://www.alphavantage.co/query?function=CPI'+'&interval=semiannual'+end
    
    # daily, weekly, and monthly Federal Fund Rate
    FEDERAL_FUNDS_RATE_day_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily'+end
    FEDERAL_FUNDS_RATE_week_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=weekly'+end
    FEDERAL_FUNDS_RATE_month_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly'+end
    
    # monthly Unemployment, retail sale, and durables
    UNEMPLOYMENT_url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT'+end
    RETAIL_SALE_url = 'https://www.alphavantage.co/query?function=RETAIL_SALES'+end
    DURABLES_url = 'https://www.alphavantage.co/query?function=DURABLES'+end
    
    # quarterly GDP per capita
    REAL_GDP_PER_CAPITA_url = 'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA'+end

    indicator_dict={'cpi_month':CPI_month_url,
                    'cpi_semiannual':CPI_semiannual_url,
                    'unemployment':UNEMPLOYMENT_url,
                    'gdp':REAL_GDP_PER_CAPITA_url,
                    'fundrate_day':FEDERAL_FUNDS_RATE_day_url,
                    'fundrate_week':FEDERAL_FUNDS_RATE_week_url,
                    'fundrate_month':FEDERAL_FUNDS_RATE_month_url,
                    'retail':RETAIL_SALE_url,
                    'durables':DURABLES_url
                    }
    for i in indicator_dict:
        name = i
        url = indicator_dict[i]
        r = requests.get(url)
        data = r.json()
        df = convert_js_to_df(data)
        df = df.rename(columns={'value': i})
        
        if not os.path.exists(os.getcwd()+'/data'):
            os.makedirs(os.getcwd()+'/data')
        df.to_csv('data/'+name+'.csv', index=False)
        
        print(name+'.csv is set')
        time.sleep(10)
    print('All csv files are stored in', os.getcwd()+'/data')

def generate_SP500_index(index='original', interval='1d', category='Close'):
    """
    Generates SP500 index data and saves it to a CSV file.

    Parameters:
        index (str, optional): The type of SP500 index to generate. 
              Options are:
              'original' = SP500 index
              'energy' = SP500 - energy sector index
              'industry' = SP500 - industry sector index
              'consumer' = SP500 - consumer sector index
              Default is 'original'.
        interval (str, optional): The interval to download the data with. 
              Options are:
              '1d' = daily data
              '1wk' = weekly data
              '1mo' = monthly data
              Default is '1d'.
        category (str, optional): The category of data to include in the data frame. 
              Options are:
              'Open'
              'High'
              'Low'
              'Close'
              'Adj Close'
              'Volume'
              Default is 'Close'.
        save_path (str, optional): The path to the directory where the CSV file will be saved. 
              Default is None.
    """
    index_dict = {
        'original': '^GSPC',
        'energy': '^GSPE',
        'industry': '^SP500-20',
        'consumer': '^SP500-30'
    }

    if index not in index_dict:
        raise ValueError(f'Invalid index value. Expected one of {list(index_dict.keys())}, got {index}')

    symbol = index_dict[index]
    df = yf.download([symbol], group_by='ticker', interval=interval)
    df = df[[category]]
    df = df.reset_index()
    df = df.rename(columns = {'Date': 'date'})
    df['date'] = df['date'].astype(str).str[:10]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    if not os.path.exists(os.getcwd()+'/data'):
        os.makedirs(os.getcwd()+'/data')   
    df.to_csv('data/'+'SP500_'+index+'.csv', index=False)
    print('SP500_'+index+'.csv', 'is saved in', os.getcwd()+'/data')

def subset(df, start_time, end_time):
    """
    Subset a data frame based on a specified time range.

    Parameters:
        df (pandas.DataFrame): The data frame to subset.
        start_time (str): The start time in the format 'YYYY-MM-DD' or 'YYYY-MM'.
        end_time (str): The end time in the format 'YYYY-MM-DD' or 'YYYY-MM'.

    Returns:
        pandas.DataFrame: The subset of the data frame.
    """
    # Check if the data frame has a column named 'date'
    if 'date' not in df.columns:
        raise ValueError("Input data frame does not have a 'date' column.")

    # Subset the data frame based on the specified time range
    selected_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
    return selected_df


def moving_average(df, col, MA=7):
    """
    Calculates the moving average of a given data frame.

    Parameters:
        df (pandas.DataFrame): The data frame to get column of data. 
        col (str): The name of column to calculate the moving average on.
        MA (int, optional): The moving average window size. Default is 7.

    Returns:
        pandas DataFrame: The original data frame with the added moving average column.
    """

    MA_data = df[col].rolling(window=MA).mean()
    col_name = 'MA' + str(MA) + '_' + col
    df[col_name] = MA_data
    return df
#%%
