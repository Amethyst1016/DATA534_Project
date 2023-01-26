#from project import *
import requests
import pandas as pd
import time
apikey = 'O6LFU5LE4ZVYXL1H'

def convert_js_to_df_monthly(data):
      df = pd.json_normalize(data['data'])
      # Convert the 'date' column into a datetime object
      df['date'] = pd.to_datetime(df['date'])
      # df['date'] = pd.DatetimeIndex(df['date']).to_period('M')
      # df['year'] = df['date'].dt.year
      # df['month'] = df['date'].dt.month
      # df = df.drop(columns=['date'])

      # Convert the 'value' column into a float
      df['value'] = pd.to_numeric(df['value'])
      # columns_order = ['date','year','month','value']
      # df = df.reindex(columns=columns_order)
      return df

def get_economic_indicators(api=apikey, interval='monthly', save_path='/Users/gawain/finance_project/dataset/'):
      CPI_url = 'https://www.alphavantage.co/query?function=CPI'+'&interval='+ interval+'&apikey='+api # monthly and semiannual are accepted.
      UNEMPLOYMENT_url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey='+api
      REAL_GDP_PER_CAPITA_url = 'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey='+api
      FEDERAL_FUNDS_RATE_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval='+ interval+'&apikey='+api # daily, weekly, and monthly
      RETAIL_SALE_url = 'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey='+api #monthly
      DURABLES_url = 'https://www.alphavantage.co/query?function=DURABLES&apikey=' + api #monthly 日用商品
      
      indicator_dict={  'cpi':CPI_url,                        # monthly and semiannual are accepted.
                        'unemployment':UNEMPLOYMENT_url,      # monthly is accepted.
                        'gdp':REAL_GDP_PER_CAPITA_url,        # quarterly is accepted.
                        'fundrate':FEDERAL_FUNDS_RATE_url,    # daily, weekly, and monthly are accepted.
                        'retail':RETAIL_SALE_url,             # monthly is accepted.
                        'durables':DURABLES_url               # monthly is accepted.日用商品
                        }
      for i in indicator_dict:
            name = i
            url = indicator_dict[i]

            df = pd.read_json(url)
            df['date'] = df['data'].apply(lambda x: x['date'])
            df['value'] = df['data'].apply(lambda x: x['value'])
            df = df[['date', 'value']]
            df.to_csv(name+'.csv')
            print(name+' csv is set')
            time.sleep(10)
            

get_economic_indicators(api=apikey)

#%%
# Get data from URL
#cpi_url = 'https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey=O6LFU5LE4ZVYXL1H'
#inflation_url = 'https://www.alphavantage.co/query?function=INFLATION&apikey=O6LFU5LE4ZVYXL1H'
#unemployment_url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=O6LFU5LE4ZVYXL1H'

#cpi_df = pd.read_json(cpi_url)
#inflation_df = pd.read_json(inflation_url)
#unemployment_df = pd.read_json(unemployment_url)

# Extract date and value from data
#cpi_df['date'] = cpi_df['data'].apply(lambda x: x['date'])
#cpi_df['value'] = cpi_df['data'].apply(lambda x: x['value'])
#cpi_df = cpi_df[['date', 'value']]
#cpi_df

#inflation_df['date'] = inflation_df['data'].apply(lambda x: x['date'])
#inflation_df['value'] = inflation_df['data'].apply(lambda x: x['value'])
#inflation_df = inflation_df[['date', 'value']]

#unemployment_df['date'] = unemployment_df['data'].apply(lambda x: x['date'])
#unemployment_df['value'] = unemployment_df['data'].apply(lambda x: x['value'])
#unemployment_df = unemployment_df[['date', 'value']]