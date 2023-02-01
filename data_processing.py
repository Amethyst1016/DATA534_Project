# from project import *
import requests
import pandas as pd
import time
import yfinance as yf

apikey = 'O6LFU5LE4ZVYXL1H'
data_store_path = '/Users/gawain/finance_project/dataset/' #这里之后要改

def convert_js_to_df(data):
      df = pd.json_normalize(data['data'])
      # Convert the 'date' column into a datetime object
      df['date'] = pd.to_datetime(df['date'])

      # Convert the 'value' column into a float
      df['value'] = pd.to_numeric(df['value'])
      return df

def generate_economic_indicators(api=apikey, save_path=data_store_path):
      CPI_month_url = 'https://www.alphavantage.co/query?function=CPI'+'&interval=monthly'+ '&apikey='+api # monthly and semiannual are accepted.
      CPI_semiannual_url = 'https://www.alphavantage.co/query?function=CPI'+'&interval=semiannual'+ '&apikey='+api # monthly and semiannual are accepted.
      UNEMPLOYMENT_url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey='+api
      REAL_GDP_PER_CAPITA_url = 'https://www.alphavantage.co/query?function=REAL_GDP_PER_CAPITA&apikey='+api
      FEDERAL_FUNDS_RATE_day_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily'+ '&apikey='+api # daily, weekly, and monthly
      FEDERAL_FUNDS_RATE_week_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=weekly'+ '&apikey='+api
      FEDERAL_FUNDS_RATE_month_url = 'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly'+ '&apikey='+api 
      
      
      RETAIL_SALE_url = 'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey='+api #monthly
      DURABLES_url = 'https://www.alphavantage.co/query?function=DURABLES&apikey=' + api #monthly 日用商品
      
      indicator_dict={  'cpi_month':CPI_month_url,                        
                        'cpi_semiannual':CPI_semiannual_url,
                        'unemployment':UNEMPLOYMENT_url,                # monthly is accepted.
                        'gdp':REAL_GDP_PER_CAPITA_url,                  # quarterly is accepted.
                        'fundrate_day':FEDERAL_FUNDS_RATE_day_url,   
                        'fundrate_week':FEDERAL_FUNDS_RATE_week_url,
                        'fundrate_month':FEDERAL_FUNDS_RATE_month_url,
                        'retail':RETAIL_SALE_url,                       # monthly is accepted.
                        'durables':DURABLES_url                         # monthly is accepted.日用商品
                        }
      for i in indicator_dict:
            name = i
            url = indicator_dict[i]
            r = requests.get(url)
            data = r.json()
            df = convert_js_to_df(data)
            df = df.rename(columns={'value': i})
            df.to_csv(save_path+name+'.csv', index=False)
            print(name+' csv is set')
            time.sleep(10)
      print('Done')


def generate_SP500_index(index='original', inter='1d', category='Close', save_path=data_store_path):
      """
      Generates SP500 index data and saves it to a CSV file.

      Parameters:
            index (str): SP500, SP500 - energy, SP500 - industry, SP500 - consumer
            inter (str): 1d, 1wk, 1mo
            category (str): Open, High, Low, Close, Adj Close, Volume
            save_path (str): Path to the directory where the CSV file will be saved.

      Returns:
            None
      """
      index_dict = {
            'original': '^GSPC',
            'energy': '^GSPE',
            'industry': '^SP500-20',
            'consumer': '^SP500-30'
      }

      if index not in index_dict:
            raise ValueError(f"Invalid index value. Expected one of {list(index_dict.keys())}, got {index}")
      
      symbol = index_dict[index]
      df = yf.download([symbol], group_by='ticker', interval=inter)
      df = df[[category]]
      df = df.reset_index()
      file_path = f"{save_path}SP500_{index}.csv"
      df.to_csv(file_path)
      print(f"SP500 {index} data saved to {file_path}")

            
def get_df(save_path=data_store_path, input=""):
      """
      Get a dataframe from the specified file.
      
      Accepted input file formats:
            cpi_monthly\n
            cpi_semiannual\n
            unemployment - monthly\n
            gdp - quarterly\n
            fundrate_day\n
            fundrate_week\n
            fundrate_month\n
            retail - month\n
            durables - month\n
            SP500 - daily\n
            SP500_sector_energy - daily\n
            SP500_sector_industry - daily\n
            SP500_sector_consumer - daily\n
            
      Args:
            save_path (str, optional): Path to the data store. Defaults to `data_store_path`.
            input (str, optional): The file name to load.
            
      Returns:
            pd.DataFrame: Dataframe loaded from the specified file.
      """
      # choice = input("Which file do you want: ")

      df = pd.read_csv(save_path + input + '.csv')
      # import pandas as pd
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
      return df

def subset(df, start_time, end_time):
      selected_df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
      return selected_df
