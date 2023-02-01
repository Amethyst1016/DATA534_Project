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


def generate_SP500_index(index='original', interval='1d', category='Close', save_path=data_store_path):
      """
      Generates SP500 index data and saves it to a CSV file.

      Parameters:
            index (str, optional): The type of SP500 index to generate. 
                  Options are:\n
                  'original' = SP500 index\n
                  'energy' = SP500 - energy sector index\n
                  'industry' = SP500 - industry sector index\n
                  'consumer' = SP500 - consumer sector index\n
                  Default is 'original'.
            interval (str, optional): The interval to download the data with. 
                  Options are:\n
                  '1d' = daily data\n
                  '1wk' = weekly data\n
                  '1mo' = monthly data\n
                  Default is '1d'.\n
            category (str, optional): The category of data to include in the data frame. 
                  Options are:\n
                  'Open'\n
                  'High'\n
                  'Low'\n
                  'Close'\n
                  'Adj Close'\n
                  'Volume'\n
                  Default is 'Close'.
            save_path (str, optional): The path to the directory where the CSV file will be saved. 
                  Default is None.

      Returns:
            pandas DataFrame: The generated SP500 index data frame.
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
      df = yf.download([symbol], group_by='ticker', interval=interval)
      df = df[[category]]
      df = df.reset_index()
      df = df.rename(columns = {"Date": "date"})
      

      if save_path is not None:
            file_path = save_path+ 'SP500_'+ index + ".csv"
            df.to_csv(file_path,index=False)
            # print(f"SP500 {index} data saved to {file_path}")
            print("SP500 - " + index + " data saved")
      return df


            
def get_df(save_path=data_store_path, input=""):
      """
      Get a data frame from the specified file.
      
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
            SP500_original - daily\n
            SP500_energy - daily\n
            SP500_industry - daily\n
            SP500_consumer - daily\n
            
      Args:
            save_path (str, optional): Path to the data store. Defaults to `data_store_path`.
            input (str, optional): The file name to load.
            
      Returns:
            pd.DataFrame: Data frame loaded from the specified file.
      """
      # choice = input("Which file do you want: ")

      df = pd.read_csv(save_path + input + '.csv', index_col=False)
      # import pandas as pd
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
      return df

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


def moving_average(df=None, MA=7):
      """
      Calculates the moving average of a given data frame.

      Parameters:
            df (pandas DataFrame, optional): The data frame to calculate the moving average on. 
                  Default is None.
            MA (int, optional): The moving average window size. Default is 7.

      Returns:
            pandas DataFrame: The original data frame with the added moving average column.
      """
      if df is None:
            raise ValueError("df parameter must be provided")

      # Calculate the moving average
      MA_data = df.loc[:, 1].rolling(window=MA).mean()
      col_name = 'MA' + str(MA)
      df[col_name] = MA_data
      return df
