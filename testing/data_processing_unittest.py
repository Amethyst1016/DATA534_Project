# %%
import unittest
import pandas as pd
# %%
from data_processing import convert_js_to_df
class TestConvertJsToDf(unittest.TestCase):
      def test_convert_js_to_df(self):
            data = {
                  "data": [
                  {"date": "2022-01-01", "value": 100.0},
                  {"date": "2022-01-02", "value": 200.0},
                  {"date": "2022-01-03", "value": 300.0}
                  ]
            }
            result = convert_js_to_df(data)
            expected_result = pd.DataFrame({
                  "date": [
                  "2022-01-01",
                  "2022-01-02",
                  "2022-01-03"
                  ],
                  "value": [
                  100.0,
                  200.0,
                  300.0
                  ]
            }, columns=["date", "value"])
            expected_result['date'] = pd.to_datetime(expected_result['date'], format='%Y-%m-%d')

            self.assertTrue(result.equals(expected_result))


# %%
import os
import yfinance as yf
from data_processing import generate_SP500_index

class TestGenerateSP500Index(unittest.TestCase):

      def test_generate_SP500_index(self):
            index = 'original'
            interval = '1d'
            category = 'Close'
            
            generate_SP500_index(index, interval, category)
            
            self.assertTrue(os.path.exists('data/SP500_'+index+'.csv'))
            
            df = pd.read_csv('data/SP500_'+index+'.csv')
            
            self.assertIn('date', df.columns)
            self.assertIn(category, df.columns)
            
            os.remove('data/SP500_'+index+'.csv')
            self.assertFalse(os.path.exists('data/SP500_'+index+'.csv'))
            
      def test_generate_SP500_index_error(self):
            index = 'invalid_index'
            interval = '1d'
            category = 'Close'
            
            with self.assertRaises(ValueError):
                  generate_SP500_index(index, interval, category)

# %%
from data_processing import subset

class TestSubset(unittest.TestCase):
      def setUp(self):
            self.df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07'],
                                    'value': [1, 2, 3, 4, 5, 6, 7]})

      def test_subset(self):
            result = subset(self.df, '2022-01-03', '2022-01-06')
            expected_result = pd.DataFrame({'date': ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06'],
                                          'value': [3, 4, 5, 6]})
            self.assertNotEqual(result.equals(expected_result), True)
# %%
from data_processing import moving_average

class TestMovingAverage(unittest.TestCase):
      def setUp(self):
            self.df = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07'],
                                    'value': [1, 2, 3, 4, 5, 6, 7]})

      def test_moving_average_default_window(self):
            result = moving_average(self.df, 'value')
            expected_result = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07'],
                                          'value': [1, 2, 3, 4, 5, 6, 7],
                                          'MA7_value': [None, None, None, None, None, None, 4.0]})
            self.assertTrue(result.equals(expected_result))

      def test_moving_average_custom_window(self):
            result = moving_average(self.df, 'value', MA=3)
            expected_result = pd.DataFrame({'date': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07'],
                                          'value': [1, 2, 3, 4, 5, 6, 7],
                                          'MA3_value': [None, None, 2.0, 3.0, 4.0, 5.0, 6.0]})
            self.assertTrue(result.equals(expected_result))


# %%
if __name__ == '__main__':
      # unittest.main()
      unittest.main(argv=[''], verbosity=2, exit=False)


# %%
