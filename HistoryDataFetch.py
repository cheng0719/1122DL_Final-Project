import yfinance as yf
import pandas as pd

# 定義股票列表
stocks = ['2330.TW', 
          '2317.TW', 
          '6505.TW', 
          '2412.TW', 
          '2882.TW', 
          '1301.TW', 
          '2308.TW', 
          '3008.TW', 
          '2002.TW', 
          '2454.TW']  # 添加更多股票代碼

# 抓取多支股票資料
all_stock_data = []

for stock in stocks:
    stock_data = yf.download(stock, start='2010-01-01', end='2023-12-31')
    stock_data = stock_data.reset_index()
    stock_data = stock_data[['Date', 'High', 'Low', 'Open', 'Close']]
    stock_data.columns = ['date', 'high', 'low', 'open', 'close']
    stock_data['ticker'] = stock  # 添加一列來標識股票
    all_stock_data.append(stock_data)

# 合併所有股票數據
combined_data = pd.concat(all_stock_data)
combined_data['date'] = pd.to_datetime(combined_data['date'])
combined_data = combined_data.sort_values(by=['ticker', 'date']).reset_index(drop=True)

# 將資料輸出至CSV檔案
combined_data.to_csv('./combined_10_stock_prices_datetime_2010-2023.csv', index=False)




# import yfinance as yf
# import pandas as pd

# # 使用YFinance抓取台積電股票資料
# stock_data = yf.download('2330.TW', start='2010-01-01', end='2023-12-31')

# # 取出所需欄位並重新命名欄位名稱
# stock_data = stock_data.reset_index()
# stock_data = stock_data[['Date', 'High', 'Low', 'Open', 'Close']]
# stock_data.columns = ['date', 'high', 'low', 'open', 'close']
# # stock_data = stock_data[['Date', 'Close']]
# # stock_data.columns = ['date', 'close']

# # 將數值的最小位設為整數個位數
# stock_data['high'] = stock_data['high']
# stock_data['low'] = stock_data['low']
# stock_data['open'] = stock_data['open']
# stock_data['close'] = stock_data['close']
# # .astype(int)

# # 轉換日期格式為'%Y%m%d'
# stock_data['date'] = stock_data['date'].dt.strftime("%Y-%m-%d %H:%M:%S")

# # 依照日期遞增排列
# stock_data = stock_data.sort_values(by='date')

# # 將資料輸出至CSV檔案
# stock_data.to_csv('./tsmc_prices_datetime_2010-2023.csv', index=False)
