import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta


stocks = { 
    '2330.TW',
    '2454.TW',
    '2317.TW',
    '3008.TW',
    '2002.TW',
    '2412.TW',
    '2882.TW',
    '2881.TW',
    '1303.TW',
    '3045.TW',
    '1216.TW',
    '1101.TW',
    '1402.TW',
    '9933.TW',
    '1605.TW',
    '2603.TW',
    '2609.TW',
    '3481.TW',
    '2303.TW',
    '2308.TW'
}

# 預約購入股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 購入張數
#   stock_price: 購入價格
# Output: 是否成功預約購入(True/False)
def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Buying stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    buy_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
    result = requests.post(buy_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'

def predict(stock):
    today = datetime.today().strftime('%Y-%m-%d')
    fifty_days_before = (datetime.today() - timedelta(days=50)).strftime('%Y-%m-%d')

    stock_data = yf.download(f'{stock}.TW', start=fifty_days_before, end=today)
    stock_data = stock_data.reset_index()
    stock_data = stock_data[['Date', 'Close']] 
    stock_data.columns = ['date', 'close'] 
    stock_data['ticker'] = stock  # 添加一列來標識股票
    # print(stock_data.head())
    # 標準化數據
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = stock_data[['close']].values # 'volume', 'open', 'high', 'low', 
    data_scaled = scaler.fit_transform(data)

    # 將數據轉換為序列數據
    # fetch the lastest 30 days data
    seq_data = []
    latest_30_data = data_scaled[-30:]
    seq_data.append(latest_30_data)
    seq_data = np.array(seq_data)
    seq_data = seq_data.reshape(seq_data.shape[0], seq_data.shape[1], seq_data.shape[2])
    # print('seq_data shape: {}'.format(seq_data.shape))

    # Load the model
    model = load_model(f'{stock}_model.keras')

    # Predict the future trend
    prediction = model.predict(seq_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'{stock} future trend prediction: {prediction[0][0]}')
    return prediction[0][0]

if __name__ == "__main__":
    predict_price = predict('2330')
    result = Buy_Stock("NQ6124052", "NQ6124", '2330', 1, predict_price)
    print(result)