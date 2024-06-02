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

# print('Tensorflow version: {}'.format(tf.__version__))


stocks = ['2330', '2454', '2317', '3008', '2002', '2412', '2882', '2881', '1303', '3045',
          '1216', '1101', '1402', '9933', '1605', '2603', '2609', '3481', '2303', '2308']

def predict():
    predictions = []
    for stock in stocks :
        # fetch2 datetime, one is today datetime, another is 50 days before today, convert these 2 datatime to string and set the formet as 'YYYY-MM-DD'
        # today = pd.to_datetime('today').strftime('%Y-%m-%d')
        # fifty_days_before = (pd.to_datetime('today') - pd.DateOffset(days=50)).strftime('%Y-%m-%d')
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
        print('seq_data shape: {}'.format(seq_data.shape))

        # Load the model
        model = load_model(f'{stock}_model.keras')

        # Predict the future trend
        prediction = model.predict(seq_data)
        prediction = scaler.inverse_transform(prediction)
        print(f'{stock} future trend prediction: {prediction[0][0]}')

        predictions.append(prediction[0][0])
    
    predictions = np.array(predictions)
    print(predictions.shape)
    return predictions

        
        
