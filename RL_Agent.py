import pandas as pd
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import yfinance as yf
import PredictFutreTrend
from datetime import datetime, timedelta
import requests
import logging

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# 创建 logger 对象
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 全局日志级别设置为 DEBUG，确保所有消息都能到达处理程序

# 创建文件处理程序，设置级别为 INFO
file_handler = logging.FileHandler('trade_decision.log')
file_handler.setLevel(logging.DEBUG)  # 文件处理程序级别设置为 DEBUG，以便应用过滤器后过滤 INFO 消息

# 创建并添加过滤器
info_filter = InfoFilter()
file_handler.addFilter(info_filter)

# 创建日志格式并设置处理程序格式
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加处理程序到 logger
logger.addHandler(file_handler)

# 移除默认的处理程序（如果有的话）
logger.propagate = False




stocks = ['2330.TW', '2454.TW', '2317.TW', '3008.TW', '2002.TW', '2412.TW', '2882.TW', '2881.TW', '1303.TW', '3045.TW',
          '1216.TW', '1101.TW', '1402.TW', '9933.TW', '1605.TW', '2603.TW', '2609.TW', '3481.TW', '2303.TW', '2308.TW']


# 取得股票資訊
# Input:
#   stock_code: 股票ID
#   start_date: 開始日期，YYYYMMDD
#   stop_date: 結束日期，YYYYMMDD
# Output: 持有股票陣列
def Get_Stock_Informations(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])



# 取得持有股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
# Output: 持有股票陣列
def Get_User_Stocks(account, password):
    data = {'account': account,
            'password': password
            }
    search_url = 'http://140.116.86.242:8081/stock/api/v1/get_user_stocks'
    result = requests.post(search_url, data=data).json()
    if(result['result'] == 'success'):
        return result['data']
    return dict([])



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



# 預約售出股票
# Input:
#   account: 使用者帳號
#   password: 使用者密碼
#   stock_code: 股票ID
#   stock_shares: 售出張數
#   stock_price: 售出價格
# Output: 是否成功預約售出(True/False)
def Sell_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Selling stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    sell_url = 'http://140.116.86.242:8081/stock/api/v1/sell'
    result = requests.post(sell_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'

###################################################################################################

def generate_stock_data(num_stocks=20):
    
    # 获取两个日期：今天和50天前
    today = datetime.today().strftime('%Y-%m-%d')
    fifty_days_ago = (datetime.today() - timedelta(days=50)).strftime('%Y-%m-%d')
    
    # 下载历史数据
    all_stock_data = []
    for stock in stocks[:num_stocks]:
        stock_data = yf.download(stock, start=fifty_days_ago, end=today)
        stock_data.reset_index(inplace=True)
        stock_data = stock_data[['Date', 'Close']]
        stock_data['ticker'] = stock
        all_stock_data.append(stock_data)
    
    # 合并所有股票数据
    combined_data = pd.concat(all_stock_data)
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data.sort_values(by=['ticker', 'Date'], inplace=True)
    combined_data.reset_index(drop=True, inplace=True)
    
    # 截取最近30天的股价数据
    recent_data = combined_data.groupby('ticker').tail(30).reset_index(drop=True)
    
    # 调整为所需格式
    stock_data = []
    for ticker in stocks[:num_stocks]:
        ticker_data = recent_data[recent_data['ticker'] == ticker]['Close'].values
        stock_data.append(ticker_data)
    
    stock_data = np.array(stock_data).T  # 转置以符合 (30, num_stocks) 的格式
    
    return stock_data


# Class for the agent
class MultiStockRLAgent:
    def __init__(self, state_size, action_size, num_stocks, initial_cash, initial_stocks):
        self.state_size = state_size * num_stocks  # 多支股票的狀態大小
        self.action_size = action_size  # 每支股票的行動大小
        self.num_stocks = num_stocks  # 股票數量
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 折扣率
        self.epsilon = 1.0   # 初始探索率，減少為 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        self.cash = initial_cash  # 初始資金
        self.stocks = initial_stocks  # 初始持有股票及其數量，格式為 {'ticker': shares}

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size * self.num_stocks, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        actions = []
        if np.random.rand() <= self.epsilon:
            actions = [random.randrange(self.action_size) for _ in range(self.num_stocks)]
        else:
            act_values = self.model.predict(state)
            actions = [np.argmax(act_values[0][i*self.action_size:(i+1)*self.action_size]) for i in range(self.num_stocks)]
        return actions
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, actions, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))
            target_f = self.model.predict(state)
            for i, action in enumerate(actions):
                target_f[0][i*self.action_size + action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def trade(agent, stock_data, user_stocks):
    logger.info("*** Start of this trade decision making ***")
    logger.info(f"\nInitial Cash: {agent.cash}\nInitial Stocks:            {agent.stocks}")

    # 获取前30天的历史数据作为当前状态
    history_state = np.reshape(stock_data[-30:], [1, -1])  # shape: (1, num_stocks * 30)
    
    # 使用 transformer_model 预测未来一天的收盘价
    predicted_prices = PredictFutreTrend.predict()
    # predicted_prices = []
    # for i in range(20):
    #     predicted_prices.append(i)
    # predicted_prices = np.array(predicted_prices)
    # predicted_prices = np.reshape(predicted_prices, (1, predicted_prices.shape[0]))
    
    # 构造新的状态，将前30天的历史数据与预测的未来一天的收盘价结合
    future_state = np.concatenate((history_state, predicted_prices), axis=1)
    # print(f"Future State shape: {future_state.shape}")
    
    # 让 Agent 根据新的状态生成动作
    actions = agent.act(future_state)
    
    # 调整动作，确保 Agent 不会在没有持有股票的情况下卖出
    adjusted_actions = []
    stock_list = list(agent.stocks.keys())
    
    for i, action in enumerate(actions):
        stock_num = stocks[i]
        begginning_price = 0
        for stock in user_stocks:
            if stock['stock_code_id'] == stock_num[:4]:
                begginning_price = stock['beginning_price']
                break
        print(f'{i+1}. Stock number: {stock_num}, action: {action}')
        print(f'predicted_prices[0][i]: {predicted_prices[0][i]}, begginning_price: {begginning_price}')
        if action ==2 and predicted_prices[0][i] < begginning_price:
            print('Predicted price is lower than the beginning price')
            print(f"Predicted price: {predicted_prices[0][i]}, Beginning price: {begginning_price}")
            adjusted_actions.append(0)
        elif action == 2 and agent.stocks.get(stock_list[i], 0) == 0:
            print('No stock to sell')
            adjusted_actions.append(0)  # 没有持有股票，改为不动作
        else:
            adjusted_actions.append(action)
    
    # print(f"Original Actions: {actions}\nAdjusted Actions: {adjusted_actions}")
    logger.info(f"\nOriginal Actions: {actions}\nAdjusted Actions: {adjusted_actions}")
    
    # 更新持有股票数量和资金
    for i, action in enumerate(adjusted_actions):
        ticker = stock_list[i]
        # fetch the first 4 letters of the ticker Ex. '2330.TW' -> '2330'
        ticker_num = ticker[:4]
        if action == 1:  # 买入
            # 假设买入时花费当前收盘价
            current_price = stock_data[-1, i]  # 获取当前收盘价
            if agent.cash >= current_price:
                agent.stocks[ticker] += 1
                agent.cash -= current_price
                if Buy_Stock('NQ6124052', 'NQ6124', ticker_num, 1, current_price)==False:
                    logger.info(f"Failed to buy {ticker} at {current_price}, rolling back the action")
                    agent.stocks[ticker] -= 1
                    agent.cash += current_price
        elif action == 2:  # 卖出
            if agent.stocks[ticker] > 0:
                current_price = stock_data[-1, i]  # 获取当前收盘价
                agent.stocks[ticker] -= 1
                agent.cash += current_price
                if Sell_Stock('NQ6124052', 'NQ6124', ticker_num, 1, current_price)==False:
                    logger.info(f"Failed to sell {ticker} at {current_price}, rolling back the action")
                    agent.stocks[ticker] += 1
                    agent.cash -= current_price
    
    # print(f"Updated Cash: {agent.cash}\nUpdated Stocks: {agent.stocks}")
    logger.info(f"\nUpdated Cash Estimation: {agent.cash}\nUpdated Stocks Estimation: {agent.stocks}")
    logger.info("*** End of this trade decision making ***\n")
    
    return adjusted_actions


if __name__ == "__main__":
    num_stocks = 20  # 假設有20支股票
    state_size = 1  # 每支股票的狀態維度
    action_size = 3  # 每支股票的動作數量（買、賣、保持）
    initial_cash = 94042675  # 初始資金
    initial_stocks = {  # 初始持有股票及其數量
        '2330.TW': 0,
        '2454.TW': 0,
        '2317.TW': 0,
        '3008.TW': 0,
        '2002.TW': 0,
        '2412.TW': 0,
        '2882.TW': 0,
        '2881.TW': 0,
        '1303.TW': 0,
        '3045.TW': 0,
        '1216.TW': 0,
        '1101.TW': 0,
        '1402.TW': 0,
        '9933.TW': 0,
        '1605.TW': 0,
        '2603.TW': 0,
        '2609.TW': 0,
        '3481.TW': 0,
        '2303.TW': 0,
        '2308.TW': 0
    }
    user_stocks = Get_User_Stocks('NQ6124052', 'NQ6124')
    for stock in user_stocks:
        initial_stocks[stock['stock_code_id']+'.TW'] = stock['shares']
    print(initial_stocks)
    # exit()
    # stocks = ['2330.TW', '2454.TW', '2317.TW', '3008.TW', '2002.TW', '2412.TW', '2882.TW', '2881.TW', '1303.TW', '3045.TW',
    #       '1216.TW', '1101.TW', '1402.TW', '9933.TW', '1605.TW', '2603.TW', '2609.TW', '3481.TW', '2303.TW', '2308.TW']
    agent = MultiStockRLAgent(state_size, action_size, num_stocks, initial_cash, initial_stocks)

    stock_data = generate_stock_data(num_stocks)

    trade(agent, stock_data, user_stocks)