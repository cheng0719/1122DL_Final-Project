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

stocks = ['2330.TW', '2454.TW', '2317.TW', '3008.TW', '2002.TW', '2412.TW', '2882.TW', '2881.TW', '1303.TW', '3045.TW',
          '1216.TW', '1101.TW', '1402.TW', '9933.TW', '1605.TW', '2603.TW', '2609.TW', '3481.TW', '2303.TW', '2308.TW']


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


def trade(agent, stock_data, num_stocks):
    # 获取前30天的历史数据作为当前状态
    history_state = np.reshape(stock_data[-30:], [1, -1])  # shape: (1, num_stocks * 30)
    
    # 使用 transformer_model 预测未来一天的收盘价
    predicted_prices = PredictFutreTrend.predict()
    
    # 构造新的状态，将前30天的历史数据与预测的未来一天的收盘价结合
    future_state = np.concatenate((history_state, predicted_prices), axis=1)
    
    # 让 Agent 根据新的状态生成动作
    actions = agent.act(future_state)
    
    # 调整动作，确保 Agent 不会在没有持有股票的情况下卖出
    adjusted_actions = []
    stock_list = list(agent.stocks.keys())
    
    for i, action in enumerate(actions):
        if action == 2 and agent.stocks.get(stock_list[i], 0) == 0:
            adjusted_actions.append(0)  # 没有持有股票，改为不动作
        else:
            adjusted_actions.append(action)
    
    print(f"Original Actions: {actions}, Adjusted Actions: {adjusted_actions}")
    
    # 更新持有股票数量和资金
    for i, action in enumerate(adjusted_actions):
        ticker = stock_list[i]
        if action == 1:  # 买入
            # 假设买入时花费当前收盘价
            current_price = stock_data[-1, i]  # 获取当前收盘价
            if agent.cash >= current_price:
                agent.stocks[ticker] += 1
                agent.cash -= current_price
        elif action == 2:  # 卖出
            if agent.stocks[ticker] > 0:
                current_price = stock_data[-1, i]  # 获取当前收盘价
                agent.stocks[ticker] -= 1
                agent.cash += current_price
    
    print(f"Updated Cash: {agent.cash}, Updated Stocks: {agent.stocks}")
    
    return adjusted_actions


if __name__ == "__main__":
    num_stocks = 20  # 假設有20支股票
    state_size = 1  # 每支股票的狀態維度
    action_size = 3  # 每支股票的動作數量（買、賣、保持）
    initial_cash = 100000000  # 初始資金
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
    # stocks = ['2330.TW', '2454.TW', '2317.TW', '3008.TW', '2002.TW', '2412.TW', '2882.TW', '2881.TW', '1303.TW', '3045.TW',
    #       '1216.TW', '1101.TW', '1402.TW', '9933.TW', '1605.TW', '2603.TW', '2609.TW', '3481.TW', '2303.TW', '2308.TW']
    agent = MultiStockRLAgent(state_size, action_size, num_stocks, initial_cash, initial_stocks)

    stock_data = generate_stock_data(num_stocks)

    trade(agent, stock_data, num_stocks)