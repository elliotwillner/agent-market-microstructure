import yfinance as yf
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Fetch historical financial data
ticker = "AAPL"  # Example ticker
data = yf.download(ticker, start="2023-01-01", end="2023-12-01", progress=False)
historical_prices = data['Close'].values  # Use the closing prices

# Define the market
class OrderBook:
    def __init__(self):
        self.buy_orders = deque()  # Sorted by price descending
        self.sell_orders = deque()  # Sorted by price ascending
        self.mid_price = historical_prices[0]  # Initialize with the first price

    def add_order(self, price, size, side):
        if side == 'buy':
            self.buy_orders.append((price, size))
            self.buy_orders = deque(sorted(self.buy_orders, key=lambda x: -x[0]))  # Sort descending
        elif side == 'sell':
            self.sell_orders.append((price, size))
            self.sell_orders = deque(sorted(self.sell_orders, key=lambda x: x[0]))  # Sort ascending

    def match_orders(self):
        while self.buy_orders and self.sell_orders and self.buy_orders[0][0] >= self.sell_orders[0][0]:
            buy_price, buy_size = self.buy_orders.popleft()
            sell_price, sell_size = self.sell_orders.popleft()

            trade_size = min(buy_size, sell_size)
            if buy_size > sell_size:
                self.buy_orders.appendleft((buy_price, buy_size - trade_size))
            elif sell_size > buy_size:
                self.sell_orders.appendleft((sell_price, sell_size - trade_size))

            self.mid_price = (buy_price + sell_price) / 2  # Update mid-price

    def get_spread(self):
        if self.buy_orders and self.sell_orders:
            return self.sell_orders[0][0] - self.buy_orders[0][0]
        return None

# Define agents
class MarketMakerAgent:
    def __init__(self, order_book):
        self.order_book = order_book

    def act(self):
        price_deviation = np.random.normal(0, 1)
        base_price = self.order_book.mid_price
        buy_price = max(1, base_price - 2 + price_deviation)
        sell_price = base_price + 2 + price_deviation
        self.order_book.add_order(buy_price, random.randint(1, 10), 'buy')
        self.order_book.add_order(sell_price, random.randint(1, 10), 'sell')

class TraderAgent:
    def __init__(self, order_book):
        self.order_book = order_book

    def act(self):
        trade_side = random.choice(['buy', 'sell'])
        price_deviation = np.random.normal(0, 5)
        trade_price = self.order_book.mid_price + price_deviation
        self.order_book.add_order(max(1, trade_price), random.randint(1, 5), trade_side)

class TradingAgent:
    def __init__(self, order_book, initial_cash=1000):
        self.order_book = order_book
        self.cash = initial_cash
        self.holdings = 0
        self.profit = 0

    def act(self):
        # Simple momentum strategy
        moving_avg = np.mean(historical_prices[:len(historical_prices)//2])
        if self.order_book.mid_price > moving_avg:  # Sell if the price is high
            if self.holdings > 0:
                sell_price = self.order_book.mid_price
                self.cash += sell_price
                self.holdings -= 1
                self.profit = self.cash + self.holdings * self.order_book.mid_price - 1000
        else:  # Buy if the price is low
            if self.cash >= self.order_book.mid_price:
                buy_price = self.order_book.mid_price
                self.cash -= buy_price
                self.holdings += 1
                self.profit = self.cash + self.holdings * self.order_book.mid_price - 1000

# Simulate the market
def simulate_market(steps=100):
    order_book = OrderBook()
    agents = [MarketMakerAgent(order_book) for _ in range(3)] + \
             [TraderAgent(order_book) for _ in range(10)] + \
             [TradingAgent(order_book)]

    mid_prices = []
    spreads = []
    profits = []

    for step in range(min(steps, len(historical_prices))):
        order_book.mid_price = historical_prices[step]
        for agent in agents:
            if isinstance(agent, TradingAgent):
                agent.act()
                profits.append(agent.profit)
            else:
                agent.act()

        order_book.match_orders()
        mid_prices.append(order_book.mid_price)
        spreads.append(order_book.get_spread())

    return mid_prices, spreads, profits

# Run the simulation
steps = 200
mid_prices, spreads, profits = simulate_market(steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(mid_prices, label='Mid Prices')
plt.title('Market Mid Prices')
plt.ylabel('Price')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(spreads, label='Bid-Ask Spread', color='orange')
plt.title('Market Bid-Ask Spread')
plt.ylabel('Spread')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(profits, label='Trading Agent Profit', color='green')
plt.title('Trading Agent Profit')
plt.ylabel('Profit')
plt.xlabel('Time Step')
plt.legend()

plt.tight_layout()
plt.show()