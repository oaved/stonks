from get_stock_data import get_stock_data
from basic_measurements import calc_moving_average, calc_volatility, add_indicators
from plots import basic_candlestick_plot, plot_basic_measurements
from neural_network import NeuralNetwork

import numpy as np
import pandas as pd

def create_targets(data, margin_per_share=20, horizon=14):
    window_size = horizon * 10
    num_targets = len(data["Close"]) - horizon
    targets = np.zeros((num_targets, 3))  # [buy, hold, sell]
    training_data = []

    for i in range(window_size, num_targets):
        training_window = data["Close"].iloc[i - window_size:i + 1]  # Training window
        training_data.append(training_window.values)

        current_price = float(data["Close"].iloc[i]) 
        future_price = float(data["Close"].iloc[i + horizon])

        # Calculate target based on price movement
        if future_price >= current_price + margin_per_share:
            targets[i] = [1, 0, 0]  # Buy
        elif future_price <= current_price - margin_per_share:
            targets[i] = [0, 0, 1]  # Sell
        else:
            targets[i] = [0, 1, 0]  # Hold

    return np.array(training_data), targets[window_size:]



if __name__ == "__main__":
    chosen_tickers = ["TSLA"]
    start = "2020-01-01"
    end = "2025-01-01"

    stock_data = get_stock_data(chosen_tickers, start, end)
    for ticker, data in stock_data.items():
        training_data, targets = create_targets(data)


