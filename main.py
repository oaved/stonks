import numpy as np
import pandas as pd

from get_stock_data import get_stock_data
from basic_measurements import calc_moving_average, calc_volatility, add_indicators
from plots import basic_candlestick_plot, plot_basic_measurements
from neural_network import NeuralNetwork, create_targets

if __name__ == "__main__":
    chosen_tickers = ["TSLA"]
    start = "2020-01-01"
    end = "2025-01-01"

    stock_data = get_stock_data(chosen_tickers, start, end)
    ticker = "TSLA"
    data = stock_data[ticker]
    training_data, targets = create_targets(data)


    my_nn = NeuralNetwork(input_size=len(training_data[0])) # len of one set of training data
    my_nn.train(training_data, targets=targets)

    
