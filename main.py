from get_stock_data import get_stock_data
from basic_measurements import calc_moving_average, calc_volatility, add_indicators
from plots import basic_candlestick_plot, plot_basic_measurements
from neural_network import NeuralNetwork

    
def create_targets(data, horisont):
    print(data, horisont)   



if __name__ == "__main__":
    chosen_tickers = ["TSLA"]
    start = "2020-01-01"
    end = "2023-11-01"

    stock_data = get_stock_data(chosen_tickers, start, end)
    for ticker, data in stock_data.items():
        print(data.head())
