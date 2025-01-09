import numpy as np
import pandas as pd

def calc_moving_average(data: dict, short_window=10, long_window=20):
    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

    #   Short MA is more responsive to new changes than long MA
    #   If short MA passes long MA => buy and opposite => sell
    data = data.dropna().copy()
    data.loc[:,"Signal"] = np.where(
        (data["Short_MA"] > data["Long_MA"]) & (data["Short_MA"].shift(1) <= data["Long_MA"].shift(1)), 1, np.where(
            (data["Short_MA"] < data["Long_MA"]) & (data["Short_MA"].shift(1) >= data["Long_MA"].shift(1)), -1, 0
        )
    )

    data = data.dropna()
    return data

def calc_volatility(data):
    data["log_returns"] = np.log(pd.to_numeric(data["Close"]) / pd.to_numeric(data["Close"].shift(1)))
    data["Rolling_Std"] = data["log_returns"].rolling(window=20).std()
    #   Exponentially Weighted Moving Average 
    data["EWMA"] = data["log_returns"].ewm(span=20).std()
    data["Annualized_Volatility"] = data["Rolling_Std"] * np.sqrt(252) #  252 trading days in a year

    data = data.dropna()
    return data

def add_indicators(data):
    delta = pd.to_numeric(data["Close"]).diff()

    #   Some trading strategy
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (abs(delta.where(delta < 0, 0))).rolling(window=14).mean()

    rs = gain / loss
    data["RSI"] = (100 - (100 / (1 + rs)))
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Upper_Band"] = data["MA_20"] + 2 * data["Close"].rolling(window=20).std()
    data["Upper_Band"] = data["MA_20"] - 2 * data["Close"].rolling(window=20).std()
    
    data = data.dropna().copy()
    return data