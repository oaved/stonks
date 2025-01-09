import yfinance as yf
import pandas as pd

def get_stock_data(tickers, start_date, end_date):
    data = {}
    
    for ticker in tickers:
        try:
            df = pd.read_csv(
                f"./CSVs/{ticker}_returns.csv",
                index_col=0,
                parse_dates=True,

            )
        except FileNotFoundError:
            print(f"File for {ticker} not found. Downloading data...")
            df = yf.download(ticker, start=start_date, end=end_date)
            df.to_csv(f"./CSVs/{ticker}_returns.csv")

        except Exception as error:
            print(f"Error reading data for {ticker}. Error: {error}")

        if "Ticker" in df.index:
            df = df.drop(index="Ticker")
        if "Date" in df.index:
            df = df.drop(index="Date")

        data[ticker] = df

    return data