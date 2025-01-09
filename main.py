import plotly.graph_objects as go
from plotly.subplots import make_subplots
from get_stock_data import get_stock_data
from basic_measurements import calc_moving_average, calc_volatility, add_indicators

def plot(data, ticker):
    #   Initiate figure consisting two plots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("Candlestick Chart", "RSI", "Volatility")
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], close=data["Close"],
        high=data["High"], low=data["Low"], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["Short_MA"], mode="lines", name="Short MA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Long_MA"], mode="lines", name="Long MA"), row=1, col=1)

    # RSI
    #   Cool names
    bullish = data[data["Signal"] == 1]
    bearish = data[data["Signal"] == -1]

    fig.add_trace(go.Scatter(
        x=bullish.index, y=bullish["Close"],
        mode="markers", marker=dict(color="green", size=10), name="Buy"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bearish.index, y=bearish["Close"],
        mode="markers", marker=dict(color="red", size=10), name="Sell"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=[70] * len(data), mode="lines",
        line=dict(dash="dash", color="red"), name="Overbought"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=[30] * len(data), mode="lines",
        line=dict(dash="dash", color="green"), name="Oversold"
    ), row=2, col=1)

    # Volatility
    fig.add_trace(go.Scatter(x=data.index, y=data["Rolling_Std"], mode="lines", name="Rolling Std"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["EWMA"], mode="lines", name="EWMA"), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Annualized_Volatility"], mode="lines", name="Annulized Volatility"), row=3, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2_title="Date",
        yaxis1_title="Price",
        yaxis2_title="RSI",
        height=800,
        showlegend=True
    )

    fig.show()


if __name__ == "__main__":
    chosen_tickers = ["AAPL", "TSLA"]
    start = "2023-01-01"
    end = "2023-11-01"

    stock_data = get_stock_data(chosen_tickers, start, end)
    for ticker, data in stock_data.items():
        data = calc_moving_average(data) 
        data = add_indicators(data)
        data = calc_volatility(data)
        plot(data, ticker)
