import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def get_data(ticker, start='2015-01-01', end=dt.datetime.today().strftime('%Y-%m-%d')):
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Adj Close'].pct_change()
    return df


def plot_volatility(returns_btc, returns_spx, window=60):
    realized_volatility_btc = returns_btc.rolling(window=window).std() * (252 ** 0.5) * 100
    realized_volatility_spx = returns_spx.rolling(window=window).std() * (252 ** 0.5) * 100
    volatility_of_volatility_btc = realized_volatility_btc.rolling(window=window).std()
    volatility_of_volatility_spx = realized_volatility_spx.rolling(window=window).std()

    plt.figure(figsize=(10, 6))
    plt.plot(realized_volatility_btc, label=f'BTC {window}-Day Rolling Realized Volatility')
    plt.plot(realized_volatility_spx, label=f'S&P 500 {window}-Day Rolling Realized Volatility')
    plt.title('Realized Volatility of Bitcoin and S&P 500 Over Time')
    plt.xlabel('Date')
    plt.ylabel('Realized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    vol_of_vol_series_btc = volatility_of_volatility_btc.dropna()
    vol_of_vol_series_spx = volatility_of_volatility_spx.dropna()
    X_btc = np.arange(len(vol_of_vol_series_btc)).reshape(-1, 1)
    y_btc = vol_of_vol_series_btc.values
    X_spx = np.arange(len(vol_of_vol_series_spx)).reshape(-1, 1)
    y_spx = vol_of_vol_series_spx.values
    model_btc = LinearRegression()
    model_spx = LinearRegression()
    model_btc.fit(X_btc, y_btc)
    model_spx.fit(X_spx, y_spx)
    y_pred_btc = model_btc.predict(X_btc)
    y_pred_spx = model_spx.predict(X_spx)
    r_squared_btc = model_btc.score(X_btc, y_btc)
    r_squared_spx = model_spx.score(X_spx, y_spx)

    plt.figure(figsize=(10, 6))
    plt.plot(volatility_of_volatility_btc, label=f'BTC {window}-Day Rolling Volatility of Volatility', color='orange')
    plt.plot(volatility_of_volatility_spx, label=f'S&P 500 {window}-Day Rolling Volatility of Volatility',
             color='green')
    plt.plot(vol_of_vol_series_btc.index, y_pred_btc, label=f'BTC Line of Best Fit (R^2 = {r_squared_btc:.2f})',
             color='blue')
    plt.plot(vol_of_vol_series_spx.index, y_pred_spx, label=f'S&P 500 Line of Best Fit (R^2 = {r_squared_spx:.2f})',
             color='red')
    plt.title('Volatility of Realized Volatility of Bitcoin and S&P 500 Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility of Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    bitcoin_data = get_data('BTC-USD')
    spx_data = get_data('^GSPC')
    bitcoin_data = bitcoin_data[bitcoin_data.index >= '2015-01-01']
    spx_data = spx_data[spx_data.index >= '2015-01-01']
    plot_volatility(bitcoin_data['Return'], spx_data['Return'])


if __name__ == '__main__':
    main()
