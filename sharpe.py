import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from dotenv import load_dotenv
import os

# Works to calculate roling sharpe on BTC

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API')

def get_btc_data(start_date='2015-01-01', end_date='2024-01-01'):
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    print("BTC Data:\n", btc_data.head())
    return btc_data

def get_10yr_treasury_rate():
    fred = Fred(api_key=FRED_API_KEY)
    treasury_data = fred.get_series('DGS10', start='2015-01-01', end='2024-01-01')
    treasury_data = treasury_data.resample('D').ffill().dropna()  # Resample to daily frequency and forward fill
    print("Treasury Data:\n", treasury_data.head())
    latest_rate = treasury_data.iloc[-1]
    daily_risk_free_rate = (1 + latest_rate / 100) ** (1/252) - 1
    print("Daily Risk-Free Rate:", daily_risk_free_rate)
    return daily_risk_free_rate

def calculate_daily_returns(data):
    data['Daily Return'] = data['Adj Close'].pct_change()
    print("Daily Returns:\n", data[['Adj Close', 'Daily Return']].head())
    return data

def calculate_rolling_sharpe_ratio(data, window=252, daily_risk_free_rate=0.0001):
    excess_returns = data['Daily Return'] - daily_risk_free_rate
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    rolling_sharpe_ratio = (rolling_mean / rolling_std) * np.sqrt(252)
    data['Rolling Sharpe Ratio'] = rolling_sharpe_ratio
    print("Rolling Sharpe Ratio:\n", data[['Rolling Sharpe Ratio']].head(window + 5))
    return data

def plot_sharpe_ratio(data):
    sns.set(style='whitegrid')
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Rolling Sharpe Ratio'], label='Rolling Sharpe Ratio', color='blue', linewidth=2)
    plt.title('Rolling Sharpe Ratio of BTC', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Sharpe Ratio', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    btc_data = get_btc_data()
    daily_risk_free_rate = get_10yr_treasury_rate()
    btc_data = calculate_daily_returns(btc_data)
    btc_data = calculate_rolling_sharpe_ratio(btc_data, daily_risk_free_rate=daily_risk_free_rate)
    btc_data.dropna(inplace=True)
    plot_sharpe_ratio(btc_data)

if __name__ == '__main__':
    main()
