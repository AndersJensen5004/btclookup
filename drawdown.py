import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy.stats import expon

# Comment Line 65 to fix second chart and get S&P data from 1985 to present

def get_data(ticker, start='2015-01-01', end=dt.datetime.today().strftime('%Y-%m-%d')):
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Adj Close'].pct_change()
    df['Cumulative Return'] = (1 + df['Return']).cumprod()
    df['Drawdown'] = df['Cumulative Return'] / df['Cumulative Return'].cummax() - 1
    return df


def plot_drawdowns_over_time(bitcoin_data, sp500_data):
    plt.figure(figsize=(14, 7))
    plt.plot(bitcoin_data.index, bitcoin_data['Drawdown'] * 100, label='Bitcoin', color='orange')
    plt.plot(sp500_data.index, sp500_data['Drawdown'] * 100, label='S&P 500', color='blue')
    plt.title('Drawdowns Over Time')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdown_distribution(bitcoin_data, sp500_data):
    plt.figure(figsize=(14, 7))
    sns.histplot(bitcoin_data['Drawdown'].dropna() * 100, bins=50, kde=False, color='orange', label='Bitcoin',
                 stat="density", alpha=0.5)
    sns.histplot(sp500_data['Drawdown'].dropna() * 100, bins=50, kde=False, color='blue', label='S&P 500',
                 stat="density", alpha=0.5)

    bitcoin_drawdowns = -bitcoin_data['Drawdown'].dropna()
    sp500_drawdowns = -sp500_data['Drawdown'].dropna()

    bitcoin_params = expon.fit(bitcoin_drawdowns)
    sp500_params = expon.fit(sp500_drawdowns)

    x = np.linspace(0, max(bitcoin_drawdowns.max(), sp500_drawdowns.max()), 1000)

    plt.plot(-x * 100, expon.pdf(x, *bitcoin_params) / 100, color='darkorange', lw=3, linestyle='solid',
             label='Bitcoin Exponential Fit')
    plt.plot(-x * 100, expon.pdf(x, *sp500_params) / 100, color='darkblue', lw=3, linestyle='solid',
             label='S&P 500 Exponential Fit')


    plt.title('Distribution of % From All Time High')
    plt.xlabel('Drawdown (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    bitcoin_data = get_data('BTC-USD')
    sp500_data = get_data('^GSPC', start='1985-01-01')

    sp500_data = sp500_data[sp500_data.index >= '2015-01-01']

    plot_drawdowns_over_time(bitcoin_data, sp500_data)
    plot_drawdown_distribution(bitcoin_data, sp500_data)


if __name__ == '__main__':
    main()
