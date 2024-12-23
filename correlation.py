import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_data(ticker, start='2015-01-01', end=dt.datetime.today().strftime('%Y-%m-%d')):
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Adj Close'].pct_change()
    return df


def plot_correlation_over_time(btc_returns, sp500_returns, window=30):
    rolling_corr = btc_returns.rolling(window=window).corr(sp500_returns)
    mean_corr = rolling_corr.mean()

    dates = np.arange(len(rolling_corr))
    mask = ~np.isnan(rolling_corr)
    model = LinearRegression()
    model.fit(dates[mask].reshape(-1, 1), rolling_corr[mask])
    line = model.predict(dates.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(rolling_corr, label=f'{window}-Day Rolling Correlation', color='orange', linewidth=2)
    plt.axhline(mean_corr, color='black', linestyle='solid', label=f'Average Correlation: {mean_corr:.4f}')
    plt.plot(rolling_corr.index, line, color='red', linestyle='solid', linewidth=2, label='Best Fit Line')
    plt.title('Rolling Correlation between Bitcoin and S&P 500 Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_correlation(btc_returns, sp500_returns):
    plt.figure(figsize=(10, 6))
    plt.scatter(btc_returns, sp500_returns, alpha=0.5)
    plt.title('Correlation between Bitcoin and S&P 500 Daily Returns')
    plt.xlabel('Bitcoin Daily Returns')
    plt.ylabel('S&P 500 Daily Returns')
    plt.grid(True)

    model = LinearRegression()
    model.fit(btc_returns.values.reshape(-1, 1), sp500_returns.values)
    line = model.predict(btc_returns.values.reshape(-1, 1))

    plt.plot(btc_returns, line, color='red', linewidth= 2)

    r2 = r2_score(sp500_returns, line)
    plt.text(0.05, 0.95, f'$R^2 = {r2:.4f}$', transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top')

    plt.tight_layout()
    plt.show()


def main():
    bitcoin_data = get_data('BTC-USD')
    sp500_data = get_data('^GSPC')

    bitcoin_data = bitcoin_data[bitcoin_data.index >= '2015-01-01']
    sp500_data = sp500_data[sp500_data.index >= '2015-01-01']

    combined_data = pd.merge(bitcoin_data['Return'], sp500_data['Return'], left_index=True, right_index=True,
                             suffixes=('_BTC', '_SP500'))
    combined_data.dropna(inplace=True)

    correlation = combined_data['Return_BTC'].corr(combined_data['Return_SP500'])
    print(f"Correlation between Bitcoin and S&P 500 daily returns: {correlation:.4f}")

    plot_correlation(combined_data['Return_BTC'], combined_data['Return_SP500'])
    plot_correlation_over_time(combined_data['Return_BTC'], combined_data['Return_SP500'])


if __name__ == '__main__':
    main()
