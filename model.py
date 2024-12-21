import pandas as pd
import numpy as np
import yfinance as yf
import requests
from dotenv import load_dotenv
import os
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import StringIO

# Unfinished, this served little purpose besides demonstrating overfitting and that historical data would provide no
# significant predictive power

load_dotenv()
FRED_API_KEY = os.getenv('FRED_API')


def get_price_data(symbol='BTC-USD', start='2015-01-01', end=dt.datetime.today().strftime('%Y-%m-%d')):
    df = yf.download(symbol, start=start, end=end)
    return df


def get_blockchain_data():
    response = requests.get('https://api.blockchain.info/charts/n-transactions?timespan=5years&format=csv')
    data = StringIO(response.content.decode('utf-8'))
    df = pd.read_csv(data, sep=' ', header=None, names=['Timestamp', 'Transactions_per_Second'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df.set_index('Timestamp')


def get_macro_data():
    urls = {
        'inflation': f'https://api.stlouisfed.org/fred/series/observations?series_id=T10YIE&api_key={FRED_API_KEY}&file_type=json',
        'interest_rate': f'https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={FRED_API_KEY}&file_type=json'
    }

    macro_data = {}
    for key, url in urls.items():
        response = requests.get(url).json()
        data = pd.DataFrame(response['observations'])
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        macro_data[key] = data['value'].astype(float)

    macro_df = pd.concat(macro_data, axis=1)
    macro_df.columns = ['Inflation_Rate', 'Interest_Rate']
    return macro_df



def merge_datasets(price_data, blockchain_data, macro_data):
    combined_df = price_data[['Adj Close']].copy()
    combined_df = combined_df.rename(columns={'Adj Close': 'Price'})

    combined_df = combined_df.merge(blockchain_data, left_index=True, right_index=True, how='left')

    combined_df = combined_df.merge(macro_data, left_index=True, right_index=True, how='left')

    combined_df.fillna(method='ffill', inplace=True)
    return combined_df



def build_model(df):
    df['Return'] = df['Price'].pct_change()
    df.dropna(inplace=True)

    X = df[['Transactions_per_Second', 'Inflation_Rate', 'Interest_Rate']]
    y = df['Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'R-squared: {r2_score(y_test, y_pred)}')

    return model



def plot_results(df, model):
    df['Predicted_Return'] = model.predict(df[['Transactions_per_Second', 'Inflation_Rate', 'Interest_Rate']].dropna())
    df[['Return', 'Predicted_Return']].plot(figsize=(10, 6))
    plt.title('Actual vs Predicted Returns')
    plt.show()


def main():
    price_data = get_price_data()
    blockchain_data = get_blockchain_data()
    macro_data = get_macro_data()

    combined_df = merge_datasets(price_data, blockchain_data, macro_data)

    model = build_model(combined_df)

    plot_results(combined_df, model)


if __name__ == '__main__':
    main()