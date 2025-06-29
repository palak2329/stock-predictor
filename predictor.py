import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Ticker', 'Date'])
    return df

def train_and_predict(df, ticker):
    stock_data = df[df['Ticker'] == ticker].copy()

    if len(stock_data) < 15:
        return None, None, f"Not enough data for {ticker}"

    X = stock_data[['Open', 'High', 'Low', 'Volume']].values
    y = stock_data['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test, label='Actual', color='blue')
    ax.plot(y_pred, label='Predicted', color='red')
    ax.set_title(f"{ticker} - Actual vs Predicted Close Price")
    ax.set_xlabel('Test Data Points')
    ax.set_ylabel('Price')
    ax.legend()

    metrics = {
        'MSE': round(mse, 4),
        'R2': round(r2, 4)
    }

    return metrics, fig, None