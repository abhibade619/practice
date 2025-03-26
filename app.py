import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import tensorflow as tf
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("📊 Stock Price Prediction App")
st.markdown("Select a stock ticker and model to forecast closing prices with visualization and evaluation.")

# --- Sidebar ---
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()
model_name = st.sidebar.selectbox("Select Model", ["ARIMA", "LSTM", "Prophet"])

# --- Load Data ---
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

df = load_data(ticker)

st.subheader(f"Raw Closing Price Data for {ticker}")
st.line_chart(df.set_index('Date')['Close'])

# --- Preprocessing ---
df['MA_20'] = df['Close'].rolling(window=20).mean()
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

scaler = MinMaxScaler()
df['Close_Scaled'] = scaler.fit_transform(df[['Close']])

train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# --- Model Training ---
def train_arima(train, test):
    model = auto_arima(train['Close'], seasonal=False, suppress_warnings=True)
    preds = model.predict(n_periods=len(test))
    return preds, test['Close'].values, test['Date']

def train_prophet(df, train_size):
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df.dropna(inplace=True)

    model = Prophet()
    model.fit(prophet_df.iloc[:train_size])
    future = model.make_future_dataframe(periods=len(df) - train_size)
    forecast = model.predict(future)
    preds = forecast['yhat'].iloc[-len(df) + train_size:].values
    return preds, df['Close'].iloc[train_size:].values, df['Date'].iloc[train_size:]

def train_lstm(df, train_size):
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    data = df['Close_Scaled'].values.reshape(-1, 1)
    X_train, y_train = create_dataset(data[:train_size])
    X_test, y_test = create_dataset(data[train_size - 60:])

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    preds_rescaled = scaler.inverse_transform(preds).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    date_range = df['Date'].iloc[train_size + 1 : train_size + 1 + len(y_test)]
    return preds_rescaled, y_true, date_range

# --- Run Selected Model ---
if st.sidebar.button("Run Prediction"):
    if model_name == "ARIMA":
        y_pred, y_true, x_dates = train_arima(train, test)

    elif model_name == "Prophet":
        y_pred, y_true, x_dates = train_prophet(df, train_size)

    elif model_name == "LSTM":
        y_pred, y_true, x_dates = train_lstm(df, train_size)

    # --- Plot Predictions ---
    st.subheader("📉 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_dates, y_true, label="Actual", color="red")
    ax.plot(x_dates, y_pred, label="LSTM Prediction" if model_name == "LSTM" else f"{model_name} Prediction", linestyle="--", color="blue")
    ax.set_title(f"{model_name} Prediction vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --- Evaluation ---
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    st.markdown("### 🧮 Model Evaluation Metrics")
    st.write(f"**MAE**: {mae:.2f}")
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")
