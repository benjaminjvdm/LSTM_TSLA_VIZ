import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load TSLA data from yfinance and preprocess data
tsla_data = yf.download("TSLA", start="2015-01-01")

# Global date range filter
start_date = st.sidebar.date_input("Start date", value=tsla_data.index.min())
end_date = st.sidebar.date_input("End date", value=tsla_data.index.max())

# Preprocess data
data = tsla_data.filter(['Close'])
dataset = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

def visualize_stock_price_history():
    # Filter data based on selected dates
    tsla_data_filtered = tsla_data.loc[start_date:end_date]

    # Relative Strength Index (RSI)
    delta = tsla_data_filtered['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().abs()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # Stochastic Oscillator
    high_14, low_14 = tsla_data_filtered['High'].rolling(window=14).max(), tsla_data_filtered['Low'].rolling(window=14).min()
    k_percent = 100 * ((tsla_data_filtered['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(window=3).mean()

    # Plot stock price history, RSI, and Stochastic Oscillator
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16,12), sharex=True)
    axes[0].plot(tsla_data_filtered.index, tsla_data_filtered['Close'])
    axes[0].set_title('Tesla (TSLA) Stock Price History')
    axes[0].set_ylabel('Closing price ($)')
    axes[1].plot(tsla_data_filtered.index, rsi)
    axes[1].set_title('Relative Strength Index (RSI)')
    axes[2].plot(tsla_data_filtered.index, k_percent, label='%K')
    axes[2].plot(tsla_data_filtered.index, d_percent, label='%D')
    axes[2].set_title('Stochastic Oscillator')
    axes[2].legend()
    st.pyplot()

# Build and train the LSTM model
def build_and_train_model():
    # Create training dataset
    training_data_len = int(len(dataset) * 0.8)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Create testing dataset
    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot forecasted stock prices
    plt.figure(figsize=(16,8))
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions
    plt.title("Predicted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Closing price ($)")
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
    st.pyplot()
    
def main():
    st.title("Tesla (TSLA) Stock Price Analysis")

    # Sidebar options
    option = st.sidebar.selectbox(
        'Choose an option:',
        ('Visualize stock price history', 'Train and evaluate LSTM model')
    )

    if option == 'Visualize stock price history':
        visualize_stock_price_history()
    elif option == 'Train and evaluate LSTM model':
        build_and_train_model()

if __name__ == "__main__":
    main()
