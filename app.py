import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

st.title("Financial Data Analysis and Machine Learning")

# Sidebar for options
st.sidebar.title("Options")

# Tabs
tabs = ["Indicators", "Machine Learning"]
selected_tab = st.sidebar.selectbox("Select tab:", tabs)

if selected_tab == "Indicators":
    st.header("Technical Indicators")

    # Stock selection
    ticker = st.sidebar.text_input("Enter stock ticker:", "TSLA")

    # Date range selection
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))

    # Fetch data from yfinance
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
        else:
            # Calculate indicators
            data['MA50'] = data['Close'].rolling(window=50).mean()

            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean().abs()
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))

            high_14, low_14 = data['High'].rolling(window=14).max(), data['Low'].rolling(window=14).min()
            k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()

            # Plotting
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)

            axes[0].plot(data['Close'], label='Close Price')
            axes[0].plot(data['MA50'], label='50-day MA')
            axes[0].legend()
            axes[0].set_ylabel('Price')

            axes[1].plot(rsi, label='RSI')
            axes[1].legend()
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)

            axes[2].plot(k_percent, label='%K')
            axes[2].plot(d_percent, label='%D')
            axes[2].legend()
            axes[2].set_ylabel('Stochastic Oscillator')
            axes[2].set_ylim(0, 100)

            axes[3].plot(data['Volume'], label='Volume')
            axes[3].legend()
            axes[3].set_ylabel('Volume')

            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")

elif selected_tab == "Machine Learning":
    st.header("Machine Learning Models")

    # Stock selection
    ticker = st.sidebar.text_input("Enter stock ticker for ML:", "TSLA")

    # Date range selection
    start_date = st.sidebar.date_input("Start date for ML", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End date for ML", value=pd.to_datetime("today"))

    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
        else:
            # Data preprocessing
            data = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

            # Split data into training and testing sets
            train_size = int(len(data) * 0.8)
            train_data, test_data = data[:train_size], data[train_size:]

            # Prepare data for LSTM
            def create_dataset(dataset, time_step=1):
                X, y = [], []
                for i in range(len(dataset) - time_step - 1):
                    a = dataset[i:(i + time_step), 0]
                    X.append(a)
                    y.append(dataset[i + time_step, 0])
                return np.array(X), np.array(y)

            # LSTM Model
            st.subheader("LSTM Model")
            time_step = 60
            X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
            X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)
            X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
            X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
            lstm_model.add(LSTM(50, return_sequences=False))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')

            lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)

            lstm_predictions = lstm_model.predict(X_test_lstm)
            lstm_predictions = scaler.inverse_transform(lstm_predictions)
            y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

            lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_predictions))

            # Predict next day's price for LSTM
            last_data_lstm = X_test_lstm[-1].reshape(1, time_step, 1)
            next_day_lstm_prediction = lstm_model.predict(last_data_lstm)
            next_day_lstm_prediction = scaler.inverse_transform(next_day_lstm_prediction)

            # Plot LSTM predictions
            fig_lstm, ax_lstm = plt.subplots(figsize=(12, 6))
            ax_lstm.plot(y_test_lstm, label="Actual")
            ax_lstm.plot(lstm_predictions, label="LSTM Predicted")
            ax_lstm.legend()
            ax_lstm.set_title("LSTM Model")
            st.pyplot(fig_lstm)

            st.write(f'LSTM Root Mean Squared Error:', lstm_rmse)
            st.write(f'LSTM Next Day Prediction: {next_day_lstm_prediction[0][0]}')

            # SVM Model
            st.subheader("SVM Model")
            X_train_svm, y_train_svm = train_data[:-1], train_data[1:]
            X_test_svm, y_test_svm = test_data[:-1], test_data[1:]

            svm_model = SVR(kernel='linear')
            svm_model.fit(X_train_svm, y_train_svm.ravel())

            svm_predictions = svm_model.predict(X_test_svm)
            svm_predictions = scaler.inverse_transform(svm_predictions.reshape(-1, 1))
            y_test_svm = scaler.inverse_transform(y_test_svm.reshape(-1, 1))

            svm_rmse = np.sqrt(mean_squared_error(y_test_svm, svm_predictions))

            # Predict next day's price for SVM
            last_data_svm = test_data[-1].reshape(1, -1)
            next_day_svm_prediction = svm_model.predict(last_data_svm)
            next_day_svm_prediction = scaler.inverse_transform(next_day_svm_prediction.reshape(1, -1))

            # Plot SVM predictions
            fig_svm, ax_svm = plt.subplots(figsize=(12, 6))
            ax_svm.plot(y_test_svm, label="Actual")
            ax_svm.plot(svm_predictions, label="SVM Predicted")
            ax_svm.legend()
            ax_svm.set_title("SVM Model")
            st.pyplot(fig_svm)

            st.write(f'SVM Root Mean Squared Error:', svm_rmse)
            st.write(f'SVM Next Day Prediction: {next_day_svm_prediction[0][0]}')

            # LightGBM Model
            st.subheader("LightGBM Model")
            X_train_lgbm, y_train_lgbm = train_data[:-1], train_data[1:]
            X_test_lgbm, y_test_lgbm = test_data[:-1], test_data[1:]

            lgbm_model = LGBMRegressor()
            lgbm_model.fit(X_train_lgbm, y_train_lgbm.ravel())

            lgbm_predictions = lgbm_model.predict(X_test_lgbm)
            lgbm_predictions = scaler.inverse_transform(lgbm_predictions.reshape(-1, 1))
            y_test_lgbm = scaler.inverse_transform(y_test_lgbm.reshape(-1, 1))

            lgbm_rmse = np.sqrt(mean_squared_error(y_test_lgbm, lgbm_predictions))

            # Predict next day's price for LightGBM
            last_data_lgbm = test_data[-1].reshape(1, -1)
            next_day_lgbm_prediction = lgbm_model.predict(last_data_lgbm)
            next_day_lgbm_prediction = scaler.inverse_transform(next_day_lgbm_prediction.reshape(1, -1))

            # Plot LightGBM predictions
            fig_lgbm, ax_lgbm = plt.subplots(figsize=(12, 6))
            ax_lgbm.plot(y_test_lgbm, label="Actual")
            ax_lgbm.plot(lgbm_predictions, label="LightGBM Predicted")
            ax_lgbm.legend()
            ax_lgbm.set_title("LightGBM Model")
            st.pyplot(fig_lgbm)

            st.write(f'LightGBM Root Mean Squared Error:', lgbm_rmse)
            st.write(f'LightGBM Next Day Prediction: {next_day_lgbm_prediction[0][0]}')

    except Exception as e:
        st.error(f"An error occurred: {e}")