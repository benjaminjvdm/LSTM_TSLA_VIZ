# Financial Data Analysis and Machine Learning

This Streamlit application provides tools for financial data analysis and machine learning-based stock price prediction.

## Features

-   **Technical Indicators:**
    -      Retrieves historical stock data from Yahoo Finance.
    -      Calculates and visualizes common technical indicators, including:
        -      50-day Moving Average (MA50)
        -      Relative Strength Index (RSI)
        -      Stochastic Oscillator (%K and %D)
        -      Volume
    -      Allows users to select a stock ticker and date range.
-   **Machine Learning Models:**
    -      Implements machine learning models for stock price prediction:
        -      Long Short-Term Memory (LSTM) neural network
        -      Support Vector Machine (SVM)
        -   Light Gradient Boosting Machine (LightGBM)
    -      Preprocesses data using MinMaxScaler.
    -      Splits data into training and testing sets.
    -      Trains and evaluates the models using Root Mean Squared Error (RMSE).
    -   Displays the prediction for the next days closing price.
    -      Visualizes the predicted vs. actual stock prices.
    -   Allows users to select a stock ticker and date range.

## Dependencies

-   streamlit
-   yfinance
-   pandas
-   matplotlib
-   scikit-learn (sklearn)
-   numpy
-   tensorflow (keras)
-   lightgbm

## How to Run

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Save the Python script:**

    Save the provided Python code as a `.py` file (e.g., `financial_analysis.py`).

3.  **Run the Streamlit app:**

    ```bash
    streamlit run financial_analysis.py
    ```

4.  **Usage:**
    -   The application will open in your web browser.
    -   Use the sidebar to select between "Indicators" and "Machine Learning" tabs.
    -   Enter the desired stock ticker and date range in the sidebar.
    -   The selected indicators or machine learning predictions will be displayed.

## Code Explanation

-   **yfinance:** Used to download historical stock data.
-   **pandas:** Used for data manipulation and analysis.
-   **matplotlib:** Used for data visualization.
-   **scikit-learn:** Used for data preprocessing (MinMaxScaler) and machine learning models (SVR).
-   **numpy:** Used for numerical operations.
-   **tensorflow (keras):** Used for building and training the LSTM neural network.
-   **lightgbm:** Used for the LightGBM regressor model.
-   **streamlit:** Used to create the interactive web application.

-   The application is divided into two main sections: "Indicators" and "Machine Learning," accessible through tabs in the sidebar.
-   The "Indicators" section calculates and displays common technical indicators.
-   The "Machine Learning" section trains and evaluates LSTM, SVM, and LightGBM models for stock price prediction.
-   Error handling is implemented to catch and display any exceptions that may occur during data retrieval or model training.
