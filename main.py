import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # Import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from arch import arch_model
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function for stock price prediction using Linear Regression
def stock_price_prediction(data):
    features = data[['Close']]
    data['Target'] = data['Close'].shift(-7)
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(train_data[['Close']], train_data['Target'])
    predictions = model.predict(data[['Close']])  # Predict on the entire dataset
    rmse = np.sqrt(mean_squared_error(data['Target'], predictions))
    print(f"Root Mean Squared Error: {rmse}")
    plt.plot(data.index, data['Target'], label='Actual Prices')
    plt.plot(data.index, predictions, label='Predicted Prices')
    plt.legend()
    plt.show()


# Function for time series forecasting using ARIMA
def time_series_forecasting(data):
    close_prices = data['Close']
    model = ARIMA(close_prices, order=(5, 1, 0))  # Example order, tune as needed

    try:
        results = model.fit()
    except ValueError as e:
        print(f"Error: {e}")
        return

    forecast_steps = 30  # Adjust as needed
    forecast = results.get_forecast(steps=forecast_steps)

    # Extract forecasted values and index
    forecast_values = forecast.predicted_mean
    forecast_index = pd.date_range(close_prices.index[-1], periods=forecast_steps + 1, freq='D')[1:]

    # Plot historical and forecasted values
    plt.plot(close_prices.index, close_prices, label='Historical Prices')
    plt.plot(forecast_index, forecast_values, label='Forecasted Prices', linestyle='--', color='orange')
    plt.fill_between(forecast_index, forecast.conf_int()['lower Close'], forecast.conf_int()['upper Close'], color='orange', alpha=0.2, label='Confidence Interval')

    plt.legend()
    plt.show()


# Function for technical indicator analysis
def technical_indicator_analysis(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    rs = average_gain / average_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    plt.plot(data.index, data['Close'], label='Close Prices')
    plt.plot(data.index, data['MA_50'], label='50-day Moving Average')
    plt.plot(data.index, data['RSI'], label='RSI')
    plt.legend()
    plt.show()

# Function for volatility prediction using GARCH
def volatility_prediction(data):
    # Drop NaN values in the 'Close' column
    data = data.dropna(subset=['Close'])

    # Calculate returns and drop NaN values
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna(subset=['Returns'])

    # Check for NaN or infinite values in 'Returns'
    if not np.isfinite(data['Returns']).all():
        raise ValueError("NaN or infinite values found in 'Returns'. Please clean your data.")

    model = arch_model(data['Returns'], vol='Garch', p=1, q=1)
    results = model.fit()
    forecast_volatility = results.conditional_volatility

    plt.plot(data.index, data['Returns'], label='Daily Returns')
    plt.plot(data.index, forecast_volatility, label='Forecasted Volatility')
    plt.legend()
    plt.show()


# Function for classification - Buy/Sell signals
def classification_buy_sell_signals(data):
    data['Target'] = (data['Close'].shift(-7) > data['Close']).astype(int)
    data.dropna(inplace=True)
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    model = RandomForestClassifier()
    model.fit(train_data[['Close']], train_data['Target'])
    predictions = model.predict(test_data[['Close']])

    # Plot confusion matrix as a heatmap
    cm = confusion_matrix(test_data['Target'], predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Sell', 'Buy'], yticklabels=['Sell', 'Buy'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot bar chart for precision, recall, and F1-score
    report = classification_report(test_data['Target'], predictions, output_dict=True)
    metrics = report['1']  # Considering '1' as the positive class, adjust if needed
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Classification Report Metrics')
    plt.show()


# Function for Monte Carlo Simulation
def monte_carlo_simulation(data, num_simulations, num_days):
    returns = data['Close'].pct_change().dropna()
    mean_daily_return = returns.mean()
    volatility = returns.std()

    simulation_results = pd.DataFrame()

    for i in range(num_simulations):
        prices = [data['Close'].iloc[-1]]

        for day in range(num_days):
            daily_return = np.random.normal(mean_daily_return, volatility)
            price = prices[-1] * (1 + daily_return)
            prices.append(price)

        simulation_results[f'Simulation_{i + 1}'] = prices

    return simulation_results

# Prompt the user to choose the analysis or visualization
print("Choose an option:")
print("1. Stock Price Prediction")
print("2. Time Series Forecasting")
print("3. Technical Indicator Analysis")
print("4. Volatility Prediction")
print("5. Classification - Buy/Sell Signals")
print("6. Monte Carlo Simulation")

choice = int(input("Enter the number of your choice: "))

# Fetch stock data
csv_filename = 'GOOG_stock_data.csv' # Replace with your actual CSV file
stock_data = pd.read_csv(csv_filename)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

# Perform the selected analysis or visualization
if choice == 1:
    stock_price_prediction(stock_data)
elif choice == 2:
    time_series_forecasting(stock_data)
elif choice == 3:
    technical_indicator_analysis(stock_data)
elif choice == 4:
    volatility_prediction(stock_data)
elif choice == 5:
    classification_buy_sell_signals(stock_data)
elif choice == 6:
    # Prompt the user for the number of simulations and days
    num_simulations = int(input("Enter the number of simulations: "))
    num_days = int(input("Enter the number of days to simulate: "))

    # Perform Monte Carlo Simulation
    simulation_data = monte_carlo_simulation(stock_data, num_simulations, num_days)

    # Plot the simulation results
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_data, linewidth=0.5)
    plt.title('Monte Carlo Simulation for Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.show()
else:
    print("Invalid choice. Please choose a number between 1 and 6.")
