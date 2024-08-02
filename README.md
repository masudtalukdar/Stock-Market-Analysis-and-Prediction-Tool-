
# Stock Market Analysis

## Technical Report

### Overview

This project aims to provide a comprehensive toolkit for fetching historical stock data and performing various analyses and predictions using machine learning and statistical models. The toolkit includes functionalities for stock price prediction, time series forecasting, technical indicator analysis, volatility prediction, buy/sell signal classification, and Monte Carlo simulations.

### Project Structure

The project consists of two main scripts:
- `data_fetch.py` - Fetches historical stock data from the Alpha Vantage API and saves it to a CSV file.
- `main.py` - Performs various analyses and predictions on the fetched stock data.

### Dependencies

The following Python packages are required to run the scripts:
- pandas
- numpy
- scikit-learn
- statsmodels
- arch
- seaborn
- matplotlib
- alpha_vantage

You can install these packages using pip:

\`\`\`bash
pip install pandas numpy scikit-learn statsmodels arch seaborn matplotlib alpha_vantage
\`\`\`

### Data Fetching

\`data_fetch.py\` script fetches historical stock data from Alpha Vantage and saves it to a CSV file. The user is prompted to enter the stock symbol, and the data is fetched using the provided API key. The fetched data includes columns such as 'Date', 'Open', 'High', 'Close', and 'Volume'.

### Analysis and Prediction

\`main.py\` script provides multiple functions for analyzing and predicting stock prices. Users can choose the type of analysis they want to perform from a menu. The functionalities provided are as follows:

1. **Stock Price Prediction**
   - Uses Linear Regression to predict future stock prices based on historical data.
   - Evaluates model performance using Root Mean Squared Error (RMSE).

2. **Time Series Forecasting**
   - Utilizes ARIMA models for forecasting future stock prices.
   - Forecasts future stock prices with confidence intervals.

3. **Technical Indicator Analysis**
   - Computes and plots moving averages and the Relative Strength Index (RSI).
   - Helps in identifying overbought or oversold conditions.

4. **Volatility Prediction**
   - Uses GARCH models to predict stock price volatility.
   - Provides insights into the risk associated with the stock.

5. **Classification - Buy/Sell Signals**
   - Applies Random Forest classification to predict buy/sell signals.
   - Evaluates performance with a confusion matrix and classification report.

6. **Monte Carlo Simulation**
   - Simulates multiple possible future price paths based on historical returns.
   - Provides a visual representation of potential stock price movements over a specified period.

### Results

#### Stock Price Prediction
- Predicts stock prices for future dates based on historical closing prices.
- Performance is evaluated using RMSE, providing an estimate of prediction accuracy.

#### Time Series Forecasting
- Forecasts future stock prices with ARIMA models, considering historical price patterns.
- Plots forecasted prices along with confidence intervals.

#### Technical Indicator Analysis
- Plots 50-day moving average and RSI to analyze stock trends.
- Identifies overbought or oversold conditions, aiding in trading decisions.

#### Volatility Prediction
- Predicts future volatility using GARCH models, indicating potential price fluctuations.
- Provides a measure of risk for the stock.

#### Buy/Sell Signals Classification
- Classifies future stock price movements as buy or sell signals using Random Forest.
- Performance is evaluated using a confusion matrix and classification report, providing precision, recall, and F1-score metrics.

#### Monte Carlo Simulation
- Simulates multiple future price paths, providing a probabilistic view of future stock prices.
- Visualizes potential stock price movements over a specified period.

### Conclusion

This project provides a versatile toolkit for stock market analysis and prediction. By leveraging machine learning and statistical models, users can gain insights into future stock prices, trends, and volatility, and make informed trading decisions.

### Future Work

- Enhance model tuning and parameter optimization.
- Integrate additional technical indicators and machine learning models.
- Develop a user-friendly interface for non-technical users.
- Incorporate real-time data fetching and analysis.