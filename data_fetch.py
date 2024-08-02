import pandas as pd
from alpha_vantage.timeseries import TimeSeries

# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(symbol, api_key, interval='daily', output_size='full'):
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')

        # Adjust the interval and output_size parameters based on your needs
        data, meta_data = ts.get_daily(symbol=symbol, outputsize=output_size)

        # Check if the request was successful
        if 'Error Message' in meta_data:
            print(f"Error: {meta_data['Error Message']}")
            return None

        # Extract 'Date', 'Open', 'High', 'Close', and 'Volume' columns
        stock_data = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
        stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        stock_data['Date'] = pd.to_datetime(stock_data.index)

        return stock_data[['Date', 'Open', 'High', 'Close', 'Volume']]

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
alpha_vantage_api_key = 'MS4MTKAMPNUGDNR5'

# Get stock symbol from the user
stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")

# Fetch stock data
stock_data = fetch_stock_data(stock_symbol, alpha_vantage_api_key)

# Check if data was successfully fetched before proceeding
if stock_data is not None:
    # Save data to a CSV file
    csv_filename = f"{stock_symbol}_stock_data.csv"
    stock_data.to_csv(csv_filename, index=False)  # Exclude index column

    # Display the first few rows of the fetched data
    print(stock_data.head())

    # Display a message indicating that the data has been saved to a CSV file
    print(f"Stock data saved to {csv_filename}")
