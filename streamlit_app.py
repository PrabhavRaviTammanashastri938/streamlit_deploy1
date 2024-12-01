import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Function to fetch data from Yahoo Finance
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches stock data for the given ticker and date range.

    Parameters:
        tickers (list): List of stock tickers (only one ticker in this case).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with stock data for the selected ticker.
    """
    all_data = []
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.warning(f"No data available for {ticker} in the selected date range.")
            continue
        stock_data['Ticker'] = ticker
        all_data.append(stock_data)
    
    return pd.concat(all_data)

# List of available stock tickers (you can customize this or fetch dynamically if needed)
available_tickers = [
    "AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "NFLX", "META", "NVDA", "IBM", "AMD", 
    "INTC", "SPY", "BABA", "V", "JPM", "DIS", "BA", "GS", "PYPL", "TWTR", "SQ", 
    "UBER", "LYFT", "INTU", "CRM", "WMT", "KO", "PEP", "MCD"
]

# Streamlit App
st.title("Prabhav - HFT Stock Dashboard ")

# Sidebar inputs for date range and single ticker selection
st.sidebar.header("Configuration")

# Selectbox for a single ticker
selected_ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    options=available_tickers,  # Predefined list of tickers
    index=available_tickers.index("AAPL")  # Default to AAPL
)

# Highlight selected ticker
st.sidebar.markdown("### Selected Ticker:")
st.sidebar.markdown(f"- **{selected_ticker}**")

# Date range input
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2023-12-31"))

# Plot type selection
plot_type = st.sidebar.radio("Select Plot Type", ["None", "Scatter Plot", "Line Plot", "Pie Chart", "All Plots"])

# Fetch data button
if st.sidebar.button("Fetch Stock Data"):
    # Fetch stock data for the selected ticker
    try:
        if not selected_ticker:
            st.warning("Please select a ticker.")
        else:
            stock_data = fetch_stock_data([selected_ticker], start_date, end_date)
            st.success(f"Stock data for {selected_ticker} fetched successfully!")
            
            # Show data in Streamlit
            st.write("### Stock Data Preview", stock_data.head())
            
            # Generate selected plot
            if plot_type == "Scatter Plot":
                st.subheader("Scatter Plot: Closing Price vs Volume")
                fig, ax = plt.subplots()
                ax.scatter(stock_data['Volume'], stock_data['Close'], color='blue')
                ax.set_xlabel('Volume')
                ax.set_ylabel('Closing Price')
                ax.set_title(f'Scatter Plot: {selected_ticker} Closing Price vs Volume')
                st.pyplot(fig)

            elif plot_type == "Line Plot":
                st.subheader("Line Plot: Closing Price over Time")
                fig, ax = plt.subplots()
                ax.plot(stock_data.index, stock_data['Close'], color='green')
                ax.set_xlabel('Date')
                ax.set_ylabel('Closing Price')
                ax.set_title(f'Line Plot: {selected_ticker} Closing Price over Time')
                st.pyplot(fig)

            elif plot_type == "Pie Chart":
                st.subheader("Pie Chart: Distribution of Daily Price Changes")
                stock_data['Price Change'] = stock_data['Close'].diff().fillna(0)
                positive_changes = len(stock_data[stock_data['Price Change'] > 0])
                negative_changes = len(stock_data[stock_data['Price Change'] < 0])
                neutral_changes = len(stock_data[stock_data['Price Change'] == 0])

                fig, ax = plt.subplots()
                ax.pie([positive_changes, negative_changes, neutral_changes], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

            elif plot_type == "All Plots":
                # Scatter Plot
                st.subheader("Scatter Plot: Closing Price vs Volume")
                fig, ax = plt.subplots()
                ax.scatter(stock_data['Volume'], stock_data['Close'], color='blue')
                ax.set_xlabel('Volume')
                ax.set_ylabel('Closing Price')
                ax.set_title(f'Scatter Plot: {selected_ticker} Closing Price vs Volume')
                st.pyplot(fig)

                # Line Plot
                st.subheader("Line Plot: Closing Price over Time")
                fig, ax = plt.subplots()
                ax.plot(stock_data.index, stock_data['Close'], color='green')
                ax.set_xlabel('Date')
                ax.set_ylabel('Closing Price')
                ax.set_title(f'Line Plot: {selected_ticker} Closing Price over Time')
                st.pyplot(fig)

                # Pie Chart
                st.subheader("Pie Chart: Distribution of Daily Price Changes")
                stock_data['Price Change'] = stock_data['Close'].diff().fillna(0)
                positive_changes = len(stock_data[stock_data['Price Change'] > 0])
                negative_changes = len(stock_data[stock_data['Price Change'] < 0])
                neutral_changes = len(stock_data[stock_data['Price Change'] == 0])

                fig, ax = plt.subplots()
                ax.pie([positive_changes, negative_changes, neutral_changes], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

            # Allow user to download the data
            csv = stock_data.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_ticker}_stock_data.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")

def feature_engineering(stock_data):
    """
    Performs feature engineering on the stock data and calculates necessary features.
    
    Args:
    - stock_data: Raw stock data including 'Close' and 'Volume'

    Returns:
    - stock_data: Updated DataFrame with additional features and labels
    """
    # Calculate daily returns (price change)
    stock_data['Daily Return'] = stock_data['Close'].pct_change()

    # Calculate rolling 30-day volatility (standard deviation of daily returns)
    stock_data['Volatility'] = stock_data['Daily Return'].rolling(window=30).std()

    # Calculate 30-day moving average of volume
    stock_data['Avg Volume'] = stock_data['Volume'].rolling(window=30).mean()

    # Rate of change in volume (percentage change in volume from previous day)
    stock_data['Volume Change'] = stock_data['Volume'].pct_change()

    # Label high-frequency trading (HFT) events (price movement > 2% and volume > average)
    stock_data['HFT'] = (stock_data['Daily Return'].abs() > 0.02) & (stock_data['Volume'] > stock_data['Avg Volume'])

    # Drop rows with missing values (due to rolling calculations)
    stock_data = stock_data.dropna()

    return stock_data

if st.sidebar.button("Fetch HFT Data"):
    # Fetch the raw stock data
    stock_data = fetch_stock_data(selected_ticker, start_date, end_date)
    
    # Perform feature engineering (calculate volatility, volume, etc.)
    stock_data_with_features = feature_engineering(stock_data)
    
    # Display the processed stock data with HFT features
    st.write(f"### Stock Data with HFT Features for {selected_ticker}")
    st.write(stock_data_with_features.head())
