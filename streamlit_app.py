import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px





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
          'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE',

      'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',

      'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'WMT', 'GOOGL', 'AMZN', 'AABA'

]

# Streamlit App
st.title("Prabhav - HFT Stock Dashboard ")

# Sidebar inputs for date range and single ticker selection

def load_sample_data():
    # Placeholder for your stock dataset loading logic
    # Ensure 'Name' column exists to filter by company
    data = pd.read_csv('HFT_dataset.csv', parse_dates=['Date'])
    return data

stock_data = load_sample_data()

col1, col2 = st.columns(2)

with col1:
        st.subheader("Layered Bar Chart: AAPL (Apple)")
        aapl_data = stock_data[stock_data['Name'] == 'AAPL']
        fig = px.bar(
            aapl_data, 
            x='Date', 
            y=['Open', 'Close', 'High', 'Low'], 
            title="Layered Bar Chart: AAPL Stock Prices",
            labels={"value": "Price", "variable": "Price Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Waterfall Chart: MMM (3M Company)")
    mmm_data = stock_data[stock_data['Name'] == 'MMM'].copy()
    mmm_data['Daily_Change'] = mmm_data['Close'] - mmm_data['Open']
    mmm_data['Cumulative_Change'] = mmm_data['Daily_Change'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.bar(
            mmm_data['Date'], 
            mmm_data['Cumulative_Change'], 
            color=(mmm_data['Daily_Change'] > 0).map({True: 'green', False: 'red'})
        )
    ax.set_title("MMM Cumulative Daily Changes")
    ax.set_ylabel("Cumulative Change")
    ax.set_xlabel("Date")
    st.pyplot(fig)

col3, col4 = st.columns(2)

    # Pie Chart for CAT
with col3:
    st.subheader("Donut Chart: CAT (Caterpillar)")
    cat_data = stock_data[stock_data['Name'] == 'CAT']
    cat_data['Price Change'] = cat_data['Close'].diff().fillna(0)
    positive_changes = len(cat_data[cat_data['Price Change'] > 0])
    negative_changes = len(cat_data[cat_data['Price Change'] < 0])
    neutral_changes = len(cat_data[cat_data['Price Change'] == 0])

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
            [positive_changes, negative_changes, neutral_changes], 
            labels=['Positive', 'Negative', 'Neutral'], 
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.3)  # This creates the donut shape
        )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Multi-Featured Scatter Plot for AMZN
    with col4:
        st.subheader("Multi-Featured Scatter Plot: AMZN (Amazon)")
        amzn_data = stock_data[stock_data['Name'] == 'AMZN']
        fig = px.scatter(
            amzn_data,
            x='Volume',
            y='Close',
            size='High',
            color='Open',
            hover_data=['Low', 'Date'],
            title="Scatter Plot: AMZN"
        )
        st.plotly_chart(fig, use_container_width=True)


st.header("Fetch your Stock or HFT Dataset")
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
plot_type = st.sidebar.radio("Select Plot Type", ["None","Scatter Plot", "Line Plot", "Pie Chart", "All Plots"])

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


# Function for loading the pre-processed HFT dataset for multiple tickers
def load_hft_data(ticker):
    
    file_path = f"HFT_dataset.csv"  
    
    try:

        hft_data = pd.read_csv(file_path, index_col=["Date", "Name"], parse_dates=True)
        

        ticker_data = hft_data.xs(ticker, level="Name")  # 'Name' corresponds to the ticker column

        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Log_Return', 'Volatility', 
                            'MA_5', 'MA_20', 'EMA_20', 'RSI', 'Volume_MA_5', 'VWAP', 'Momentum_5', 'Target']

        if not all(col in ticker_data.columns for col in expected_columns):
            st.warning(f"Some columns are missing in the HFT data for {ticker}.")
        
        return ticker_data
    except FileNotFoundError:
        st.error(f"File for HFT data not found!")
        return None

# Function to determine if a company is suitable for HFT based on engineered features
# Function to determine if a company is suitable for HFT based on engineered features
def is_suitable_for_hft(hft_data):
    """
    Determines whether the company is suitable for High-Frequency Trading (HFT)
    based on its engineered features like Volume, Volatility, Price Change, RSI, etc.
    
    Args:
    - hft_data: DataFrame containing the engineered HFT features for a given company.
    
    Returns:
    - suitability: A string indicating whether the company is suitable for HFT ("Yes" or "No")
    """
    # Updated thresholds for the features to make more companies eligible for HFT
    volume_threshold = 500  # Lower threshold for high volume (previously 1 million)
    volatility_threshold = 0.01  # Lower volatility threshold (previously 0.02, now 1%)
    rsi_threshold = 45  # Increase RSI threshold to 45 (previously 50)
    price_change_threshold = 0.005  # Lower price change threshold (previously 1%, now 0.5%)

    # Check the average of relevant columns over a specific period (e.g., the last 5 days)
    recent_data = hft_data.tail(5)

    avg_volume = recent_data['Volume'].mean()
    avg_volatility = recent_data['Volatility'].mean()
    avg_rsi = recent_data['RSI'].mean()
    avg_price_change = recent_data['Price_Change'].mean()

    # Criteria for suitability (adjusted for more flexibility)
    if avg_volume > volume_threshold and avg_volatility > volatility_threshold and avg_rsi < rsi_threshold and avg_price_change > price_change_threshold:
        suitability = "Yes"
    else:
        suitability = "No"

    return suitability



if st.sidebar.button("Fetch HFT Data"):

    hft_data = load_hft_data(selected_ticker)
    
    if hft_data is not None:
        # Display the HFT dataset for the selected ticker
        st.write(f"### HFT Data with Engineered Features for {selected_ticker}")
        st.write(hft_data.head())

        # Optionally, plot some of the features for the selected ticker
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hft_data.index, hft_data['Close'], label='Close Price', color='blue', alpha=0.7)
        ax.plot(hft_data.index, hft_data['MA_5'], label='5-Day Moving Average', color='green', alpha=0.7)
        ax.plot(hft_data.index, hft_data['MA_20'], label='20-Day Moving Average', color='red', alpha=0.7)
        ax.set_title(f"{selected_ticker} Closing Price and Moving Averages")
        ax.legend()
        st.pyplot(fig)
        
        # Optionally, plot Volatility or other indicators
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hft_data.index, hft_data['Volatility'], label='Volatility', color='orange')
        ax.set_title(f"{selected_ticker} Volatility Over Time")
        ax.legend()
        st.pyplot(fig)

        suitability = is_suitable_for_hft(hft_data)
        st.write(f"### Is {selected_ticker} Suitable for HFT?")
        st.write(f"The company is {'suitable' if suitability == 'Yes' else 'not suitable'} for High-Frequency Trading.")

