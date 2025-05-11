import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load data
@st.cache_data
def load_sample_data():
    return pd.read_csv('HFT_dataset.csv', parse_dates=['Date'])

def fetch_stock_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.warning(f"No data available for {ticker} in the selected date range.")
            continue
        stock_data['Ticker'] = ticker
        all_data.append(stock_data)
    return pd.concat(all_data)

def load_hft_data(ticker):
    try:
        hft_data = pd.read_csv("HFT_dataset.csv", index_col=["Date", "Name"], parse_dates=True)
        ticker_data = hft_data.xs(ticker, level="Name")
        return ticker_data
    except FileNotFoundError:
        st.error(f"HFT data not found.")
        return None

def is_suitable_for_hft(hft_data):
    volume_threshold = 500000000
    volatility_threshold = 10000
    rsi_threshold = 55
    price_change_threshold = 4000
    recent_data = hft_data.tail(5)
    avg_volume = recent_data['Volume'].mean()
    avg_volatility = recent_data['Volatility'].mean()
    avg_rsi = recent_data['RSI'].mean()
    avg_price_change = recent_data['Price_Change'].mean()
    if avg_volume > volume_threshold and avg_volatility > volatility_threshold and avg_rsi < rsi_threshold and avg_price_change > price_change_threshold:
        return "Yes"
    return "No"

# Available tickers
available_tickers = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE',
    'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',
    'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'WMT', 'GOOGL', 'AMZN', 'AABA'
]

# Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Generate Dataset", "Line Chart Comparison"])

# Page 1 - Main Dashboard
if page == "Main Dashboard":
    st.title("HFT Stock Dashboard")
    stock_data = load_sample_data()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Layered Bar Chart: AAPL (Apple)")
        aapl_data = stock_data[stock_data['Name'] == 'AAPL']
        fig = px.bar(
            aapl_data, x='Date', y=['Open', 'Close', 'High', 'Low'],
            title="AAPL Stock Prices",
            labels={"value": "Price", "variable": "Price Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Waterfall Chart: MMM (3M Company)")
        mmm_data = stock_data[stock_data['Name'] == 'MMM'].copy()
        mmm_data['Daily_Change'] = mmm_data['Close'] - mmm_data['Open']
        mmm_data['Cumulative_Change'] = mmm_data['Daily_Change'].cumsum()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(mmm_data['Date'], mmm_data['Cumulative_Change'],
               color=(mmm_data['Daily_Change'] > 0).map({True: 'green', False: 'red'}))
        ax.set_title("MMM Cumulative Daily Changes")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Donut Chart: CAT (Caterpillar)")
        cat_data = stock_data[stock_data['Name'] == 'CAT']
        cat_data['Price Change'] = cat_data['Close'].diff().fillna(0)
        pos, neg, neu = len(cat_data[cat_data['Price Change'] > 0]), len(cat_data[cat_data['Price Change'] < 0]), len(cat_data[cat_data['Price Change'] == 0])
        fig, ax = plt.subplots()
        ax.pie([pos, neg, neu], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%',
               startangle=90, wedgeprops=dict(width=0.3))
        ax.axis('equal')
        st.pyplot(fig)

    with col4:
        st.subheader("Scatter Plot: AMZN (Amazon)")
        amzn_data = stock_data[stock_data['Name'] == 'AMZN']
        fig = px.scatter(
            amzn_data, x='Volume', y='Close', size='High', color='Open',
            hover_data=['Low', 'Date'], title="Scatter Plot: AMZN"
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 2 - Generate Dataset (with toggle for date filtering)
elif page == "Generate Dataset":
    st.title("Generate Dataset from Yahoo Finance")

    selected_ticker = st.selectbox("Select Ticker", available_tickers, index=available_tickers.index("AAPL"))
    enable_date_filter = st.toggle("Enable Date Filter")

    if enable_date_filter:
        start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
    else:
        start_date = "2023-01-01"
        end_date = "2023-12-31"

    if st.button("Generate Dataset"):
        data = fetch_stock_data([selected_ticker], start_date, end_date)
        st.success("Dataset fetched successfully!")
        st.dataframe(data.head())
        st.download_button("Download CSV", data.to_csv().encode(), f"{selected_ticker}_data.csv", mime='text/csv')

# Page 3 - Line Chart Comparison
elif page == "Line Chart Comparison":
    st.title("Line Chart Comparison of Multiple Companies")
    stock_data = load_sample_data()

    selected_companies = st.multiselect("Select Companies", available_tickers, default=["AAPL", "MSFT", "GOOGL"])
    metric = st.selectbox("Select Metric", ["Close", "Open", "Volume"])

    if not selected_companies:
        st.warning("Please select at least one company.")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        for company in selected_companies:
            company_data = stock_data[stock_data['Name'] == company]
            ax.plot(company_data['Date'], company_data[metric], label=company)
        ax.set_title(f"{metric} Price Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.legend()
        st.pyplot(fig)
