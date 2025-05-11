import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to fetch data from Yahoo Finance
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

def load_sample_data():
    data = pd.read_csv('HFT_dataset.csv', parse_dates=['Date'])
    return data

stock_data = load_sample_data()

available_tickers = [
    'MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE',
    'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',
    'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'WMT', 'GOOGL', 'AMZN', 'AABA'
]

st.title("HFT Stock Dashboard")

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

with col3:
    st.subheader("Donut Chart: CAT (Caterpillar)")
    cat_data = stock_data[stock_data['Name'] == 'CAT']
    cat_data['Price Change'] = cat_data['Close'].diff().fillna(0)
    counts = [len(cat_data[cat_data['Price Change'] > 0]),
              len(cat_data[cat_data['Price Change'] < 0]),
              len(cat_data[cat_data['Price Change'] == 0])]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%',
           startangle=90, wedgeprops=dict(width=0.3))
    ax.axis('equal')
    st.pyplot(fig)

with col4:
    st.subheader("Multi-Featured Scatter Plot: AMZN (Amazon)")
    amzn_data = stock_data[stock_data['Name'] == 'AMZN']
    fig = px.scatter(
        amzn_data, x='Volume', y='Close', size='High', color='Open',
        hover_data=['Low', 'Date'], title="Scatter Plot: AMZN"
    )
    st.plotly_chart(fig, use_container_width=True)

st.header("Fetch your Stock or HFT Dataset")
st.sidebar.header("Configuration")

selected_ticker = st.sidebar.selectbox("Select Stock Ticker", options=available_tickers, index=available_tickers.index("AAPL"))
st.sidebar.markdown(f"### Selected Ticker:\n- **{selected_ticker}**")

start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2023-12-31"))
plot_type = st.sidebar.radio("Select Plot Type", ["None", "Scatter Plot", "Line Plot", "Pie Chart", "All Plots"])

# === Feature 1: Dataset Only ===
if st.sidebar.button("Generate Dataset Only"):
    filter_toggle = st.sidebar.checkbox("Filter by Date Range", value=True)
    df = stock_data[stock_data['Name'] == selected_ticker]
    if filter_toggle:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    st.subheader(f"Dataset: {selected_ticker} {'(Filtered)' if filter_toggle else '(Full)'}")
    st.write(df)

# === Feature 2: Line Chart for Multiple Companies ===
if st.sidebar.button("Generate Multi-Ticker Line Chart"):
    selected_companies = st.sidebar.multiselect("Select Companies for Line Chart", options=available_tickers, default=['AAPL', 'AMZN', 'MSFT'])
    filtered_data = stock_data[stock_data['Name'].isin(selected_companies)]
    fig = px.line(filtered_data, x='Date', y='Close', color='Name', title="Closing Prices for Selected Companies")
    st.plotly_chart(fig, use_container_width=True)

# === Original Feature: Fetch & Plot ===
if st.sidebar.button("Fetch Stock Data"):
    try:
        stock_data = fetch_stock_data([selected_ticker], start_date, end_date)
        st.success(f"Stock data for {selected_ticker} fetched successfully!")
        st.write("### Stock Data Preview", stock_data.head())

        if plot_type == "Scatter Plot":
            st.subheader("Scatter Plot: Closing Price vs Volume")
            fig, ax = plt.subplots()
            ax.scatter(stock_data['Volume'], stock_data['Close'], color='blue')
            ax.set_xlabel('Volume'); ax.set_ylabel('Closing Price')
            st.pyplot(fig)

        elif plot_type == "Line Plot":
            st.subheader("Line Plot: Closing Price over Time")
            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data['Close'], color='green')
            st.pyplot(fig)

        elif plot_type == "Pie Chart":
            st.subheader("Pie Chart: Distribution of Daily Price Changes")
            stock_data['Price Change'] = stock_data['Close'].diff().fillna(0)
            counts = [len(stock_data[stock_data['Price Change'] > 0]),
                      len(stock_data[stock_data['Price Change'] < 0]),
                      len(stock_data[stock_data['Price Change'] == 0])]
            fig, ax = plt.subplots()
            ax.pie(counts, labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        elif plot_type == "All Plots":
            st.subheader("Scatter Plot")
            fig, ax = plt.subplots()
            ax.scatter(stock_data['Volume'], stock_data['Close'], color='blue')
            st.pyplot(fig)

            st.subheader("Line Plot")
            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data['Close'], color='green')
            st.pyplot(fig)

            st.subheader("Pie Chart")
            stock_data['Price Change'] = stock_data['Close'].diff().fillna(0)
            counts = [len(stock_data[stock_data['Price Change'] > 0]),
                      len(stock_data[stock_data['Price Change'] < 0]),
                      len(stock_data[stock_data['Price Change'] == 0])]
            fig, ax = plt.subplots()
            ax.pie(counts, labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)

        csv = stock_data.to_csv().encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_ticker}_stock_data.csv", "text/csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# HFT Check Logic remains unchanged
def load_hft_data(ticker):
    file_path = f"HFT_dataset.csv"
    try:
        hft_data = pd.read_csv(file_path, index_col=["Date", "Name"], parse_dates=True)
        return hft_data.xs(ticker, level="Name")
    except FileNotFoundError:
        st.error(f"HFT data file not found.")
        return None

def is_suitable_for_hft(hft_data):
    recent_data = hft_data.tail(5)
    avg_volume = recent_data['Volume'].mean()
    avg_volatility = recent_data['Volatility'].mean()
    avg_rsi = recent_data['RSI'].mean()
    avg_price_change = recent_data['Price_Change'].mean()

    if avg_volume > 500000000 and avg_volatility > 10000 and avg_rsi < 55 and avg_price_change > 4000:
        return "Yes"
    return "No"

if st.sidebar.button("Check HFT Status"):
    hft_data = load_hft_data(selected_ticker)
    if hft_data is not None:
        st.write(f"### HFT Data for {selected_ticker}")
        st.write(hft_data.head())

        fig, ax = plt.subplots()
        ax.plot(hft_data.index, hft_data['Close'], label='Close')
        ax.plot(hft_data.index, hft_data['MA_5'], label='MA_5')
        ax.plot(hft_data.index, hft_data['MA_20'], label='MA_20')
        ax.set_title("Close & Moving Averages")
        ax.legend()
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.plot(hft_data.index, hft_data['Volatility'], color='orange')
        ax.set_title("Volatility Over Time")
        st.pyplot(fig)

        recent = hft_data.tail(5)
        st.table({
            "Parameter": ["Average Volume", "Average Volatility", "Average RSI", "Average Price Change", "Average Momentum"],
            "Value": [f"{recent['Volume'].mean():.2f}", f"{recent['Volatility'].mean():.2f}",
                      f"{recent['RSI'].mean():.2f}", f"{recent['Price_Change'].mean():.2f}",
                      f"{recent['Momentum_5'].mean():.2f}"]
        })

        suitable = is_suitable_for_hft(hft_data)
        st.write(f"### Is {selected_ticker} Suitable for HFT?")
        st.write(f"The company is **{'suitable' if suitable == 'Yes' else 'not suitable'}** for High-Frequency Trading.")
