import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn2
import json
import geopandas as gpd
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

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

# Preprocess HFT Data for India Map
@st.cache_data
def preprocess_hft_data_for_map(hft_data):
    state_attribute = hft_data.groupby('State')['Volume'].sum().reset_index()
    return state_attribute

# Venn Diagram Function
def plot_venn(hft_data):
    vol_set = set(hft_data[hft_data['Volume'] > 1000000].index)
    vola_set = set(hft_data[hft_data['Volatility'] > 0.02].index)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    venn2([vol_set, vola_set], set_labels=('Volume > 1M', 'Volatility > 2%'))
    st.pyplot(fig)

# Semi-Circle Chart Function
def plot_semi_circle_chart(hft_data):
    volume_sum = hft_data['Volume'].sum()

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=volume_sum,
        delta={'reference': 1000000},
        gauge={'shape': "semi"},
        title={'text': "Total Volume in HFT Dataset"}
    ))

    st.plotly_chart(fig)

# India Map Function
def plot_india_map(hft_data):
    india_geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/india-states.geojson"
    india_geojson = json.loads(open(india_geojson_url).read())

    state_data_map = dict(zip(hft_data['State'], hft_data['Volume']))

    for feature in india_geojson['features']:
        state_name = feature['properties']['name']
        if state_name in state_data_map:
            feature['properties']['hft_volume'] = state_data_map[state_name]
        else:
            feature['properties']['hft_volume'] = 0

    fig = px.choropleth(
        geojson=india_geojson,
        locations=[feature['properties']['name'] for feature in india_geojson['features']],
        color=[feature['properties']['hft_volume'] for feature in india_geojson['features']],
        hover_name=[feature['properties']['name'] for feature in india_geojson['features']],
        hover_data=['hft_volume'],
        color_continuous_scale="Viridis",
        labels={"hft_volume": "HFT Volume by State"},
        title="HFT Volume Across Indian States"
    )
    fig.update_geos(fitbounds="locations")
    st.plotly_chart(fig)

# Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Generate Dataset", "Line Chart Comparison","Check HFT Status", "Chatbot"])

# Main Dashboard
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

# Generate Dataset Page
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

# Line Chart Comparison Page
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

elif page == "Check HFT Status":
    st.title("Check HFT Suitability Status")

    stock_data = load_sample_data()
    selected_ticker = st.selectbox("Select Company", available_tickers)

    ticker_data = stock_data[stock_data['Name'] == selected_ticker]

    if ticker_data.empty:
        st.warning("No data found for the selected company.")
    else:
        # Prepare necessary columns
        hft_df = ticker_data.copy()
        hft_df['Volatility'] = hft_df['High'] - hft_df['Low']
        hft_df['RSI'] = (100 - (100 / (1 + hft_df['Close'].pct_change().rolling(14).mean())))  # Approx RSI
        hft_df['Price_Change'] = hft_df['Close'].diff()

        result = is_suitable_for_hft(hft_df)
        st.subheader(f"HFT Suitability for {selected_ticker}:")
        if result == "Yes":
            st.success("Yes, suitable for HFT 🚀")
        else:
            st.error("No, not suitable for HFT ❌")

        st.markdown("### Last 5 Days Used for Evaluation")
        st.dataframe(hft_df.tail(5)[['Volume', 'Volatility', 'RSI', 'Price_Change']])

elif page == "Chatbot":
    import openai

    st.title("📢 Ask the Stock Chatbot")

    # Initialize OpenAI client with API key
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask me anything about stocks or HFT...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant with expertise in stock trading and high-frequency trading."},
                    *st.session_state.chat_history
                ]
            )
            bot_response = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

            with st.chat_message("assistant"):
                st.markdown(bot_response)


