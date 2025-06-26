import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
from textblob import TextBlob
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Market Analyzer AI Agent")

# Sidebar
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter stock ticker (e.g. AAPL, MSFT):", "AAPL")
period = st.sidebar.selectbox("Select period", ["1mo", "3mo", "6mo", "1y", "2y"], index=0)
interval = st.sidebar.selectbox("Data interval", ["1d", "1wk", "1mo"], index=0)

# Get stock data
data = yf.download(ticker, period=period, interval=interval)
if data.empty:
    st.error("No data found. Please check the ticker symbol.")
    st.stop()

# Compute indicators
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = ta.rsi(data['Close'], length=14)

# Compute MACD
macd_df = ta.macd(data['Close'], fast=12, slow=26, signal=9)
if macd_df is not None and not macd_df.empty:
    macd_df.columns = ['MACD_Line', 'MACD_Signal', 'MACD_Hist']
    data = pd.concat([data, macd_df], axis=1)
    macd_available = True
else:
    macd_available = False

# Fixed OBV computation (no pandas_ta)
obv_available = False
if 'Close' in data.columns and 'Volume' in data.columns:
    try:
        close = data['Close'].values
        volume = data['Volume'].values
        obv = [0]  # Initialize OBV

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv.append(obv[-1] + volume[i])
            elif close[i] < close[i - 1]:
                obv.append(obv[-1] - volume[i])
            else:
                obv.append(obv[-1])

        data['OBV'] = obv
        obv_available = True
    except Exception as e:
        st.warning(f"OBV could not be computed: {e}")
else:
    st.warning("Required columns for OBV calculation missing.")

# Sentiment Analysis (placeholder using ticker string)
sentiment = TextBlob(ticker).sentiment.polarity
if sentiment > 0.1:
    sentiment_label = "Positive 🙂"
elif sentiment < -0.1:
    sentiment_label = "Negative 🙁"
else:
    sentiment_label = "Neutral 😐"

# Investment Suggestion
latest_close = float(data['Close'].iloc[-1])
latest_sma = data['SMA20'].iloc[-1]
if pd.notna(latest_sma):
    diff = latest_close - latest_sma
    threshold = latest_sma * 0.01
    if diff > threshold:
        suggestion = ("BUY 📈", "green")
    elif diff < -threshold:
        suggestion = ("SELL 📉", "red")
    else:
        suggestion = ("HOLD 🤝", "gray")
else:
    suggestion = ("Not enough data for suggestion", "orange")

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Closing Price & 20-day SMA")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label="Close Price")
    ax.plot(data.index, data['SMA20'], label="20-Day SMA", linestyle="--")
    ax.legend()
    ax.set_ylabel("Price (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Investment suggestion block
    if suggestion[0] == "BUY 📈":
        st.markdown(
            """
            <div style='
                background-color: #e6ffe6;
                padding: 1em;
                border-radius: 10px;
                border: 2px solid green;
                text-align: center;
                animation: blink 1s infinite;
            '>
                <h2 style='color: green; font-weight: bold;'>💹 BUY</h2>
            </div>
            <style>
                @keyframes blink {
                    50% { opacity: 0.3; }
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    elif suggestion[0] == "SELL 📉":
        st.markdown(
            """
            <div style='
                background-color: #ffe6e6;
                padding: 1em;
                border-radius: 10px;
                border: 2px solid red;
                text-align: center;
            '>
                <h2 style='color: red; font-weight: bold;'>🔻 SELL</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='
                background-color: #f0f0f0;
                padding: 1em;
                border-radius: 10px;
                text-align: center;
            '>
                <h2 style='color: gray; font-weight: bold;'>🤝 HOLD</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info("**SMA20** is the average of the last 20 closing prices. It smooths short-term volatility.")

with col2:
    # RSI Chart
    st.markdown("### 📈 Relative Strength Index (RSI)")
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 2.5))
    ax_rsi.plot(data.index, data['RSI'], color="purple")
    ax_rsi.axhline(70, color="red", linestyle="--", label="Overbought")
    ax_rsi.axhline(30, color="green", linestyle="--", label="Oversold")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend()
    st.pyplot(fig_rsi)
    st.info("**RSI** measures momentum. Above 70: overbought (possible SELL). Below 30: oversold (possible BUY).")

    # MACD Chart
    st.markdown("### 📉 MACD (Moving Average Convergence Divergence)")
    if macd_available:
        fig_macd, ax_macd = plt.subplots(figsize=(10, 3))
        ax_macd.plot(data.index, data['MACD_Line'], label="MACD Line", color="blue")
        ax_macd.plot(data.index, data['MACD_Signal'], label="Signal Line", color="red")
        ax_macd.bar(data.index, data['MACD_Hist'], label="Histogram", color="gray")
        ax_macd.legend()
        st.pyplot(fig_macd)
        st.info("**MACD** helps spot trend reversals. MACD > Signal = potential BUY. MACD < Signal = potential SELL.")
    else:
        st.warning("MACD data not available or failed to compute.")

    # OBV Chart
    st.markdown("### 📊 On-Balance Volume (OBV)")
    if obv_available:
        fig_obv, ax_obv = plt.subplots(figsize=(10, 2.5))
        ax_obv.plot(data.index, data['OBV'], label="OBV", color="orange")
        ax_obv.legend()
        st.pyplot(fig_obv)
        st.info("**OBV** measures buying and selling pressure using volume flow.")
    else:
        st.warning("OBV data not available or failed to compute.")

# Sentiment
st.markdown("---")
st.markdown(f"### 📰 Sentiment Analysis (Ticker Name): {sentiment_label}")
st.caption("Note: Replace this with real news sentiment API for accurate sentiment.")

# Footer
st.markdown("---")
st.caption("📘 This is an educational tool. No financial advice is provided.")
