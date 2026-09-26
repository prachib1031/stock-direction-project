import streamlit as st
import yfinance as yf

st.title("Stock Direction Explorer")
st.write("Explore stock history and test next-day predictions.")

ticker = st.text_input("Enter a stock symbol", value="AAPL")
ticker = ticker.strip().upper()

if st.button("Load stock data"):
    if not ticker:
        st.warning("Enter a stock symbol first.")
    else:
        with st.spinner("Loading stock history..."):
            history = yf.Ticker(ticker).history(period="1y")

        if history.empty:
            st.warning("No data found. Try another stock symbol.")
        else:
            st.subheader("Price history")
            st.line_chart(history["Close"])
            st.dataframe(history.tail())