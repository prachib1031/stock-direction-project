import streamlit as st
import yfinance as yf

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


st.title("Stock Direction Explorer")
st.write("Explore stock history and test next-day predictions.")

ticker = st.text_input("Enter a stock symbol", value="AAPL")
ticker = ticker.strip().upper()

if st.button("Load stock data"):
    if not ticker:
        st.warning("Enter a stock symbol first.")
    else:
        with st.spinner("Loading stock history..."):
            history = yf.Ticker(ticker).history(period="5y")

        if history.empty:
            st.warning("No data found. Try another stock symbol.")
        else:
            st.subheader("Price history")
            st.line_chart(history["Close"])

            history["Daily Return"] = history["Close"].pct_change()
            history["5-Day Return"] = history["Close"].pct_change(5)

            st.dataframe(
                history[["Close", "Daily Return", "5-Day Return"]].tail()
            )

            next_close = history["Close"].shift(-1)
            history["Target"] = (
                (next_close > history["Close"])
                .astype("Int64")
                .where(next_close.notna())
            )

            st.subheader("Historical answers")
            st.dataframe(history[["Close", "Target"]].tail())

            features = ["Daily Return", "5-Day Return"]
            model_data = history.dropna(subset=features + ["Target"])

            split = int(len(model_data) * 0.8)
            train = model_data.iloc[:split].copy()
            test = model_data.iloc[split:].copy()

            st.write("Training days:", len(train))
            st.write("Testing days:", len(test))

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000)
            )

            model.fit(train[features], train["Target"].astype(int))
            predictions = model.predict(test[features])

            accuracy = (
                predictions == test["Target"].to_numpy(dtype=int)
            ).mean()
            baseline = test["Target"].mean()

            st.metric("Model test accuracy", f"{accuracy:.1%}")
            st.metric("Always predict up accuracy", f"{baseline:.1%}")

            st.write("Model predicted up:", int((predictions == 1).sum()))
            st.write("Model predicted down or unchanged:", int((predictions == 0).sum()))            
            results = test[["Close", "Target"]].copy()
            results["Prediction"] = predictions
            results["Correct"] = results["Prediction"] == results["Target"]

            st.subheader("Last 10 test predictions")
            st.dataframe(results.tail(10))