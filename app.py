import streamlit as st
from predictor import load_data, train_and_predict

st.set_page_config(page_title="📈 Stock Predictor", layout="centered")

st.title("📊 Multi-Stock Close Price Predictor")
st.write("Upload your stock dataset and predict closing prices for selected tickers.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    tickers = df['Ticker'].unique().tolist()

    selected_ticker = st.selectbox("Select a stock ticker", tickers)

    if st.button("Predict"):
        with st.spinner(f"Training model for {selected_ticker}..."):
            metrics, fig, error = train_and_predict(df, selected_ticker)

            if error:
                st.warning(error)
            else:
                st.success("Prediction complete!")
                st.write("### 📈 Prediction Metrics")
                st.write(metrics)
                st.pyplot(fig)
else:
    st.info("👆 Upload a CSV file to begin.")