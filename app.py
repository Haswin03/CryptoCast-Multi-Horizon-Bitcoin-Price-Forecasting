import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import os
import yfinance as yf

st.set_page_config(page_title="CryptoCast Live Demo", layout="wide")

metrics = {
    "Simple RNN": {"1-Day": "2.40%", "3-Day": "3.42%", "7-Day": "4.54%"},
    "LSTM":       {"1-Day": "2.46%", "3-Day": "3.25%", "7-Day": "4.53%"},
    "1D-CNN":     {"1-Day": "4.59%", "3-Day": "4.09%", "7-Day": "6.21%"},
    "Transformer":{"1-Day": "7.87%", "3-Day": "8.66%", "7-Day": "9.09%"}
}

st.sidebar.header("Forecast Settings")
arch = st.sidebar.selectbox("Architecture", ["Simple RNN", "LSTM", "1D-CNN", "Transformer"])
horizon = st.sidebar.selectbox("Horizon", ["1-Day", "3-Day", "7-Day"])

arch_file_map = {
    "Simple RNN": "rnn",
    "LSTM": "lstm",
    "1D-CNN": "cnn",
    "Transformer": "transformer"
}
horizon_file_map = {
    "1-Day": "1d",
    "3-Day": "3d",
    "7-Day": "7d"
}

st.title("CryptoCast: Multi-Horizon Forecast Engine")
tab1, tab2 = st.tabs(["Live Inference", "Performance Dashboard"])

with tab1:
    st.subheader(f"Live Market Inference: {arch} ({horizon} Horizon)")
    
    st.info("Target Asset: **Bitcoin (BTC-USD)** — The models are trained for BTC volatility scales.")
    ticker = "BTC-USD"
    
    if st.button("Fetch Live Data & Generate Forecast"):
        with st.spinner("Fetching real-time Bitcoin data from Yahoo Finance and running neural networks..."):
            try:
                data = yf.download(ticker, period="6mo", interval="1d", progress=False)
                
                if len(data) < 60:
                    st.error(f"Not enough data found for {ticker}. The model requires at least 60 days of history.")
                else:
                    recent_data = data.tail(60)
                    prices_array = recent_data['Close'].values.reshape(-1, 1)
                    historical_dates = recent_data.index
                    
                    scaler_path = os.path.join('models', 'scaler.pkl')
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    model_name = f"{arch_file_map[arch]}_{horizon_file_map[horizon]}_model.keras"
                    model_path = os.path.join('models', model_name)
                    model = tf.keras.models.load_model(model_path)
                    
                    scaled_input = scaler.transform(prices_array).reshape(1, 60, 1)
                    pred_scaled = model.predict(scaled_input, verbose=0)
                    pred_usd = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                    
                    last_date = historical_dates[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(pred_usd))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical_dates, y=prices_array.flatten(), 
                        mode='lines', name="Recent History", line=dict(color='#3b82f6', width=2)
                    ))
                    
                    connect_x = [last_date] + list(future_dates)
                    connect_y = [prices_array[-1][0]] + list(pred_usd)
                    
                    fig.add_trace(go.Scatter(
                        x=connect_x, y=connect_y, 
                        mode='lines+markers', name="AI Forecast", 
                        line=dict(color='#f59e0b', width=3, dash='dot'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} Forecast Path ({arch} - {horizon})", 
                        xaxis_title="Date", 
                        yaxis_title="Price (USD)",
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("Forecast Calculated Successfully!")
                    
                    st.markdown("### Projected Values")
                    
                    forecast_df = pd.DataFrame({
                        "Date": [date.strftime('%b %d, %Y') for date in future_dates],
                        "Projected Price": [f"${price:,.2f}" for price in pred_usd]
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                        
            except FileNotFoundError as e:
                st.error(f"File Missing: Ensure your 'models' directory contains `{e.filename}`.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Dashboard
with tab2:
    st.subheader("Historical Model Metrics (MAPE)")
    st.markdown("This matrix tracks the Mean Absolute Percentage Error across all 12 model variants during the testing phase.")
    st.table(pd.DataFrame(metrics).T)