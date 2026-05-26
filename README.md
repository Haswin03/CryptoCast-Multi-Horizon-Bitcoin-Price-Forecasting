# 🚀 CryptoCast: Multi-Horizon Bitcoin Price Forecasting Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

### 🔗 [Live Demo - Try CryptoCast Here](https://cryptocast-multi-horizon-bitcoin-price-forecasting.streamlit.app/))

---

## 📌 Project Overview
**CryptoCast** is an end-to-end deep learning framework designed to forecast the future price path of Bitcoin (BTC-USD). Financial time-series data is notoriously noisy and non-stationary. To tackle this, this project benchmarks **4 distinct neural network architectures** across **3 temporal horizons** (1-Day, 3-Day, and 7-Day windows), resulting in a unified ecosystem of 12 highly specialized models.

The goal is not just to predict a single price, but to analyze how different AI architectures digest short-term vs. long-term market momentum using a rolling 60-day historical lookback window.

---

## 🧠 The Deep Learning Architectures
To evaluate structural advantages, four fundamentally different mathematical approaches were built and trained from scratch:

1. **Simple RNN:** Evaluates raw, sequential step-by-step memory. Found to be highly reactive and dominant in short-term (24-hour) immediate predictions.
2. **LSTM (Long Short-Term Memory):** Utilizes complex gating mechanisms to prevent vanishing gradients. This serves as the system's structural champion, retaining longer-term memory and dominating the multi-day forecast windows.
3. **1D-CNN (Convolutional Neural Network):** Applies parallel sliding-window filters to extract localized volatility patterns. Proven to be the most computationally efficient architecture in the suite.
4. **Transformer (Self-Attention):** Uses multi-head attention to view the sequence simultaneously rather than sequentially. Dimensional expansion layers were implemented to prevent univariate gradient collapse.

---

## 📊 Performance Matrix (Testing Phase)
The models were evaluated using Mean Absolute Percentage Error (MAPE) against unseen test data. The lower the percentage, the higher the accuracy.

| Model Architecture | 1-Day Horizon | 3-Day Horizon | 7-Day Horizon |
| :--- | :--- | :--- | :--- |
| **Simple RNN** | **2.40%** 🏆 | 3.42% | 4.54% |
| **LSTM** | 2.46% | **3.25%** 🏆 | **4.53%** 🏆 |
| **1D-CNN** | 4.59% | 4.09% | 6.21% |
| **Transformer** | 7.87% | 8.66% | 9.09% |

*Key Takeaway: The Simple RNN captures immediate next-day momentum best, while the LSTM handles multi-step sequential degradation with the highest stability.*

---

## ⚙️ The Inference Pipeline (How it Works)
This project moves beyond Jupyter Notebooks into a production-grade inference engine:
1. **Live Data Fetching:** Uses the `yfinance` API to silently fetch the most recent 60 days of real-world BTC data.
2. **Preprocessing:** Data is scaled using the exact `MinMaxScaler` fitted during the initial training phase to prevent data leakage or skew.
3. **Tensor Reshaping:** Converts the 1D array into a 3D Tensor `(1, 60, 1)` required by Keras models.
4. **Dynamic Routing:** Instantly loads the specific `.keras` binary chosen by the user in the UI.
5. **Inverse Transformation:** Converts the model's normalized (0 to 1) output back into real-world USD figures.
6. **Visualization:** Overlays the predicted future vector onto the historical trajectory using Plotly.

---

## 📁 Repository Structure
```text
cryptocast/
│
├── models/
│   ├── scaler.pkl                  # Fitted MinMaxScaler
│   ├── rnn_1d_model.keras          # Compiled model binaries
│   ├── lstm_3d_model.keras         
│   └── ... (All 12 architecture variations)
│
├── notebooks/
│   └── CryptoCast_Training.ipynb   # Complete R&D, EDA, and Training Pipeline
│
├── app.py                          # Streamlit UI & Inference Engine
├── requirements.txt                # Dependency list
└── README.md                       # Project documentation
