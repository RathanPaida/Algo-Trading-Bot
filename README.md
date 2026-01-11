# 📈 Hybrid XGBoost + LSTM Quantitative Trading Strategy  
### *NIFTY 50 Universe | Indian Equity Markets*  
<br>

<p align="center">
  <strong>
    A hybrid machine learning–driven quantitative equity strategy combining<br>
    cross-sectional and time-series models for Indian stock markets
  </strong>
</p>

<br>

---

## ✨ Overview  
<br>

This repository implements a **hybrid ML-based quantitative trading system** for Indian equities.<br>
The strategy combines:

- **XGBoost** → Cross-sectional alpha generation  
- **LSTM** → Time-series pattern learning  

<br>

Stocks are **ranked weekly**, and a **long-only, equal-weighted portfolio** is constructed and<br>
benchmarked against the **NIFTY 50 index**.

<br>

✔️ Backtested on **2024**  
<br>
✔️ Validated via **full-year paper trading in 2025**

<br>

---

## 📌 Strategy Summary  
<br>

Universe : NSE equities (2017 NIFTY universe)
Benchmark : NIFTY 50 (^NSEI)
Rebalancing : Weekly (Fridays)
Portfolio Type : Long-only
Selection : Top-K ranked stocks
Weighting : Equal-weight

yaml
Copy code

<br>

---

## 🧠 Machine Learning Architecture  
<br>

### 🔹 1. XGBoost – Cross-Sectional Alpha Model  
<br>

- Trained on **all stocks pooled together**  
- Predicts **forward returns**  
- Learns **relative stock strength**  
- Effective for factor-style ranking  

<br>

---

### 🔹 2. LSTM – Time-Series Model  
<br>

- Trained **per stock**  
- Rolling **60-day lookback window**  
- Captures **temporal dependencies & regime shifts**  
- Complements XGBoost’s static cross-sectional view  

<br>

---

### 🔹 3. Ensemble Scoring  
<br>

Final ranking score:

Final Score = w_xgb × XGB_Prediction + w_lstm × LSTM_Prediction

markdown
Copy code

<br>

**Default Weights**  
<br>
- 🟦 XGBoost: **0.8**  
- 🟥 LSTM: **0.2**

<br>

---

## 📊 Feature Engineering  
<br>

Each stock is represented using the following features:

<br>

### 🔁 Returns & Momentum  
- `ret1`, `ret5`, `ret10`  
- `mom20`

<br>

### 📉 Volatility  
- `vol20`, `vol60`

<br>

### 📈 Trend Indicators  
- `sma10`, `sma20`, `sma50`, `sma200`

<br>

### ⚠️ Risk & Range  
- True Range (`tr`)  
- Average True Range (`atr14`)

<br>

### 🚀 Momentum Indicators  
- `rsi14`  
- `macd`, `macd_signal`, `macd_hist`

<br>

### 📦 Volume-Based Features  
- Log-volume z-score (20-day)  
- 5-day volume change

<br>

---

### 🎯 Target Variable  
<br>

Forward Return = Close(t + horizon) / Close(t) − 1

yaml
Copy code

<br>

---

## ⏱ Data Splits  
<br>

| Phase | Date Range |
|------|-----------|
| 🧪 Training | 2010-01-01 → 2021-12-31 |
| 🔍 Validation | 2022-01-01 → 2023-12-31 |
| 📈 Backtest | 2024-01-01 → 2024-12-31 |

<br>

✔️ Strict chronological splitting  
<br>
✔️ No look-ahead bias  

<br>

---

## 💼 Backtesting Framework  
<br>

- 🔄 Weekly rebalancing (Fridays)  
- ⚖️ Equal-weight allocation  
- 💸 Transaction costs applied:  
  - Brokerage: **10 bps**  
  - Slippage: **5 bps**  
- 🔁 Full portfolio rebalance each cycle  

<br>

---

## 📐 Performance Metrics  
<br>

- 📈 CAGR  
- 📊 Sharpe Ratio  
- 📉 Maximum Drawdown  
- 🆚 Strategy vs NIFTY 50 equity curve  
- 🗓 Monthly return table (2024)

<br>

---

## 📈 Outputs  
<br>

- Equity curve plots  
- Monthly & cumulative returns  
- Console-printed performance statistics  
- Trade-level logs (paper trading)

<br>

---

## 🧪 Live Paper Trading Validation (2025)  
<br>

- 📅 Full-year paper trading conducted in 2025  
- 🧠 Same universe, features, models, and parameters  
- 💸 Same transaction cost assumptions  
- 🚫 No real capital deployed  
- ⚙️ Trades executed in simulated live conditions  

<br>

This phase helped evaluate:<br>
- Drawdown behavior  
- Turnover stability  
- Real-time model robustness  

<br>

---

## ⚙️ Configuration  
<br>

All strategy parameters are controlled via a centralized `CFG` dictionary:

- Lookback window  
- Prediction horizon  
- Top-K selection  
- Model hyperparameters  
- Transaction costs  
- Ensemble weights  

<br>

✔️ Easy experimentation  
<br>
✔️ Fully reproducible  

<br>

## 🚀 How to Run  
<br>

### 1️⃣ Install Dependencies  

pip install numpy pandas yfinance xgboost torch matplotlib
<br>

### 2️⃣ Run the Strategy
bash
Copy code
python main.py
<br>
⚡ CUDA is used automatically if available
<br>
🖥 CPU fallback is enabled otherwise
<br>

🧪 Reproducibility
<br>
Fixed random seeds for NumPy and PyTorch
<br>
Deterministic behavior where supported
<br>
Consistent results across runs (hardware permitting)

<br>
📌 Notes & Limitations
<br>
⚠️ Survivorship bias partially mitigated
<br>
🚫 No short selling
<br>
📊 Raw OHLCV data used (no auto-adjustment)
<br>
🎓 Intended for research and educational purposes only
<br>
<br>
📜 Disclaimer
<br>
This project is not financial advice.
Past performance does not guarantee future results.
Use at your own risk.

<br>
<br>
<br>
👤 Authors
<br>
Jishnu · Rathan · Samhitha
<br>


Hybrid ML-based quantitative equity strategy for Indian markets

<br>
⭐ If you find this repository useful, consider starring it.
