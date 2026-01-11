# Hybrid XGBoost + LSTM Quantitative Trading Strategy (NIFTY 50 Universe)

This repository implements a **hybrid machine learning trading strategy** for Indian equities using **XGBoost (cross-sectional model)** and **LSTM (time-series model)**. The strategy ranks stocks weekly and backtests a **top-k equal-weighted portfolio** against the **NIFTY 50 index** for the year 2024.

---

## 📌 Strategy Overview

* **Universe**: NSE stocks (2017 NIFTY universe with survivorship handling)
* **Index Benchmark**: NIFTY 50 (`^NSEI`)
* **Models Used**:

  * **XGBoost Regressor** – learns cross-sectional alpha from engineered features
  * **LSTM Regressor** – learns temporal patterns from rolling lookback windows
* **Final Score**: Weighted ensemble of XGBoost and LSTM predictions
* **Rebalancing**: Weekly (Fridays)
* **Portfolio Construction**:

  * Long-only
  * Top-K ranked stocks
  * Equal weight allocation

---

## 🧠 Machine Learning Architecture

### 1. XGBoost (Cross-Sectional Model)

* Trained on all stocks pooled together
* Predicts **forward returns over a fixed horizon**
* Captures relative stock strength across the universe

### 2. LSTM (Time-Series Model)

* Separate rolling sequences per stock
* Learns temporal dependencies from historical features
* Uses **lookback window of 60 trading days**

### 3. Ensemble Scoring

```text
Final Score = w_xgb × XGB_Prediction + w_lstm × LSTM_Prediction
```

Default weights:

* XGBoost: **0.8**
* LSTM: **0.2**

---

## 📊 Feature Engineering

The following technical and statistical features are computed per stock:

* Returns: `ret1`, `ret5`, `ret10`, `mom20`
* Volatility: `vol20`, `vol60`
* Trend: `sma10`, `sma20`, `sma50`, `sma200`
* Range & Risk: `tr`, `atr14`
* Momentum Indicators: `rsi14`, `macd`, `macd_sig`, `macd_hist`
* Volume-based:

  * Log-volume z-score (20-day)
  * 5-day volume change

Target Variable:

```text
Forward Return = Close(t + horizon) / Close(t) - 1
```

---

## ⏱ Data Splits

| Split           | Date Range              |
| --------------- | ----------------------- |
| Train           | 2010-01-01 → 2021-12-31 |
| Validation      | 2022-01-01 → 2023-12-31 |
| Test (Backtest) | 2024-01-01 → 2024-12-31 |

---

## 💼 Backtesting Logic

* Weekly rebalancing on **Fridays**
* Portfolio turnover cost applied:

  * Transaction cost: **10 bps**
  * Slippage: **5 bps**
* Equal-weight allocation across selected stocks

### Performance Metrics

* CAGR
* Sharpe Ratio
* Maximum Drawdown
* Benchmark comparison (NIFTY)

---

## 📈 Outputs

### Paper Trading (2025)

In addition to historical backtesting, this strategy was **paper traded for the entire year 2025** using the same logic, parameters, and rebalancing rules.

* No real capital was deployed

* Trades were executed in a simulated environment

* Portfolio decisions were generated live using model predictions

* This helped validate strategy robustness beyond the backtest period

* Equity curve comparison:

  * Strategy vs NIFTY

* Monthly return table (2024)

* Performance statistics printed to console

---

## ⚙️ Configuration

All strategy parameters are controlled via the `CFG` dictionary:

Key parameters include:

* Lookback window
* Prediction horizon
* Top-K stocks
* Model hyperparameters
* Cost assumptions

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas yfinance xgboost torch matplotlib
```

### 2. Run the Script

```bash
python main.py
```

(Ensure CUDA is available for GPU acceleration, otherwise CPU is used automatically.)

---

## 🧪 Reproducibility

* Random seeds fixed for NumPy and PyTorch
* Deterministic training behavior (where possible)

---

## 📌 Notes & Limitations

* Survivorship bias partially mitigated via symbol aliases
* No short selling
* Corporate actions handled via raw OHLCV (no auto-adjust)
* Designed for **research and educational purposes only**

---

## 🧪 Live Paper Trading Validation

* Full-year **paper trading conducted during 2025**
* Same universe, features, models, and transaction cost assumptions
* Used to observe real-time behavior, drawdowns, and turnover

---

## 📜 Disclaimer

This project is **not financial advice**. Past performance does not guarantee future results. Use at your own risk.

---

## 👤 Author

Developed by **Jishnu, Rathan and Samhitha**
Hybrid ML-based quantitative equity strategy for Indian markets.

---

⭐ If you find this useful, consider starring the repository.
