📈 Hybrid XGBoost + LSTM Quantitative Trading Strategy

NIFTY 50 Universe | Indian Equity Markets

This repository implements a hybrid machine learning–based quantitative trading strategy for Indian equities.
The system combines XGBoost (cross-sectional alpha model) and LSTM (time-series model) to rank stocks weekly and construct a long-only, equal-weighted portfolio, benchmarked against the NIFTY 50 index.

The strategy is backtested on 2024 data and additionally validated through full-year paper trading in 2025 to assess real-time robustness.

📌 Strategy Summary

Universe: NSE equities (2017 NIFTY universe with survivorship handling)

Benchmark: NIFTY 50 Index (^NSEI)

Rebalancing Frequency: Weekly (Fridays)

Portfolio Type: Long-only

Stock Selection: Top-K ranked stocks

Weighting Scheme: Equal weight

🧠 Machine Learning Architecture
1️⃣ XGBoost – Cross-Sectional Alpha Model

Trained on all stocks pooled together

Predicts forward returns over a fixed horizon

Captures relative strength and factor-based alpha

Particularly effective for ranking stocks cross-sectionally

2️⃣ LSTM – Time-Series Model

Trained per stock using rolling sequences

Learns temporal dependencies in historical features

Lookback window: 60 trading days

Complements XGBoost by capturing dynamic market regimes

3️⃣ Ensemble Scoring

Final stock score is computed as a weighted ensemble:

Final Score = w_xgb × XGBoost Prediction + w_lstm × LSTM Prediction


Default weights:

XGBoost: 0.8

LSTM: 0.2

This weighting prioritizes cross-sectional alpha while retaining temporal context.

📊 Feature Engineering

Each stock is represented using a rich set of technical and statistical features:

🔁 Returns & Momentum

ret1, ret5, ret10

mom20

📉 Volatility

vol20, vol60

📈 Trend Indicators

sma10, sma20, sma50, sma200

⚠️ Risk & Range

True Range (tr)

Average True Range (atr14)

🚀 Momentum Indicators

rsi14

macd, macd_signal, macd_hist

📦 Volume-Based Features

Log-volume z-score (20-day)

5-day volume change

🎯 Target Variable
Forward Return = Close(t + horizon) / Close(t) − 1

⏱ Data Splits
Phase	Date Range
Training	2010-01-01 → 2021-12-31
Validation	2022-01-01 → 2023-12-31
Backtest (Test)	2024-01-01 → 2024-12-31

Strict chronological splitting ensures no look-ahead bias.

💼 Backtesting Framework

Weekly rebalancing on Fridays

Transaction Costs Applied:

Brokerage / fees: 10 bps

Slippage: 5 bps

Equal-weight allocation across selected stocks

Portfolio fully rebalanced each week

📐 Performance Metrics

The following metrics are computed and reported:

CAGR

Sharpe Ratio

Maximum Drawdown

Equity Curve Comparison: Strategy vs NIFTY 50

Monthly Return Table (2024)

📈 Outputs & Results

Strategy vs Benchmark equity curves

Monthly and cumulative return tables

Console-printed performance statistics

Trade-level logs (during paper trading)

🧪 Live Paper Trading Validation (2025)

To validate robustness beyond historical backtests:

Full-year paper trading conducted in 2025

Same:

Universe

Features

Models

Parameters

Transaction cost assumptions

No real capital deployed

Trades executed in a simulated live environment

Portfolio decisions generated in real time

This helped evaluate:

Live drawdowns

Turnover behavior

Stability under unseen market conditions

⚙️ Configuration

All strategy parameters are controlled via a centralized CFG dictionary, including:

Lookback window

Prediction horizon

Top-K stock selection

Model hyperparameters

Transaction cost assumptions

Ensemble weights

This design allows easy experimentation and reproducibility.

🚀 How to Run
1️⃣ Install Dependencies
pip install numpy pandas yfinance xgboost torch matplotlib

2️⃣ Run the Strategy
python main.py


CUDA is automatically used if available

Falls back to CPU otherwise

🧪 Reproducibility

Fixed random seeds for NumPy and PyTorch

Deterministic behavior where supported

Identical results reproducible across runs (subject to hardware differences)

📌 Notes & Limitations

Survivorship bias partially mitigated via symbol alias handling

No short selling

Corporate actions handled via raw OHLCV data (no auto-adjust)

Designed for research and educational purposes only

📜 Disclaimer

This project is not financial advice.
Past performance does not guarantee future results.
Use at your own risk.

👤 Authors

Developed by Jishnu, Rathan, and Samhitha
Hybrid ML-based quantitative equity strategy for Indian markets.

⭐ If you find this project useful, consider starring the repository.
