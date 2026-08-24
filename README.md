# Hybrid XGBoost + LSTM Quantitative Trading System

A research-oriented quantitative equity ranking and backtesting system for Indian equities.

The project combines a cross-sectional XGBoost model with a time-series LSTM model to estimate forward stock returns, rank securities, construct a long-only portfolio, and evaluate the strategy against the NIFTY 50 benchmark.

## Overview

The pipeline includes:

- Historical OHLCV data collection using Yahoo Finance
- Technical and statistical feature engineering
- XGBoost-based cross-sectional return prediction
- LSTM-based time-series prediction
- Weighted model ensembling
- Weekly stock ranking and portfolio construction
- Volatility-based position sizing
- Momentum filtering
- Sector exposure constraints
- Stop-loss handling
- Transaction cost and slippage modelling
- Chronological validation and testing
- 2025 simulated paper evaluation
- Performance analysis and visualization

## Model Architecture

### XGBoost

The XGBoost model learns cross-sectional relationships between engineered market features and future returns across multiple stocks.

### LSTM

The LSTM processes rolling 90-day feature sequences to capture temporal behaviour in individual stocks.

### Ensemble

The final prediction score combines both models:

```text
Final Score =
0.60 × XGBoost Prediction
+
0.40 × LSTM Prediction
