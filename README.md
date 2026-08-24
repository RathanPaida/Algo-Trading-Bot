## Hybrid XGBoost + LSTM Quantitative Trading System

A research-oriented quantitative equity ranking and backtesting system for Indian equities.

The project combines a cross-sectional XGBoost model with a time-series LSTM model, generates weekly stock rankings, constructs a long-only portfolio, and evaluates performance against the NIFTY 50 benchmark.

## Overview

The pipeline includes:

- Historical OHLCV data collection using Yahoo Finance
- Technical and statistical feature engineering
- XGBoost-based cross-sectional return prediction
- LSTM-based time-series prediction
- Weighted ensemble ranking
- Weekly portfolio construction
- Transaction-cost and slippage modeling
- Validation, testing and 2025 simulated evaluation
- Performance analysis and visualization

## Model Architecture

### XGBoost

The XGBoost model is trained across stocks using engineered technical and statistical features to predict forward returns.

### LSTM

The LSTM model learns temporal structure from rolling historical feature windows.

### Ensemble

The final ranking score is formed using:

```text
Final Score =
0.60 × XGBoost Prediction
+
0.40 × LSTM Prediction
