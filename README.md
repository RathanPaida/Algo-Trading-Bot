# Hybrid XGBoost + LSTM Quantitative Trading System

A quantitative equity ranking and backtesting system for Indian stocks that combines **XGBoost** and **LSTM** models to predict forward returns, rank stocks, construct portfolios, and evaluate trading performance against the NIFTY 50 benchmark.

The project is designed as a research-oriented end-to-end pipeline covering data collection, feature engineering, machine learning, portfolio construction, backtesting, risk analysis, and visualization.

---

## Overview

The system:

- Downloads historical market data using `yfinance`
- Engineers technical and statistical features
- Predicts future stock returns using XGBoost
- Models temporal behaviour using an LSTM
- Combines both predictions through a weighted ensemble
- Ranks stocks according to predicted forward returns
- Constructs a long-only portfolio from the highest-ranked stocks
- Rebalances the portfolio weekly
- Models transaction costs and slippage
- Applies momentum, volatility, sector exposure, and stop-loss constraints
- Evaluates the strategy on chronologically separated validation and test periods
- Performs a simulated 2025 paper evaluation
- Generates detailed performance reports, plots, and transaction records

---

## Model Architecture

### XGBoost

XGBoost is used to learn cross-sectional relationships between engineered market features and future stock returns.

At each point in time, the model compares characteristics across multiple stocks and estimates their expected forward returns.

### LSTM

The LSTM model captures time-dependent behaviour using rolling historical sequences.

Each sample contains:

```text
90 trading days of historical features
            ↓
          LSTM
            ↓
Predicted 5-day forward return
