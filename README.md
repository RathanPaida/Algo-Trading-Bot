# Hybrid XGBoost + LSTM Quantitative Trading System

An end-to-end quantitative equity research and backtesting system for Indian equities.

The project combines **XGBoost** and a **Long Short-Term Memory (LSTM)** network to predict short-horizon stock returns, rank securities, construct a long-only portfolio, and evaluate its performance against the **NIFTY 50** benchmark.

The pipeline covers the complete workflow from market-data collection and feature engineering to model training, portfolio construction, transaction-cost-aware backtesting, and performance analysis.

---

## Overview

The system:

- Downloads historical OHLCV market data using Yahoo Finance
- Builds technical, momentum, volatility, trend, volume, and candlestick features
- Predicts 5-day forward returns using XGBoost
- Learns temporal patterns from 90-day sequences using an LSTM
- Combines both models using a weighted ensemble
- Ranks stocks according to predicted forward returns
- Selects the highest-ranked stocks for a long-only portfolio
- Rebalances the portfolio weekly
- Uses inverse-volatility-based position sizing
- Applies momentum and volatility filters
- Limits sector concentration
- Models transaction costs and slippage
- Supports stop-loss logic
- Evaluates performance using chronological validation and testing periods
- Runs a separate simulated 2025 evaluation
- Generates detailed strategy, model, trade, and risk analytics

---

## System Architecture

```text
Historical Market Data
        |
        v
Data Cleaning & Normalization
        |
        v
Feature Engineering
        |
        v
5-Day Forward Return Target
        |
        +-----------------------+
        |                       |
        v                       v
    XGBoost                   LSTM
Cross-Sectional          Time-Series Model
    Model                (90-Day Sequences)
        |                       |
        +-----------+-----------+
                    |
                    v
             Weighted Ensemble
          60% XGBoost + 40% LSTM
                    |
                    v
                Ranking
                    |
                    v
          Portfolio Construction
                    |
                    v
        Risk & Exposure Controls
                    |
                    v
            Weekly Rebalancing
                    |
                    v
      Transaction Costs + Slippage
                    |
                    v
                Backtest
                    |
                    v
         Performance Evaluation
```

---

## Dataset and Market Universe

The system uses historical price and volume data downloaded through `yfinance`.

The current universe is based on a selected set of NIFTY 50 equities, while:

```text
^NSEI
```

is used as the benchmark index.

Historical data is requested from:

```text
2010-01-01 to 2025-12-31
```

A minimum amount of historical data is required before a stock is included in the modelling pipeline.

---

## Feature Engineering

The system currently generates **39 features** for every stock.

### Returns and Momentum

```text
ret1
ret3
ret5
ret10
ret20
mom20
mom60
roc10
roc20
```

### Volatility

```text
vol5
vol20
vol60
vol_ratio
```

### Moving Averages and Trend

```text
sma10
sma20
sma50
sma200
sma_cross
```

### True Range and ATR

```text
tr
atr14
atr_z
```

### RSI

```text
rsi7
rsi14
rsi21
```

### MACD

```text
macd
macd_sig
macd_hist
macd_cross
```

### Bollinger Bands

```text
bb_upper
bb_lower
bb_width
bb_position
```

### Volume Features

```text
logvol_z20
vol_chg5
vol_sma_ratio
```

### Price and Candlestick Structure

```text
high_low_ratio
close_open
upper_shadow
lower_shadow
```

The prediction target is the stock's forward return over the next:

```text
5 trading days
```

---

# Models

## 1. XGBoost

XGBoost learns cross-sectional nonlinear relationships between engineered features and future stock returns.

The current model uses:

```text
Estimators:          2000
Learning rate:       0.015
Maximum depth:       7
Subsample:           0.85
Column sample:       0.85
L2 regularization:   1.5
L1 regularization:   0.5
Objective:           Squared-error regression
```

The model is trained on observations up to the configured training cutoff.

---

## 2. LSTM

The LSTM captures temporal structure that cannot be represented by a single cross-sectional feature vector.

For each stock, the network receives:

```text
90 consecutive trading days
        |
        v
39 features per day
        |
        v
       LSTM
        |
        v
Predicted 5-day forward return
```

Current configuration:

| Parameter | Value |
|---|---:|
| Lookback | 90 days |
| Hidden size | 96 |
| Batch size | 512 |
| Epochs | 15 |
| Learning rate | 5e-4 |
| Dropout | 0.20 |

CUDA is used automatically when available. Otherwise, the model runs on CPU.

---

## 3. Ensemble

The final prediction score combines both models:

```text
Final Score
    =
0.60 x XGBoost Prediction
    +
0.40 x LSTM Prediction
```

Stocks are ranked in descending order according to this score.

The ensemble is intended to combine:

- XGBoost's ability to model nonlinear cross-sectional relationships
- LSTM's ability to capture temporal behaviour

---

# Portfolio Construction

The strategy constructs a long-only portfolio from the highest-ranked securities.

Current configuration:

| Parameter | Value |
|---|---:|
| Portfolio size | Top 15 stocks |
| Rebalance weekday | Friday |
| Rebalance threshold | 15% |
| Minimum momentum | 2% |
| Maximum volatility | 50% |
| Maximum sector weight | 35% |
| Stop loss | -15% |
| Transaction cost | 8 bps |
| Slippage | 3 bps |

---

## Ranking

For every rebalance date:

1. Model predictions are generated.
2. The XGBoost and LSTM predictions are combined.
3. Stocks are ranked by the ensemble score.
4. Portfolio filters are applied.
5. The highest-ranked eligible stocks are selected.

---

## Volatility-Based Position Sizing

Instead of always assigning equal weights, the strategy can use inverse-volatility-based position sizing.

Lower-volatility securities receive relatively larger portfolio weights, while higher-volatility securities receive smaller allocations.

This is intended to reduce excessive concentration of portfolio risk in volatile stocks.

---

## Momentum Filter

The strategy can require stocks to satisfy a minimum momentum threshold before entering the portfolio.

Current threshold:

```text
2%
```

---

## Volatility Filter

Stocks with estimated volatility above:

```text
50%
```

can be excluded from selection.

---

## Sector Exposure Constraint

The portfolio limits exposure to any individual sector to approximately:

```text
35%
```

This reduces excessive sector concentration.

---

## Rebalancing Threshold

The system does not necessarily rebuild the portfolio for insignificant allocation changes.

A rebalance can be skipped when estimated portfolio turnover is below:

```text
15%
```

This helps reduce unnecessary trading.

---

## Stop Loss

The strategy includes configurable stop-loss behaviour.

Current threshold:

```text
-15%
```

---

# Transaction Costs

Trading simulations include execution friction rather than assuming free trading.

Current assumptions:

```text
Transaction Cost : 8 basis points
Slippage         : 3 basis points
```

These costs are incorporated into simulated trading performance.

---

# Chronological Evaluation

Financial time-series data should not be randomly shuffled between training and testing periods.

The project therefore uses chronological splits:

| Stage | Period |
|---|---|
| Training | Up to 2023-12-31 |
| Validation | 2024-01-01 to 2024-06-30 |
| Testing | 2024-07-01 to 2024-12-31 |
| Simulated 2025 Evaluation | 2025-01-01 onward |

This prevents future observations from being included in the main model-training period.

---

# Strategy Performance

The committed experiment produced the following results:

| Period | Total Return | Sharpe Ratio | Max Drawdown | NIFTY 50 Return | Alpha |
|---|---:|---:|---:|---:|---:|
| Validation H1 2024 | **15.93%** | **2.17** | -5.97% | 10.43% | **+5.49%** |
| Testing H2 2024 | **-6.25%** | **-0.54** | -14.34% | -2.06% | **-4.19%** |
| Simulated 2025 Evaluation | **10.41%** | **1.27** | -8.07% | 10.25% | **+0.16%** |

Additional statistics from the same experiment:

| Period | CAGR | Sortino | Volatility | Trades |
|---|---:|---:|---:|---:|
| Validation H1 2024 | 36.75% | 2.41 | 17.56% | 20 |
| Testing H2 2024 | -12.21% | -0.74 | 13.59% | 17 |
| Simulated 2025 Evaluation | 10.77% | 1.98 | 12.39% | 35 |

The negative H2 2024 result is deliberately retained.

The project does not select only successful periods for presentation. The test period demonstrates that model performance varies significantly across market regimes.

The 2025 result is a **historical simulation**, not live trading performance.

---

# Evaluation Metrics

The backtesting pipeline calculates metrics including:

- Total return
- CAGR
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Annualized volatility
- Win rate
- Best trading day
- Worst trading day
- Number of trades
- Benchmark return
- Benchmark Sharpe ratio
- Alpha

The prediction models are separately evaluated using:

- Mean Squared Error
- Root Mean Squared Error
- Mean Absolute Error
- R²
- Directional accuracy

---

# Repository Structure

```text
Algo-Trading-Bot/
|
|-- notebooks/
|   `-- hybrid_trading_system.ipynb
|
|-- results/
|   |
|   |-- figures/
|   |   |-- drawdown_curves.png
|   |   |-- equity_curves.png
|   |   |-- feature_importance.png
|   |   |-- metric_comparison.png
|   |   |-- monthly_returns_heatmap.png
|   |   |-- prediction_scatter_Paper_2025.png
|   |   |-- prediction_scatter_Testing.png
|   |   |-- prediction_scatter_Validation.png
|   |   |-- returns_distribution.png
|   |   `-- rolling_metrics.png
|   |
|   |-- tables/
|   |   |-- all_trades.csv
|   |   |-- detailed_transactions_2025.csv
|   |   |-- model_performance.csv
|   |   `-- performance_summary.csv
|   |
|   `-- detailed_report.txt
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

# Generated Analysis

The repository contains several outputs for inspecting the strategy from different angles.

## Equity Curves

```text
results/figures/equity_curves.png
```

Tracks cumulative portfolio performance across evaluation periods.

## Drawdown Curves

```text
results/figures/drawdown_curves.png
```

Shows declines from previous portfolio peaks.

## Monthly Returns

```text
results/figures/monthly_returns_heatmap.png
```

Displays monthly strategy returns.

## Rolling Metrics

```text
results/figures/rolling_metrics.png
```

Tracks changes in risk and performance metrics through time.

## Return Distribution

```text
results/figures/returns_distribution.png
```

Visualizes the distribution of realized strategy returns.

## Feature Importance

```text
results/figures/feature_importance.png
```

Shows XGBoost feature importance.

## Prediction Diagnostics

```text
results/figures/prediction_scatter_Validation.png
results/figures/prediction_scatter_Testing.png
results/figures/prediction_scatter_Paper_2025.png
```

These plots compare realized returns against model predictions across evaluation periods.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/RathanPaida/Algo-Trading-Bot.git
cd Algo-Trading-Bot
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
numpy
pandas
yfinance
xgboost
scikit-learn
torch
matplotlib
seaborn
jupyter
```

---

# Running the Project

Start Jupyter:

```bash
python -m jupyter notebook
```

Then open:

```text
notebooks/hybrid_trading_system.ipynb
```

Run the notebook cells sequentially.

The pipeline will:

```text
1. Load NIFTY 50 benchmark data
2. Download stock data
3. Build engineered features
4. Train XGBoost
5. Train the LSTM
6. Generate LSTM predictions
7. Build the ensemble score
8. Run validation backtest
9. Run testing backtest
10. Run the simulated 2025 evaluation
11. Generate performance reports and visualizations
```

An internet connection is required because market data is retrieved through Yahoo Finance.

LSTM training can take significantly longer on CPU than on CUDA-capable hardware.

---

# Reproducibility

The project sets deterministic random seeds for NumPy and PyTorch where applicable.

However, exact numerical reproduction can still depend on:

- Python and library versions
- CPU/GPU hardware
- CUDA behaviour
- Yahoo Finance data changes
- External data availability
- Floating-point behaviour
- Parallel execution

The committed result files represent the output of a completed experimental run.

---

# Important Experimental Considerations

## Historical Constituents

The current stock universe is based on a selected NIFTY 50 constituent list.

It does not reconstruct the complete historical NIFTY 50 membership for every past date.

This can introduce survivorship bias.

## Market Data

Yahoo Finance is convenient for research but may contain:

- Missing observations
- Symbol changes
- Data revisions
- Delisted-security limitations
- Corporate-action inconsistencies

## Execution Model

Transaction costs and slippage are approximations.

Real trading performance may additionally depend on:

- Bid-ask spreads
- Liquidity
- Market impact
- Order execution
- Taxes and brokerage
- Position limits
- Execution latency

## Model Risk

Machine-learning relationships learned from historical financial data may weaken or disappear as market regimes change.

The negative H2 2024 test period demonstrates this directly.

---

# Limitations

This is a quantitative research project, not a production trading platform.

Current limitations include:

- Historical simulation rather than live execution
- No complete point-in-time reconstruction of historical index constituents
- Simplified transaction-cost modelling
- Simplified slippage modelling
- Limited liquidity modelling
- Limited market-impact modelling
- Dependence on Yahoo Finance data
- No guarantee of model stability across market regimes
- No exhaustive hyperparameter optimization
- No formal probabilistic uncertainty model
- No live order-management or brokerage integration

---

# Possible Future Improvements

Potential extensions include:

- Walk-forward model retraining
- Rolling or expanding training windows
- Historical NIFTY constituent reconstruction
- Hyperparameter optimization
- Better liquidity and market-impact models
- Fundamental financial features
- Macroeconomic features
- Regime detection
- Prediction uncertainty estimation
- Portfolio optimization using covariance information
- Risk-parity allocation
- Dynamic ensemble weights
- Model calibration
- Live paper-trading infrastructure
- Broker API integration
- Automated experiment tracking

---

# Tech Stack

### Programming

- Python

### Data

- NumPy
- Pandas
- yfinance

### Machine Learning

- XGBoost
- scikit-learn

### Deep Learning

- PyTorch
- LSTM

### Visualization

- Matplotlib
- Seaborn

### Development

- Jupyter Notebook
- Git
- GitHub

---

# Project Scope

This repository focuses on the engineering and quantitative-research aspects of building a machine-learning-driven trading pipeline:

```text
Data
 -> Features
 -> Prediction
 -> Ranking
 -> Portfolio
 -> Backtesting
 -> Risk Analysis
 -> Evaluation
```

The goal is not to claim a consistently profitable trading strategy, but to build and evaluate a complete experimental framework while preserving both successful and unsuccessful out-of-sample results.

---

# Disclaimer

This project is provided solely for **educational and research purposes**.

It does not constitute financial, investment, or trading advice.

All reported performance is historical or simulated. Past performance does not guarantee future results.
