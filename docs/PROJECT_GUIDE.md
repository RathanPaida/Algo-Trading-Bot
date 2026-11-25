# Project guide - how every part works, in plain language

Read this top to bottom once; each section says **what** a part does, **why** it exists,
and **where** it lives in the code. Key terms are in bold.

---

## 1. The big picture

The platform answers one business question: *"Which NIFTY 50 stocks should a weekly
portfolio hold, how risky is that portfolio, and can we trust the model behind it?"*

Data flows in one direction:

```
Yahoo Finance → raw files → checked files → features → models → portfolio backtest
             → risk + model diagnostics → warehouse → dashboard / API / AI analyst
```

Run all of it with `python -m equity_intel run`. Each step is also a separate command,
which is how real data pipelines are built (a failed step can be re-run on its own).

---

## 2. Data engineering (`equity_intel/data/`)

**Ingestion** (`lake.py`) downloads daily prices (open, high, low, close, volume) for 51
stocks and the NIFTY 50 index. It is **incremental**: if a stock's file already ends on
20 Nov, only days after 20 Nov are downloaded. That saves time and is how production
pipelines avoid re-loading 16 years every day.

**Medallion (bronze / silver / gold) layers** - a standard data-lake pattern:

| Layer | Folder | Contents |
|---|---|---|
| Bronze | `data/raw/` | exactly what the source returned (Parquet files) |
| Silver | `data/clean/` | validated and repaired data |
| Gold | `data/warehouse.duckdb` | analysis-ready tables for SQL / BI |

**Parquet** is a compressed, column-based file format - much faster and smaller than CSV.
**DuckDB** is a database stored in a single file (like SQLite) but built for analytics.

**Data quality** (`quality.py`):
- *Schema checks* with **pandera**: prices > 0, volume ≥ 0, High ≥ Low ...
- *Market checks*: missing trading days, a stock that stopped updating ("stale"), one-day
  moves above 20 %, prices stuck at the same value.
- Each stock gets **PASS / WARN / FAIL**; FAIL stocks are excluded from modelling.

Real problems it found: Tata Motors' Oct 2025 demerger and a TRENT bonus issue looked like
33-40 % crashes because Yahoo had not adjusted the prices. These are recorded in
`config/corporate_actions.yaml` (a **corporate-actions master**) with the reason, so the
fix is documented and reviewable rather than hidden in code.

**Star schema** (`warehouse.py`): *fact* tables hold events/measurements (prices, trades,
predictions); *dimension* tables describe things (`dim_stock`: sector, quality status).
BI tools like Power BI expect this shape.

---

## 3. SQL and BI (`sql/`)

- `views.sql` - reusable **views** (saved queries), e.g. `v_latest_holdings`,
  `v_monthly_returns`, `v_drawdown`.
- `business_queries.sql` - 12 business questions answered in SQL. Concepts used:
  `GROUP BY`, `JOIN`, window functions (`OVER (PARTITION BY ... ORDER BY ...)`),
  `row_number()` for "best per group", `FILTER` for conditional counts.
- `python -m equity_intel export-bi` writes every table as Parquet for **Power BI**;
  `docs/powerbi_guide.md` lists the relationships and DAX measures.

---

## 4. Features and target (`features.py`)

For every stock and day, 39 **features** are calculated from past prices only:
returns over 1-20 days, momentum, volatility, moving-average gaps, RSI, MACD,
Bollinger bands, volume changes, candle shapes.

The **target** is the return over the *next* 5 trading days.

**Look-ahead bias** is the classic mistake in finance ML: using information that would
not have been known at the time. `tests/test_features.py` changes future prices and checks
that past features do not change - an automated proof there is no leakage.

**Embargo**: the target for the last 5 training days needs prices from the validation
period, so those 5 days are removed from training.

---

## 5. Models (`equity_intel/models/`)

- **XGBoost** - many small decision trees, each correcting the previous ones (gradient
  boosting). Looks at one day's 39 features at a time.
- **LSTM** - a recurrent neural network (PyTorch) that reads the last 90 days of features
  as a sequence, so it can learn patterns over time. Trained on the GPU with
  **early stopping** (stop when validation error stops improving).
- **Ensemble** - final score = 60 % XGBoost + 40 % LSTM.
- **Walk-forward retraining** (`walk_forward.py`) - retrain every 6 months on all data
  known at that time and predict only the next 6 months. This mimics how a model would be
  used in production.

---

## 6. Portfolio backtest (`backtest.py`)

Every Friday: rank stocks by score → drop stocks with weak momentum or very high
volatility → hold the top 15 → weight calmer stocks more (**inverse-volatility sizing**)
→ cap any sector at 35 % → skip the trade if it changes less than 15 % of the portfolio →
pay 0.11 % cost on everything traded.

Key metrics (`metrics.py`):

| Metric | Meaning |
|---|---|
| Total return / CAGR | growth over the period / per year |
| Sharpe ratio | return per unit of volatility (higher is better; > 1 is good) |
| Sortino ratio | like Sharpe but only counts downside volatility |
| Max drawdown | the largest fall from a previous peak |
| Excess return | strategy return minus NIFTY 50 return |

---

## 7. Is the model actually good? (`baselines.py`, `monitoring.py`, `explain.py`)

- **Baselines** - the same rules run with *no model*: equal weights, momentum only, and
  random ranking. If the model can't beat random ranking, the rules - not the model - made
  the money. (Finding: that is largely the case.)
- **Information Coefficient (IC)** - each day, the rank correlation between predicted and
  actual returns across stocks. 0 = no skill; 0.02-0.05 is typical for useful signals.
  Training IC 0.49 vs ~0.02 out of sample = **overfitting**.
- **SHAP** - splits each prediction into "how much each feature pushed it up or down".
  The contributions add up exactly to the prediction (tested). Banks need this to explain
  model decisions to risk and audit teams.
- **PSI (Population Stability Index)** - compares a feature's recent distribution with
  training data. < 0.10 stable, 0.10-0.25 watch, > 0.25 alert. Banks use PSI to monitor
  credit and market models.
- **MLflow** - records every run's settings, metrics and files so any result can be
  reproduced (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

---

## 8. Risk analytics (`risk.py`)

| Measure | Plain meaning |
|---|---|
| **VaR 95 %** (Value at Risk) | on 95 % of days the loss is smaller than this |
| **CVaR / Expected Shortfall** | the *average* loss on the worst 5 % of days |
| **Beta** | portfolio moves this much when NIFTY moves 1 % |
| **Alpha** | return not explained by market exposure |
| **Risk contribution** | share of total portfolio risk caused by each holding |
| **Stress test** | replay a past crisis (COVID 2020, 2022 rate hikes ...) on today's weights |

Historical VaR = the 5th-worst percentile of past daily returns. With INR 10 lakh and
VaR 1.57 %, the "bad day" loss is about INR 15,700.

---

## 9. Generative AI (`equity_intel/genai/`)

**Text-to-SQL** (`text_to_sql.py`):
1. The LLM gets the list of tables/columns and example queries (never the raw data).
2. It writes one SQL query.
3. Guard rails: only a single `SELECT` is allowed; it runs on a **read-only** connection
   with file access disabled and a row limit (`sql_safety.py`). Two layers of protection
   = **defence in depth**.
4. If the SQL errors, the error goes back to the LLM once to fix it.
5. The LLM writes the answer using only the returned rows; the SQL is always shown.

**Commentary** (`commentary.py`) - facts are computed from the warehouse first, the LLM
writes the note from those facts, then a **grounding check** verifies every number in the
note exists in the facts (a simple hallucination guard).

**Providers** (`llm.py`) - Claude through Anthropic's official Python SDK, a local Ollama
model, or offline mode (templates), so the app works even without an API key.

---

## 10. Applications

- **Dashboard** (`dashboard/app.py`, Streamlit): Overview, Performance, Risk, Model health,
  Data quality, SQL explorer, AI analyst.
- **REST API** (`api/main.py`, FastAPI): other systems can fetch signals, risk or ask
  questions over HTTP. FastAPI generates interactive docs at `/docs` automatically and
  validates inputs with Pydantic.
- **CLI** (`equity_intel/cli.py`): every pipeline stage as a command.

---

## 11. Engineering practice

- **Tests** (`tests/`, pytest, 71 tests): feature leakage, metric maths, backtest rules,
  data-quality checks, SQL safety, risk maths, API, every dashboard page, GenAI with a
  fake LLM (no internet needed).
- **Lint** (ruff): consistent style and catches common bugs.
- **Config** (`config/config.yaml`): every number lives in one file, not in code.
- **Docker**: one image runs the pipeline, API and dashboard the same way on any machine.
- **CI/CD** (`.github/workflows/ci.yml`): every push runs lint, tests and a Docker build.
- **Scheduled job** (`daily-data.yml`): weekdays after market close, ingest new prices and
  run quality checks; the job fails (and alerts) if any stock FAILS.
