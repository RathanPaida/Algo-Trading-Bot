# Using the warehouse in Power BI

The pipeline's gold layer is a DuckDB database. For Power BI (or Tableau / Excel),
export every table and view as Parquet:

```bash
python -m equity_intel export-bi      # writes exports/powerbi/*.parquet
```

In Power BI Desktop: **Get Data → Parquet** (one file per table), or **Get Data →
Folder** on `exports/powerbi` and combine.

## Data model (star schema)

| Table | Grain | Key columns |
|---|---|---|
| `dim_stock` | one row per stock | `symbol`, `sector`, `dq_status` |
| `fact_prices` | stock × day | `date`, `symbol`, OHLCV |
| `fact_predictions` | stock × day | `date`, `symbol`, `score`, `pred_xgb`, `pred_lstm` |
| `fact_transactions` | one trade | `date`, `symbol`, `action`, `weight`, `period` |
| `fact_portfolio_weights` | stock × rebalance | `date`, `symbol`, `weight`, `period` |
| `fact_daily_returns` | day × period | `date`, `period`, `equity`, `benchmark_equity` |
| `holdings_risk` | current holding | `symbol`, `weight`, `beta`, `risk_contribution_pct` |
| `perf_summary`, `risk_summary`, `model_performance` | one row per period | `period` |

Relationships (single direction, many-to-one):

```
dim_stock[symbol] 1 ──< fact_prices[symbol]
dim_stock[symbol] 1 ──< fact_predictions[symbol]
dim_stock[symbol] 1 ──< fact_transactions[symbol]
dim_stock[symbol] 1 ──< fact_portfolio_weights[symbol]
dim_stock[symbol] 1 ──< holdings_risk[symbol]
Date[date]        1 ──< every fact table's [date]
```

Create a `Date` table with `CALENDAR(MIN(fact_prices[date]), MAX(fact_prices[date]))`
and mark it as the date table.

## Suggested DAX measures

```DAX
Strategy Return % =
VAR firstEq = CALCULATE ( MIN ( fact_daily_returns[equity] ), FIRSTDATE ( fact_daily_returns[date] ) )
VAR lastEq  = CALCULATE ( MAX ( fact_daily_returns[equity] ), LASTDATE ( fact_daily_returns[date] ) )
RETURN DIVIDE ( lastEq, firstEq ) - 1

NIFTY Return % =
VAR firstB = CALCULATE ( MIN ( fact_daily_returns[benchmark_equity] ), FIRSTDATE ( fact_daily_returns[date] ) )
VAR lastB  = CALCULATE ( MAX ( fact_daily_returns[benchmark_equity] ), LASTDATE ( fact_daily_returns[date] ) )
RETURN DIVIDE ( lastB, firstB ) - 1

Excess Return % = [Strategy Return %] - [NIFTY Return %]

Buy Orders = CALCULATE ( COUNTROWS ( fact_transactions ), fact_transactions[action] = "BUY" )

Sector Weight % = SUM ( fact_portfolio_weights[weight] )
```

## Suggested report pages

1. **Executive summary** - KPI cards (return, excess return, Sharpe, max drawdown, VaR),
   equity line chart with a `period` slicer.
2. **Portfolio** - sector exposure bar, holdings table, buy/sell counts by sector.
3. **Risk** - `holdings_risk` weight vs risk contribution, `stress_tests` bar chart.
4. **Model monitoring** - `model_performance` IC by period, `drift_report` PSI with
   conditional formatting (green < 0.10, amber < 0.25, red above).
5. **Data quality** - `dim_stock` / `dq_report` status counts and issue list.
