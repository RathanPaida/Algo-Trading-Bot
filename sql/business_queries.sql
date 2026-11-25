-- =====================================================================
-- Business questions answered with SQL on the DuckDB warehouse.
-- Run any of them with:
--     python -m equity_intel sql "<query>"
-- or interactively in the dashboard's SQL tab.
-- =====================================================================

-- Q1. Which periods did the strategy beat NIFTY 50, and by how much?
SELECT period, "Total_Return", "Benchmark_Return", "Excess_Return", "Sharpe", "MaxDD"
FROM perf_summary
ORDER BY "Start";

-- Q2. Best and worst months for the strategy
SELECT month, round(strategy_return * 100, 2) AS strategy_pct,
       round(benchmark_return * 100, 2) AS nifty_pct
FROM v_monthly_returns
ORDER BY strategy_return DESC
LIMIT 5;

-- Q3. Which sectors does the model favour? (share of all BUY orders)
SELECT sector, count(*) AS buys,
       round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct_of_buys
FROM fact_transactions
WHERE action = 'BUY'
GROUP BY sector
ORDER BY buys DESC;

-- Q4. Top 10 most frequently bought stocks
SELECT symbol, sector, buys, sells
FROM v_stock_trade_counts
LIMIT 10;

-- Q5. Current portfolio and sector exposure
SELECT * FROM v_sector_exposure_latest;

-- Q6. Today's top 10 model picks
SELECT rank, symbol, sector, round(score * 100, 2) AS predicted_5d_return_pct
FROM v_latest_signals
LIMIT 10;

-- Q7. Total transaction costs paid per period (in % of portfolio)
SELECT period, count(*) AS rebalances,
       round(sum(cost) * 100, 2) AS total_cost_pct,
       round(avg(turnover) * 100, 1) AS avg_turnover_pct
FROM fact_trades
GROUP BY period;

-- Q8. Did the model's ranking skill (IC) hold up out of sample?
SELECT period, round("XGB_IC", 4) AS xgb_ic, round("LSTM_IC", 4) AS lstm_ic,
       round("Ensemble_IC", 4) AS ensemble_ic, round("Ensemble_Direction_Acc", 3) AS direction_acc
FROM model_performance;

-- Q9. Worst drawdown per period and when it happened
SELECT period, min(drawdown) AS max_drawdown,
       arg_min(date, drawdown) AS worst_day
FROM v_drawdown
GROUP BY period;

-- Q10. Best performing sector per year (average stock return)
SELECT year, sector, avg_return
FROM (
    SELECT r.year, s.sector, avg(r.yearly_return) AS avg_return,
           row_number() OVER (PARTITION BY r.year ORDER BY avg(r.yearly_return) DESC) AS rn
    FROM v_stock_yearly_returns r
    JOIN dim_stock s USING (symbol)
    GROUP BY r.year, s.sector
)
WHERE rn = 1
ORDER BY year;

-- Q11. Data-quality status of every stock
SELECT symbol, sector, dq_status, dq_issues
FROM dim_stock
WHERE dq_status <> 'PASS'
ORDER BY dq_status;

-- Q12. Pipeline run history (operational monitoring)
SELECT stage, status, started_at, finished_at,
       datediff('second', started_at, finished_at) AS seconds, details
FROM pipeline_runs
ORDER BY started_at DESC
LIMIT 20;
