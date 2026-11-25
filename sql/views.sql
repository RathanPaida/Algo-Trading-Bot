-- =====================================================================
-- Reusable BI views on top of the warehouse tables.
-- Refreshed automatically by:  python -m equity_intel warehouse
-- =====================================================================

-- Monthly strategy vs NIFTY 50 returns, per evaluation period
CREATE OR REPLACE VIEW v_monthly_returns AS
SELECT
    period,
    date_trunc('month', date)                         AS month,
    exp(sum(ln(1 + strategy_return)))  - 1            AS strategy_return,
    exp(sum(ln(1 + benchmark_return))) - 1            AS benchmark_return
FROM fact_daily_returns
GROUP BY period, date_trunc('month', date)
ORDER BY month;

-- Holdings from the most recent rebalance
CREATE OR REPLACE VIEW v_latest_holdings AS
SELECT w.symbol, w.sector, w.weight, w.date AS rebalance_date
FROM fact_portfolio_weights w
WHERE w.date = (SELECT max(date) FROM fact_portfolio_weights)
ORDER BY w.weight DESC;

-- Sector exposure of the latest portfolio
CREATE OR REPLACE VIEW v_sector_exposure_latest AS
SELECT sector, sum(weight) AS weight, count(*) AS n_stocks
FROM v_latest_holdings
GROUP BY sector
ORDER BY weight DESC;

-- Today's model ranking (latest date with predictions)
CREATE OR REPLACE VIEW v_latest_signals AS
SELECT p.date, p.symbol, s.sector, p.score, p.pred_xgb, p.pred_lstm,
       p.mom20, p.vol60 * sqrt(252) AS ann_volatility,
       rank() OVER (ORDER BY p.score DESC) AS rank
FROM fact_predictions p
JOIN dim_stock s USING (symbol)
WHERE p.date = (SELECT max(date) FROM fact_predictions)
ORDER BY rank;

-- How often each stock was bought, and its sector
CREATE OR REPLACE VIEW v_stock_trade_counts AS
SELECT t.symbol, d.sector,
       count(*) FILTER (WHERE action = 'BUY')       AS buys,
       count(*) FILTER (WHERE action = 'SELL')      AS sells,
       count(*) FILTER (WHERE action = 'STOP_LOSS') AS stop_losses
FROM fact_transactions t
JOIN dim_stock d USING (symbol)
GROUP BY t.symbol, d.sector
ORDER BY buys DESC;

-- Drawdown of the strategy over time
CREATE OR REPLACE VIEW v_drawdown AS
SELECT period, date, equity,
       equity / max(equity) OVER (PARTITION BY period ORDER BY date) - 1 AS drawdown
FROM fact_daily_returns
ORDER BY period, date;

-- Annual return per stock (price-only), useful for sector analysis
CREATE OR REPLACE VIEW v_stock_yearly_returns AS
SELECT symbol, year(date) AS year,
       last(close ORDER BY date) / first(close ORDER BY date) - 1 AS yearly_return
FROM fact_prices
GROUP BY symbol, year(date);
