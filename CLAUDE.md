# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETF Portfolio Construction & Backtesting Tool - a local web app for professional portfolio managers to build, backtest, and hedge ETF-only portfolios.

## Tech Stack

- **UI Framework:** Plotly Dash (single Python app)
- **Data Source:** Yahoo Finance (yfinance)
- **Analytics:** pandas, numpy
- **Charts:** Plotly
- **Deployment:** Docker + docker-compose

## Build & Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
# Opens at http://localhost:8050

# Run with Docker
docker compose up --build
# Opens at http://localhost:8050

# Run tests
pytest tests/
```

## Project Structure

```
MasterPortfolio/
  app.py                    # Main Dash application entry point
  requirements.txt          # Python dependencies
  Dockerfile
  docker-compose.yml

  engines/
    portfolio_builder.py    # Portfolio construction logic (dispatches by strategy)
    risk_parity.py          # Risk parity allocation engine
    backtester.py           # Backtest simulation with rebalancing
    metrics.py              # Performance metrics (CAGR, Sharpe, etc.)
    hedging.py              # Rule-based hedge recommendations
    diversification.py      # Allocation analysis, correlations
    stress.py               # Stress test scenarios

  services/
    data_client.py          # Yahoo Finance wrapper with disk caching
    universe.py             # ETF universe loader and filters

  domain/
    schemas.py              # Pydantic models for all data types

  ui/
    layouts.py              # Dash tab layouts
    callbacks.py            # Dash callback handlers
    charts.py               # Plotly chart builders

  data/
    etf_universe.py         # Hardcoded ETF universe (~35 ETFs)
    world_portfolio.py      # Default 60/40 baseline allocation
    stress_scenarios.py     # Predefined stress test configs

  tests/
    test_metrics.py
    test_portfolio_builder.py
    test_backtester.py
    test_risk_parity.py

  cache/                    # Local cache directory (gitignored)
```

## Core Engine Logic

### Allocation Strategies

Three portfolio construction strategies are available:

1. **Strategic** (default) - Traditional baseline allocation with tilts
2. **Risk Parity** - Equal risk contribution weighting
3. **Equal Weight** - Simple equal weighting across ETFs

### Portfolio Builder (`engines/portfolio_builder.py`)

**Strategic Strategy:**
1. Start from baseline allocation (US Equity 40%, Intl Equity 20%, Bonds 30%, Alternatives 10%)
2. Apply risk profile adjustments (Conservative/Moderate/Aggressive)
3. Apply user tilts (region/sector)
4. Map sleeves to specific ETFs (prefer low-cost, high-liquidity)
5. Enforce constraints (max weight, sector caps, excluded tickers)
6. Normalize weights to 100%

### Risk Parity (`engines/risk_parity.py`)

Two methods available:
- **Inverse Volatility**: `weight_i = (1/vol_i) / sum(1/vol_j)`
- **Equal Risk Contribution**: Iterative optimization for equal risk contribution

Predefined universes:
- `balanced`: SPY, EFA, EEM, AGG, TLT, GLD, VNQ
- `equity_focused`: SPY, QQQ, IWM, EFA, VWO, AGG
- `defensive`: AGG, TLT, GLD, SPY, VNQ, SHY
- `global`: VTI, VEA, VWO, BND, GLD, VNQ, VNQI

### Backtester (`engines/backtester.py`)
- Fetches adjusted close prices via yfinance with disk caching
- Calculates daily returns
- Simulates portfolio with periodic rebalancing (None/Monthly/Quarterly/Annual)
- Applies simple transaction costs (flat bps per rebalance)
- Tracks equity curve and drawdowns

### Metrics (`engines/metrics.py`)
- CAGR: `(end_value/start_value)^(1/years) - 1`
- Vol: `std(daily_returns) * sqrt(252)`
- Sharpe: `(mean(daily_excess_return) * 252) / vol`
- Sortino: downside deviation uses only negative daily returns
- Max drawdown: `min(value/peak - 1)` over time

### Hedging (`engines/hedging.py`)
Rule-based ETF hedge suggestions:
- High beta (>1.2) → SH (inverse S&P 500)
- Tech concentration (>25%) → PSQ (inverse Nasdaq)
- High equity (>70%) → XLP (defensive sector)
- Long duration bonds → TBF (inverse treasury)
- Tail risk → GLD (gold)

## UI Tabs

1. **Portfolio Builder** - Risk slider, region tilts, constraints, generate button
2. **Backtest** - Date range, rebalance frequency, benchmark selector, run button
3. **Analytics** - Metrics cards, equity curve, drawdown chart
4. **Diversification** - Allocation pies, correlation heatmap, stress test bars
5. **Hedges** - Recommendations list, apply hedge button
6. **Export** - CSV/JSON download for portfolio and backtest results

## ETF Universe

~35 ETFs covering:
- **US Equity:** SPY, VOO, VTI, QQQ, IWM, DIA
- **International:** EFA, VEA, IEFA, EEM, VWO
- **Sectors:** XLK, XLF, XLE, XLV, XLI, XLP, XLU
- **Fixed Income:** AGG, BND, TLT, IEF, SHY, LQD, HYG
- **Alternatives:** GLD, IAU, SLV, VNQ, VNQI
- **Inverse/Hedge:** SH, PSQ, SDS, TBF

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py -v

# Key test cases:
# - test_metrics.py: CAGR, volatility, Sharpe against known values
# - test_portfolio_builder.py: constraints, tilts, normalization
# - test_backtester.py: rebalance dates, weight drift, turnover
```

## Data Caching

- Price data cached to disk via `diskcache` in `./cache/`
- Cache key: hash of `(ticker, start_date, end_date, data_type)`
- TTL: 1 day for prices, 7 days for metadata
- Clear cache: `services.data_client.clear_cache()`

## Key Constraints

- Server binds to localhost (127.0.0.1) only
- No external API keys required (uses free Yahoo Finance)
- Execution module not implemented in MVP
- All weights normalized to sum to 1.0
