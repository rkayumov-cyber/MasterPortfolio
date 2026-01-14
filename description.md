# ETF Portfolio Construction & Backtesting Tool (Local Web App)
**Target users:** Professional portfolio managers
**Universe:** ETFs only (100+ liquid ETFs)
**Deployment:** Standalone local (Docker-first)
**Data Source:** Yahoo Finance (yfinance) with disk caching
**UI Framework:** Plotly Dash with Bloomberg-style theming

---

## Implemented Features (Current State)

### Core Features
- **Portfolio Builder** - Strategic, Risk Parity, Equal Weight, and preset strategies
- **Backtesting Engine** - Historical simulation with rebalancing and transaction costs
- **Portfolio Optimizer** - Exhaustive search with efficient frontier visualization
- **Market Regime Detection** - Real-time regime classification with VIX/SMA indicators
- **ETF Screener** - Filter and search 100+ liquid ETFs
- **Hedging Recommendations** - Rule-based hedge suggestions

### Analytics & Visualization
- **Performance Metrics** - CAGR, Sharpe, Sortino, Max Drawdown, etc.
- **Advanced Analytics** - Monte Carlo simulations, VaR/CVaR, rolling metrics
- **Diversification Analysis** - Allocation breakdown, correlation heatmap, concentration metrics
- **Stress Testing** - Historical and hypothetical scenario analysis
- **Multi-Theme UI** - Bloomberg, Dark, Light, Modern, Professional themes

### Data Features
- **Intraday Data Download** - 1m to 1h intervals with CSV export
- **Expanded ETF Universe** - 100+ ETFs including leveraged, volatility, sector, country ETFs
- **Fear & Greed Index** - Live CNN Fear & Greed integration
- **Analyst Consensus** - Aggregated buy/hold/sell ratings from yfinance
- **PDF Research Upload** - Extract sentiment from research documents

### Export & Integration
- **CSV/JSON Export** - Portfolio and backtest results
- **Portfolio Import** - Import holdings from CSV
- **Rebalancing Calculator** - Calculate trades needed to rebalance

---

## Original Specification

---

## 0) What Claude Code Should Build
### MVP deliverable (v1)
A local web app that lets a PM:
1) Define portfolio preferences (risk, region/sector tilts, constraints, hedging toggles)  
2) Generate an ETF-only portfolio (weights + rationale)  
3) Backtest over a chosen period (rebalance rules, benchmark)  
4) View analytics + charts (performance, drawdowns, correlations, allocation)  
5) Get hedge ideas (rule-based, explainable)  
6) Export outputs (CSV/JSON + charts)

### Non-goals (v1)
- Full OMS/EMS replacement  
- Intraday backtesting at tick/1-min scale  
- Complex derivatives pricing (options are “idea output”; execution optional and controlled)  
- Multi-user auth/role-based permissions (local single-user assumed)

---

## 1) Product Requirements

### 1.1 Core Modules
**A) Dynamic Portfolio Construction**
- Inputs:
  - Risk profile: `Conservative | Moderate | Aggressive` (or numeric risk slider)
  - Region tilts: overweight/underweight by region
  - Sector tilts: overweight/underweight by sector
  - Constraints: max weight per ETF, min weight, max sector concentration, exclude list
  - Liquidity filter: min ADV / AUM / bid-ask proxy (from MCP metadata if available)
  - Optional “Research tilt preset” (e.g., GOAL 2026 tilt pack)
- Output:
  - Proposed portfolio: `[ticker, name, category tags, weight, rationale]`
  - A “diff view” when user adjusts weights manually

**B) Backtesting Engine**
- Inputs:
  - Time range (start/end) + frequency (daily default)
  - Rebalancing: `None | Monthly | Quarterly | Annual | Custom`
  - Benchmark: single ETF ticker or blended benchmark (weights)
  - Costs toggle (v1 simple): optional flat bps per rebalance + slippage bps
- Outputs:
  - Equity curve (portfolio vs benchmark)
  - Metrics:
    - Total return, CAGR
    - Volatility (annualized)
    - Sharpe (risk-free proxy configurable)
    - Sortino
    - Max drawdown + drawdown duration
    - Tracking error, information ratio (optional)
  - Rolling analytics:
    - rolling vol, rolling returns, rolling Sharpe
  - Exportable results (CSV + JSON)

**C) Tactical Hedging Module (Explainable, Rule-Based v1)**
- Goal: Suggest ETF-level hedges based on portfolio exposures and “cheap protection” heuristics.
- Inputs:
  - Current portfolio
  - User preference: hedging on/off, aggressiveness, hedge budget (% notional)
- Outputs:
  - Ranked hedge suggestions:
    - instrument (ETF or “option idea”), rationale, risk targeted, rough sizing
  - “Add hedge to portfolio” button that creates a hedge sleeve and re-runs backtest

**D) Diversification Analytics**
- Allocation breakdown:
  - by asset class, region, sector
- Concentration metrics:
  - HHI, top-1/top-3 weight, effective number of bets
- Correlations:
  - correlation matrix + heatmap
- Stress tests:
  - historical windows (e.g., GFC, Mar 2020) if data available
  - hypothetical shocks (rates up, equity down, tech drawdown)

**E) Visualization Dashboard**
- Tabs:
  - **Portfolio** (holdings, weights, constraints, rationale)
  - **Backtest** (equity curve, metrics)
  - **Risk** (drawdowns, rolling vol/Sharpe)
  - **Diversification** (allocation charts, correlation heatmap)
  - **Stress Tests** (scenario bars)
  - **Hedges** (recommendations + impact comparison)

---

## 2) Data & Integration Requirements

### 2.1 MCP Data Requirements
Minimum required endpoints (wrap these in a clean adapter):
- Historical prices (adjusted close) OR total return series for ETF ticker
- ETF metadata: name, currency, exchange, expense ratio (optional), AUM/ADV (optional)
- Optional look-through data: region/sector exposures (if MCP provides it)

**Caching**
- Cache historical series per ticker/timeframe locally (disk) to avoid repeated pulls.
- Cache policy: `ttl_days` for metadata; time series keyed by `(ticker, start, end, freq)`.

### 2.2 Liquid ETS Execution Requirements (Optional v1, gated)
- Place ETF orders (market/limit) as a basket for the target portfolio
- Must run in **“simulation mode” by default**
- Require explicit user toggle + confirmations before any live order

**Execution outputs**
- Order preview (basket)
- API result and statuses
- Audit log file written locally

---

## 3) Technical Architecture (Recommended for Claude Code)

### 3.1 Preferred Stack (Clean separation)
- **Backend:** Python + FastAPI
- **Analytics:** pandas, numpy
- **Charts:** Plotly (frontend) or served JSON traces
- **Frontend:** React + Vite (or Next.js if preferred)
- **Local packaging:** Docker + docker-compose

> If you want fastest MVP with fewer moving parts: build the UI in **Plotly Dash** and keep everything Python. The modules below still apply; only the UI layer changes.

### 3.2 Runtime Flow
1) User selects preferences → frontend calls `/api/portfolio/construct`
2) Backend selects ETFs + weights → returns portfolio + rationale
3) User runs backtest → frontend calls `/api/backtest/run`
4) Backend pulls MCP series (cached), simulates, computes metrics → returns results
5) User requests hedges → `/api/hedge/recommend`
6) (Optional) user executes → `/api/execution/preview` then `/api/execution/submit`

---

## 4) Domain Model (Pydantic Schemas)

### 4.1 ETF Universe
```json
{
  "ticker": "SPY",
  "name": "SPDR S&P 500 ETF Trust",
  "asset_class": "Equity",
  "region": "US",
  "sector": "Broad",
  "currency": "USD",
  "tags": ["large-cap", "core"],
  "liquidity": {"aum_usd": 0, "adv_usd": 0, "spread_bps": 0}
}
4.2 Portfolio Construction Request
json
Copy code
{
  "risk_profile": "Moderate",
  "constraints": {
    "max_weight_per_etf": 0.20,
    "min_weight_per_etf": 0.00,
    "max_sector_weight": 0.35,
    "excluded_tickers": ["XYZ"]
  },
  "tilts": {
    "regions": {"AsiaPac": 0.05, "Europe": -0.05},
    "sectors": {"Technology": 0.05}
  },
  "hedging": {"enabled": true, "budget_pct": 0.02},
  "research_preset": "GOAL_2026",
  "base_benchmark": "WORLD_PORTFOLIO"
}
4.3 Portfolio Response
json
Copy code
{
  "portfolio": [
    {"ticker": "SPY", "weight": 0.30, "rationale": "Core US equity sleeve"},
    {"ticker": "AGG", "weight": 0.30, "rationale": "Core bond sleeve"},
    {"ticker": "GLD", "weight": 0.05, "rationale": "Diversifier sleeve"}
  ],
  "notes": ["Applied AsiaPac overweight tilt", "Liquidity filter passed for all holdings"]
}
4.4 Backtest Request
json
Copy code
{
  "portfolio": [{"ticker": "SPY", "weight": 0.60}, {"ticker": "AGG", "weight": 0.40}],
  "start": "2018-01-01",
  "end": "2023-12-31",
  "rebalance": "Monthly",
  "benchmark": {"type": "single", "ticker": "SPY"},
  "costs": {"enabled": true, "rebalance_bps": 2.0, "slippage_bps": 1.0}
}
4.5 Backtest Response (Core)
json
Copy code
{
  "equity_curve": [{"date": "2018-01-02", "portfolio": 100.0, "benchmark": 100.0}],
  "metrics": {
    "cagr": 0.08,
    "vol": 0.12,
    "sharpe": 0.60,
    "sortino": 0.85,
    "max_drawdown": -0.28
  },
  "drawdown_curve": [{"date": "2020-03-20", "dd": -0.22}],
  "rolling": {"vol_252d": [], "ret_252d": []}
}
5) Portfolio Construction Logic (Deterministic v1)
5.1 Baseline: “World Portfolio” Anchor
Start from a baseline allocation (config file) representing a neutral global mix.

Then:

Map baseline sleeves → ETF candidates

Apply user tilts (region/sector)

Enforce constraints

Normalize weights to 100%

Keep the “world portfolio” weights as a configurable file, not hard-coded.

5.2 ETF Selection Rules (v1)
For each sleeve (e.g., US Equity, AsiaPac Equity, IG Bonds, Gold):

Select primary ETF = highest liquidity + lowest fee (if available)

Provide 1–2 alternates for user override

5.3 Constraints
Clip weights to max_weight_per_etf

If sector/region caps exceeded:

reduce offending sleeve(s) proportionally

re-allocate to “core diversifier sleeves” (e.g., broad global equity, aggregate bonds)

6) Backtesting Engine Specs
6.1 Calculation Method
Use total return series if available (preferred)

Else:

use adjusted close and compute daily returns

Rebalancing

On rebalance dates:

reset weights to target

apply transaction cost model (flat bps * turnover)

6.2 Metrics (Definitions)
CAGR: (end_value/start_value)^(1/years) - 1

Vol: std(daily_returns) * sqrt(252)

Sharpe: (mean(daily_excess_return) * 252) / (std(daily_return) * sqrt(252))

Sortino: downside deviation uses only negative daily returns

Max drawdown: min over time of value/peak - 1

6.3 Validation Tests (Must Have)
100% SPY portfolio backtest matches SPY series (within tolerance)

Rebalance “None” equals buy-and-hold

Metrics sanity checks (vol >= 0, drawdown <= 0)

7) Tactical Hedging Module (Rule-Based v1)
7.1 Exposure Detection
Compute:

Portfolio beta to a broad equity proxy (e.g., SPY)

Sector concentration flags (tech-heavy, etc.)

Duration proxy for bond sleeve (simple: map bond ETF type to duration bucket)

7.2 Hedge Suggestion Rules (Example Set)
Broad equity risk high (beta high + drawdown risk):

Suggest: inverse equity ETF sleeve OR put-idea on broad equity proxy

Tech concentration > threshold:

Suggest: tech hedge (ETF-level hedge idea) with sizing guidance

Long-duration bond exposure + rising rate risk:

Suggest: duration hedge via short-duration ETF or inverse duration ETF

7.3 Output Format
Each recommendation:

risk_targeted

instrument

suggested_size_pct

cost_proxy (if available; else “N/A”)

why_now (explainable text)

how_to_apply (add to portfolio + rerun backtest)

8) Diversification Analytics & Stress Tests
8.1 Diversification Outputs
Allocation by:

asset class

region

sector (look-through if available; else use ETF labels)

Concentration:

HHI = sum(w_i^2)

Effective N = 1/HHI

Correlations:

return correlation matrix + heatmap

highlight: highest pair corr, lowest corr (diversifier)

8.2 Stress Tests
Historical (if data span permits)

Provide window presets (config):

2008–2009, 2020-02 to 2020-04, 2022 rates shock, etc.
Hypothetical

Equity -15%, bonds +3%, gold +8% (configurable)

Rates +200bp shock (mapped to duration bucket losses)

9) UI/UX Requirements (MVP)
9.1 Screens
Portfolio Builder

risk slider, tilts, constraints, universe filters

“Generate portfolio” + “Lock/edit weights”

Backtest

timeframe picker, rebalance picker, benchmark selector

“Run” button + results tiles

Dashboard

equity curve

drawdown chart

allocation charts

correlation heatmap

Hedges

recommendations list

“Apply hedge” + “Re-run backtest”

Export

CSV/JSON download for portfolio + results

9.2 Visualization List (MVP)
Line: portfolio vs benchmark equity curve

Area/line: drawdown curve

Pie/treemap: allocation

Heatmap: correlations

Bars: stress tests outcomes

10) Project Structure (Claude Code Friendly)
10.1 Repo Tree (FastAPI + React)
pgsql
Copy code
etf-portfolio-tool/
  backend/
    app/
      main.py
      api/
        portfolio.py
        backtest.py
        hedge.py
        execution.py
      core/
        config.py
        logging.py
        cache.py
      services/
        mcp_client.py
        liquid_ets_client.py
        universe.py
      domain/
        schemas.py
      engines/
        portfolio_builder.py
        backtester.py
        metrics.py
        hedging.py
        diversification.py
        stress.py
      tests/
        test_backtest_basic.py
        test_metrics.py
    requirements.txt
    Dockerfile
  frontend/
    src/
      pages/
      components/
      api/
    package.json
    Dockerfile
  docker-compose.yml
  .env.example
  README.md
  configs/
    universe.yaml
    world_portfolio_weights.yaml
    research_presets.yaml
    stress_scenarios.yaml
10.2 Repo Tree (Dash Single-App Alternative)
pgsql
Copy code
etf-portfolio-tool/
  app.py
  engines/
  services/
  domain/
  configs/
  tests/
  Dockerfile
  requirements.txt
  README.md
11) API Endpoints (Backend)
11.1 Portfolio
POST /api/portfolio/construct

POST /api/portfolio/validate

GET /api/universe/list

11.2 Backtest
POST /api/backtest/run

POST /api/backtest/metrics

GET /api/backtest/cache/status

11.3 Hedge
POST /api/hedge/recommend

POST /api/hedge/apply (returns updated portfolio)

11.4 Execution (Gated)
POST /api/execution/preview

POST /api/execution/submit (requires explicit enable flag)

12) Configuration & Secrets
12.1 Environment Variables
MCP_API_KEY=...

MCP_BASE_URL=...

LIQUID_ETS_API_KEY=...

LIQUID_ETS_BASE_URL=...

APP_MODE=simulation|live

CACHE_DIR=./.cache

LOG_LEVEL=INFO

12.2 Config Files
configs/universe.yaml: ETF universe + tags (asset class/region/sector)

configs/world_portfolio_weights.yaml: baseline sleeves

configs/research_presets.yaml: GOAL_2026 tilt pack etc.

configs/stress_scenarios.yaml: scenario definitions

13) Security, Logging, and Guardrails
Bind server to localhost by default

Store secrets in env vars (never commit)

Write audit logs:

data fetch calls (ticker/time, not secret values)

backtest runs (inputs hash)

execution previews/submissions (basket + timestamps)

Execution safety:

Default simulation mode

Require typed confirmation to switch to live mode

14) Docker Deployment (Preferred)
14.1 Quick Start (compose)
bash
Copy code
cp .env.example .env
docker compose up --build
14.2 Local URLs
Frontend: http://localhost:5173 (or your configured port)

Backend: http://localhost:8000

15) Testing Plan (Must Ship)
15.1 Unit Tests
backtester: rebalancing correctness, curve math

metrics: known toy series (hand-calculated)

portfolio_builder: constraint enforcement and normalization

15.2 Integration Tests (Optional v1)
MCP client mocked (recorded responses)

Liquid ETS client mocked (no real orders)

16) Extensibility (v2+ Ideas)
Portfolio optimization mode (mean-variance / max Sharpe / risk parity)

Additional data sources (pluggable adapters)

More advanced hedging cost model (implied vol / skew if accessible)

Multi-portfolio workspace + saved runs

PDF report generation

17) Claude Code Build Script (Step-by-Step Tasks)
Use this checklist as the build plan.

Phase 1 — Skeleton + Config
 Create repo structure (backend + frontend)

 Add config files in /configs

 Implement universe.py loader + validators

 Implement MCP adapter interface + mock mode

Acceptance: GET /api/universe/list returns ETFs from config.

Phase 2 — Portfolio Builder Engine
 Implement baseline allocator (world portfolio anchor)

 Implement tilt application + constraints + normalization

 Implement POST /api/portfolio/construct

Acceptance: constructing with no tilts returns stable baseline; tilts change weights predictably.

Phase 3 — Backtester
 Implement series fetch (cached)

 Implement portfolio simulation with rebalancing

 Implement metrics

 Implement POST /api/backtest/run

Acceptance: 100% SPY backtest matches SPY curve within tolerance.

Phase 4 — Diversification + Stress
 Allocation breakdown

 Correlation matrix

 Stress scenarios engine

 Endpoints for analytics

Acceptance: heatmap data + stress scenario results returned for any portfolio.

Phase 5 — Hedging Recommendations
 Exposure detection

 Rule-based hedge recommender

 Apply hedge → new portfolio object

Acceptance: tech-heavy portfolio triggers a tech hedge recommendation.

Phase 6 — UI + Charts
 Portfolio builder form

 Backtest runner and charts

 Risk + diversification tabs

 Hedge tab with “apply + rerun”

Acceptance: user can complete full loop without errors and export results.

Phase 7 — Execution (Optional, gated)
 Basket preview endpoint

 Submit endpoint (live mode only)

 UI confirmations + audit log

Acceptance: in simulation mode, no live order calls are made.

18) Notes / Open Assumptions (Make Explicit in Code)
"World Portfolio" weights are approximate and must be configurable.

ETF look-through (sector/region) depends on MCP availability; fallback is tag-based.

Option hedges are "idea output" unless execution permissions exist; default is ETF-based hedges.

---

## Current Project Structure (Implemented)

```
MasterPortfolio/
├── app.py                      # Main Dash application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile
├── docker-compose.yml
│
├── engines/
│   ├── portfolio_builder.py    # Portfolio construction logic
│   ├── risk_parity.py          # Risk parity allocation engine
│   ├── backtester.py           # Backtest simulation
│   ├── optimizer.py            # Portfolio optimization (efficient frontier)
│   ├── metrics.py              # Performance metrics
│   ├── advanced_analytics.py   # Monte Carlo, VaR, rolling metrics
│   ├── hedging.py              # Rule-based hedge recommendations
│   ├── diversification.py      # Allocation analysis, correlations
│   ├── stress.py               # Stress test scenarios
│   ├── regime_detector.py      # Market regime detection (VIX/SMA)
│   └── regime_tilts.py         # Regime-based portfolio tilts
│
├── services/
│   ├── data_client.py          # Yahoo Finance wrapper with caching
│   ├── universe.py             # ETF universe loader
│   ├── sentiment_client.py     # Fear & Greed, analyst ratings
│   └── pdf_processor.py        # PDF text extraction
│
├── domain/
│   └── schemas.py              # Pydantic models for all data types
│
├── ui/
│   ├── layouts.py              # Dash tab layouts
│   ├── callbacks.py            # Dash callback handlers
│   └── charts.py               # Plotly chart builders
│
├── data/
│   ├── etf_universe.py         # 100+ ETF definitions
│   ├── world_portfolio.py      # Default baseline allocation
│   ├── stress_scenarios.py     # Stress test configurations
│   └── regime_configs.py       # Regime tilt configurations
│
├── assets/
│   └── themes.css              # Multi-theme stylesheets
│
├── tests/
│   ├── test_metrics.py
│   ├── test_portfolio_builder.py
│   ├── test_backtester.py
│   └── test_risk_parity.py
│
└── cache/                      # Local cache directory (gitignored)
```

## UI Tabs (Current)

1. **Portfolio Builder** - Risk profile, strategy selection, tilts, constraints
2. **Optimizer** - ETF selection, efficient frontier, optimal weights
3. **Data Download** - Intraday/daily data fetch and CSV export
4. **ETF Screener** - Filter by asset class, region, sector
5. **Backtest** - Date range, rebalancing, benchmark comparison
6. **Rebalance** - Calculate rebalancing trades
7. **Analytics** - Performance metrics, equity curve, drawdowns
8. **Advanced** - Monte Carlo, VaR/CVaR, rolling metrics
9. **Diversification** - Allocation pies, correlation heatmap
10. **Compare Strategies** - Side-by-side strategy comparison
11. **Hedges** - Hedge recommendations
12. **Market Regime** - Regime detection, sentiment, recommendations
13. **Export** - CSV/JSON download

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
# Opens at http://localhost:8050

# Run with Docker
docker compose up --build
```

End of spec.