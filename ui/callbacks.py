"""Dash callbacks for the ETF Portfolio Tool."""

import json
from datetime import date, timedelta

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, html, no_update

from domain.schemas import (
    AllocationStrategy,
    BacktestRequest,
    BenchmarkConfig,
    Constraints,
    CostConfig,
    Portfolio,
    PortfolioHolding,
    PortfolioRequest,
    RebalanceFrequency,
    RiskParityConfig,
    RiskProfile,
    Tilts,
)
from engines.backtester import run_backtest
from engines.diversification import (
    analyze_allocation,
    calculate_concentration_metrics,
    calculate_correlation_matrix,
    get_correlation_insights,
)
from engines.hedging import get_hedge_recommendations, apply_hedge_to_portfolio
from engines.portfolio_builder import build_portfolio
from engines.stress import get_stress_test_summary
from ui.charts import (
    create_allocation_pie,
    create_correlation_heatmap,
    create_drawdown_chart,
    create_equity_curve_chart,
    create_holdings_bar,
    create_stress_test_bars,
)
from ui.layouts import create_metrics_row


def register_callbacks(app):
    """Register all callbacks with the Dash app."""

    # =========================================================================
    # Portfolio Builder Callbacks
    # =========================================================================

    @app.callback(
        [Output("risk-parity-options", "style"),
         Output("strategic-options", "style"),
         Output("tilts-options", "style")],
        Input("strategy-dropdown", "value"),
    )
    def toggle_strategy_options(strategy):
        """Show/hide options based on selected strategy."""
        if strategy == "Risk Parity":
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif strategy == "Equal Weight":
            return {"display": "none"}, {"display": "none"}, {"display": "none"}
        else:  # Strategic
            return {"display": "none"}, {"display": "block"}, {"display": "block"}

    @app.callback(
        [Output("portfolio-store", "data"),
         Output("portfolio-display", "children"),
         Output("holdings-chart", "figure")],
        Input("generate-portfolio-btn", "n_clicks"),
        [State("strategy-dropdown", "value"),
         State("risk-profile-dropdown", "value"),
         State("max-weight-slider", "value"),
         State("us-tilt-slider", "value"),
         State("em-tilt-slider", "value"),
         State("excluded-tickers-dropdown", "value"),
         State("rp-universe-dropdown", "value"),
         State("rp-method-dropdown", "value")],
        prevent_initial_call=True,
    )
    def generate_portfolio(n_clicks, strategy, risk_profile, max_weight,
                          us_tilt, em_tilt, excluded, rp_universe, rp_method):
        """Generate portfolio based on user settings."""
        if not n_clicks:
            return no_update, no_update, no_update

        # Build constraints
        constraints = Constraints(
            max_weight_per_etf=max_weight,
            excluded_tickers=excluded or [],
        )

        # Build request based on strategy
        if strategy == "Risk Parity":
            request = PortfolioRequest(
                strategy=AllocationStrategy.RISK_PARITY,
                constraints=constraints,
                risk_parity_config=RiskParityConfig(
                    universe=rp_universe or "balanced",
                    method=rp_method or "inverse_vol",
                ),
            )
        elif strategy == "Equal Weight":
            request = PortfolioRequest(
                strategy=AllocationStrategy.EQUAL_WEIGHT,
                constraints=constraints,
            )
        else:
            # Strategic
            tilts = Tilts(
                regions={
                    "US": us_tilt or 0,
                    "Emerging Markets": em_tilt or 0,
                }
            )
            request = PortfolioRequest(
                strategy=AllocationStrategy.STRATEGIC,
                risk_profile=RiskProfile(risk_profile),
                constraints=constraints,
                tilts=tilts,
            )

        # Build portfolio
        portfolio = build_portfolio(request)

        # Create display
        holdings_data = [
            {"ticker": h.ticker, "weight": h.weight, "rationale": h.rationale}
            for h in portfolio.holdings
        ]

        display = html.Div([
            html.H5(f"Portfolio with {len(portfolio.holdings)} holdings"),
            html.Hr(),
            html.Ul([
                html.Li(f"{h.ticker}: {h.weight:.1%} - {h.rationale}")
                for h in portfolio.holdings
            ]),
            html.Hr(),
            html.Small([html.P(note) for note in portfolio.notes], className="text-muted"),
        ])

        # Create chart
        chart = create_holdings_bar(holdings_data)

        # Store data
        store_data = {
            "holdings": holdings_data,
            "notes": portfolio.notes,
        }

        return store_data, display, chart

    # =========================================================================
    # Backtest Callbacks
    # =========================================================================

    @app.callback(
        [Output("backtest-store", "data"),
         Output("metrics-cards", "children"),
         Output("equity-curve-chart", "figure"),
         Output("drawdown-chart", "figure"),
         Output("backtest-loading", "children")],
        Input("run-backtest-btn", "n_clicks"),
        [State("portfolio-store", "data"),
         State("backtest-date-range", "start_date"),
         State("backtest-date-range", "end_date"),
         State("rebalance-dropdown", "value"),
         State("benchmark-dropdown", "value"),
         State("costs-switch", "value")],
        prevent_initial_call=True,
    )
    def run_backtest_callback(n_clicks, portfolio_data, start_date, end_date,
                              rebalance, benchmark, include_costs):
        """Run backtest on the portfolio."""
        if not n_clicks or not portfolio_data:
            return no_update, no_update, no_update, no_update, "No portfolio generated"

        # Parse dates
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date.split("T")[0])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date.split("T")[0])

        # Build holdings
        holdings = [
            PortfolioHolding(ticker=h["ticker"], weight=h["weight"])
            for h in portfolio_data["holdings"]
        ]

        # Build request
        request = BacktestRequest(
            portfolio=holdings,
            start_date=start_date,
            end_date=end_date,
            rebalance=RebalanceFrequency(rebalance),
            benchmark=BenchmarkConfig(ticker=benchmark),
            costs=CostConfig(enabled=include_costs),
        )

        try:
            # Run backtest
            result = run_backtest(request)

            # Create metrics cards
            metrics_row = create_metrics_row(result.metrics.model_dump())

            # Create charts
            equity_chart = create_equity_curve_chart(result.equity_curve)
            drawdown_chart = create_drawdown_chart(result.drawdown_curve)

            # Store data
            store_data = {
                "equity_curve": result.equity_curve,
                "drawdown_curve": result.drawdown_curve,
                "metrics": result.metrics.model_dump(),
                "benchmark_metrics": result.benchmark_metrics.model_dump(),
            }

            return store_data, metrics_row, equity_chart, drawdown_chart, ""

        except Exception as e:
            error_msg = html.Div([
                html.P(f"Error running backtest: {str(e)}", className="text-danger"),
            ])
            return no_update, error_msg, no_update, no_update, ""

    # =========================================================================
    # Analytics Callbacks
    # =========================================================================

    @app.callback(
        [Output("detailed-metrics", "children"),
         Output("benchmark-comparison", "children")],
        Input("backtest-store", "data"),
    )
    def update_analytics(backtest_data):
        """Update analytics tab when backtest completes."""
        if not backtest_data:
            placeholder = html.P("Run a backtest to see detailed analytics",
                               className="text-muted")
            return placeholder, placeholder

        metrics = backtest_data.get("metrics", {})
        benchmark_metrics = backtest_data.get("benchmark_metrics", {})

        # Detailed metrics display
        detailed = html.Div([
            html.H5("Portfolio Performance"),
            html.Table([
                html.Tbody([
                    html.Tr([html.Td("Total Return"), html.Td(f"{metrics.get('total_return', 0):.2%}")]),
                    html.Tr([html.Td("CAGR"), html.Td(f"{metrics.get('cagr', 0):.2%}")]),
                    html.Tr([html.Td("Volatility"), html.Td(f"{metrics.get('volatility', 0):.2%}")]),
                    html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{metrics.get('sharpe_ratio', 0):.2f}")]),
                    html.Tr([html.Td("Sortino Ratio"), html.Td(f"{metrics.get('sortino_ratio', 0):.2f}")]),
                    html.Tr([html.Td("Max Drawdown"), html.Td(f"{metrics.get('max_drawdown', 0):.2%}")]),
                ])
            ], className="table table-sm"),
        ])

        # Benchmark comparison
        comparison = html.Div([
            html.H5("Portfolio vs Benchmark"),
            html.Table([
                html.Thead([
                    html.Tr([html.Th("Metric"), html.Th("Portfolio"), html.Th("Benchmark"), html.Th("Difference")])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("Total Return"),
                        html.Td(f"{metrics.get('total_return', 0):.2%}"),
                        html.Td(f"{benchmark_metrics.get('total_return', 0):.2%}"),
                        html.Td(f"{metrics.get('total_return', 0) - benchmark_metrics.get('total_return', 0):+.2%}"),
                    ]),
                    html.Tr([
                        html.Td("CAGR"),
                        html.Td(f"{metrics.get('cagr', 0):.2%}"),
                        html.Td(f"{benchmark_metrics.get('cagr', 0):.2%}"),
                        html.Td(f"{metrics.get('cagr', 0) - benchmark_metrics.get('cagr', 0):+.2%}"),
                    ]),
                    html.Tr([
                        html.Td("Volatility"),
                        html.Td(f"{metrics.get('volatility', 0):.2%}"),
                        html.Td(f"{benchmark_metrics.get('volatility', 0):.2%}"),
                        html.Td(f"{metrics.get('volatility', 0) - benchmark_metrics.get('volatility', 0):+.2%}"),
                    ]),
                    html.Tr([
                        html.Td("Sharpe"),
                        html.Td(f"{metrics.get('sharpe_ratio', 0):.2f}"),
                        html.Td(f"{benchmark_metrics.get('sharpe_ratio', 0):.2f}"),
                        html.Td(f"{metrics.get('sharpe_ratio', 0) - benchmark_metrics.get('sharpe_ratio', 0):+.2f}"),
                    ]),
                    html.Tr([
                        html.Td("Max Drawdown"),
                        html.Td(f"{metrics.get('max_drawdown', 0):.2%}"),
                        html.Td(f"{benchmark_metrics.get('max_drawdown', 0):.2%}"),
                        html.Td(f"{metrics.get('max_drawdown', 0) - benchmark_metrics.get('max_drawdown', 0):+.2%}"),
                    ]),
                ])
            ], className="table table-sm table-striped"),
        ])

        return detailed, comparison

    # =========================================================================
    # Diversification Callbacks
    # =========================================================================

    @app.callback(
        [Output("asset-class-pie", "figure"),
         Output("region-pie", "figure"),
         Output("sector-pie", "figure"),
         Output("concentration-metrics", "children"),
         Output("correlation-heatmap", "figure"),
         Output("stress-test-chart", "figure")],
        Input("portfolio-store", "data"),
        State("backtest-date-range", "start_date"),
        State("backtest-date-range", "end_date"),
    )
    def update_diversification(portfolio_data, start_date, end_date):
        """Update diversification tab when portfolio changes."""
        if not portfolio_data:
            empty_fig = {}
            return empty_fig, empty_fig, empty_fig, "", empty_fig, empty_fig

        # Parse dates
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date.split("T")[0])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date.split("T")[0])

        # Build portfolio object
        holdings = [
            PortfolioHolding(ticker=h["ticker"], weight=h["weight"])
            for h in portfolio_data["holdings"]
        ]
        portfolio = Portfolio(holdings=holdings)

        # Allocation breakdown
        allocation = analyze_allocation(portfolio)

        asset_pie = create_allocation_pie(
            allocation["by_asset_class"],
            "Asset Class Allocation"
        )
        region_pie = create_allocation_pie(
            allocation["by_region"],
            "Region Allocation"
        )
        sector_pie = create_allocation_pie(
            allocation["by_sector"],
            "Sector Allocation"
        )

        # Concentration metrics
        concentration = calculate_concentration_metrics(portfolio)
        concentration_display = html.Div([
            html.P([html.Strong("HHI: "), f"{concentration['hhi']:.4f}"]),
            html.P([html.Strong("Effective N: "), f"{concentration['effective_n']:.1f}"]),
            html.P([html.Strong("Top 1 Weight: "), f"{concentration['top_1_weight']:.1%}"]),
            html.P([html.Strong("Top 3 Weight: "), f"{concentration['top_3_weight']:.1%}"]),
            html.P([html.Strong("Holdings: "), f"{concentration['num_holdings']}"]),
        ])

        # Correlation matrix
        corr_matrix = calculate_correlation_matrix(portfolio, start_date, end_date)
        if corr_matrix is not None:
            corr_heatmap = create_correlation_heatmap(corr_matrix)
        else:
            corr_heatmap = {}

        # Stress tests
        stress_summary = get_stress_test_summary(portfolio)
        stress_results = stress_summary.get("hypothetical_tests", [])
        if stress_results:
            stress_chart = create_stress_test_bars(stress_results)
        else:
            stress_chart = {}

        return asset_pie, region_pie, sector_pie, concentration_display, corr_heatmap, stress_chart

    # =========================================================================
    # Hedging Callbacks
    # =========================================================================

    @app.callback(
        [Output("hedge-recommendations", "children"),
         Output("hedge-store", "data"),
         Output("hedge-select-dropdown", "options"),
         Output("apply-hedge-btn", "disabled")],
        Input("get-hedges-btn", "n_clicks"),
        [State("portfolio-store", "data"),
         State("hedge-aggressiveness", "value"),
         State("backtest-date-range", "start_date"),
         State("backtest-date-range", "end_date")],
        prevent_initial_call=True,
    )
    def get_hedges(n_clicks, portfolio_data, aggressiveness, start_date, end_date):
        """Get hedge recommendations."""
        if not n_clicks or not portfolio_data:
            return "Generate a portfolio first", None, [], True

        # Parse dates
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date.split("T")[0])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date.split("T")[0])

        # Build portfolio
        holdings = [
            PortfolioHolding(ticker=h["ticker"], weight=h["weight"])
            for h in portfolio_data["holdings"]
        ]
        portfolio = Portfolio(holdings=holdings)

        # Get recommendations
        recommendations = get_hedge_recommendations(
            portfolio, start_date, end_date, aggressiveness
        )

        if not recommendations:
            return html.P("No hedge recommendations at this time",
                         className="text-muted"), None, [], True

        # Create display
        rec_cards = []
        dropdown_options = []
        rec_data = []

        for i, rec in enumerate(recommendations):
            rec_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{rec.instrument} - {rec.instrument_name}"),
                        html.P([html.Strong("Risk: "), rec.risk_targeted]),
                        html.P([html.Strong("Suggested Weight: "), f"{rec.suggested_weight:.1%}"]),
                        html.P(rec.rationale, className="text-muted small"),
                    ]),
                ], className="mb-2")
            )

            dropdown_options.append({
                "label": f"{rec.instrument} ({rec.suggested_weight:.1%})",
                "value": i,
            })

            rec_data.append({
                "instrument": rec.instrument,
                "instrument_name": rec.instrument_name,
                "risk_targeted": rec.risk_targeted,
                "suggested_weight": rec.suggested_weight,
                "rationale": rec.rationale,
            })

        display = html.Div(rec_cards)

        return display, rec_data, dropdown_options, False

    @app.callback(
        Output("portfolio-store", "data", allow_duplicate=True),
        Input("apply-hedge-btn", "n_clicks"),
        [State("portfolio-store", "data"),
         State("hedge-store", "data"),
         State("hedge-select-dropdown", "value")],
        prevent_initial_call=True,
    )
    def apply_hedge(n_clicks, portfolio_data, hedge_data, selected_idx):
        """Apply selected hedge to portfolio."""
        if not n_clicks or not portfolio_data or not hedge_data or selected_idx is None:
            return no_update

        # Build portfolio
        holdings = [
            PortfolioHolding(ticker=h["ticker"], weight=h["weight"])
            for h in portfolio_data["holdings"]
        ]
        portfolio = Portfolio(holdings=holdings, notes=portfolio_data.get("notes", []))

        # Get selected hedge
        hedge = hedge_data[selected_idx]
        from domain.schemas import HedgeRecommendation
        rec = HedgeRecommendation(
            instrument=hedge["instrument"],
            instrument_name=hedge["instrument_name"],
            risk_targeted=hedge["risk_targeted"],
            suggested_weight=hedge["suggested_weight"],
            rationale=hedge["rationale"],
        )

        # Apply hedge
        new_portfolio = apply_hedge_to_portfolio(portfolio, rec)

        # Return updated store data
        return {
            "holdings": [
                {"ticker": h.ticker, "weight": h.weight, "rationale": h.rationale}
                for h in new_portfolio.holdings
            ],
            "notes": new_portfolio.notes,
        }

    # =========================================================================
    # Export Callbacks
    # =========================================================================

    @app.callback(
        Output("download-portfolio-csv", "data"),
        Input("export-portfolio-csv-btn", "n_clicks"),
        State("portfolio-store", "data"),
        prevent_initial_call=True,
    )
    def export_portfolio_csv(n_clicks, portfolio_data):
        """Export portfolio to CSV."""
        if not n_clicks or not portfolio_data:
            return no_update

        df = pd.DataFrame(portfolio_data["holdings"])
        return dict(content=df.to_csv(index=False), filename="portfolio.csv")

    @app.callback(
        Output("download-portfolio-json", "data"),
        Input("export-portfolio-json-btn", "n_clicks"),
        State("portfolio-store", "data"),
        prevent_initial_call=True,
    )
    def export_portfolio_json(n_clicks, portfolio_data):
        """Export portfolio to JSON."""
        if not n_clicks or not portfolio_data:
            return no_update

        return dict(content=json.dumps(portfolio_data, indent=2), filename="portfolio.json")

    @app.callback(
        Output("download-backtest-csv", "data"),
        Input("export-backtest-csv-btn", "n_clicks"),
        State("backtest-store", "data"),
        prevent_initial_call=True,
    )
    def export_backtest_csv(n_clicks, backtest_data):
        """Export backtest results to CSV."""
        if not n_clicks or not backtest_data:
            return no_update

        df = pd.DataFrame(backtest_data["equity_curve"])
        return dict(content=df.to_csv(index=False), filename="backtest_results.csv")

    @app.callback(
        Output("download-backtest-json", "data"),
        Input("export-backtest-json-btn", "n_clicks"),
        State("backtest-store", "data"),
        prevent_initial_call=True,
    )
    def export_backtest_json(n_clicks, backtest_data):
        """Export backtest results to JSON."""
        if not n_clicks or not backtest_data:
            return no_update

        return dict(content=json.dumps(backtest_data, indent=2), filename="backtest_results.json")

    @app.callback(
        Output("export-summary", "children"),
        [Input("portfolio-store", "data"),
         Input("backtest-store", "data")],
    )
    def update_export_summary(portfolio_data, backtest_data):
        """Update export summary."""
        items = []

        if portfolio_data:
            items.append(html.Li(f"Portfolio: {len(portfolio_data['holdings'])} holdings"))
        else:
            items.append(html.Li("Portfolio: Not generated", className="text-muted"))

        if backtest_data:
            items.append(html.Li(f"Backtest: {len(backtest_data['equity_curve'])} data points"))
        else:
            items.append(html.Li("Backtest: Not run", className="text-muted"))

        return html.Ul(items)
