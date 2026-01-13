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
        # Predefined strategies that don't need extra options
        predefined = ["60/40", "Growth", "Conservative", "Aggressive", "Income", "All Weather", "Equal Weight"]

        if strategy == "Risk Parity":
            return {"display": "block"}, {"display": "none"}, {"display": "none"}
        elif strategy in predefined:
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

        # Map strategy dropdown value to enum
        strategy_map = {
            "Strategic": AllocationStrategy.STRATEGIC,
            "Risk Parity": AllocationStrategy.RISK_PARITY,
            "Equal Weight": AllocationStrategy.EQUAL_WEIGHT,
            "60/40": AllocationStrategy.CLASSIC_60_40,
            "Growth": AllocationStrategy.GROWTH,
            "Conservative": AllocationStrategy.CONSERVATIVE,
            "Aggressive": AllocationStrategy.AGGRESSIVE,
            "Income": AllocationStrategy.INCOME,
            "All Weather": AllocationStrategy.ALL_WEATHER,
        }
        strategy_enum = strategy_map.get(strategy, AllocationStrategy.STRATEGIC)

        # Build request based on strategy
        if strategy == "Risk Parity":
            request = PortfolioRequest(
                strategy=strategy_enum,
                constraints=constraints,
                risk_parity_config=RiskParityConfig(
                    universe=rp_universe or "balanced",
                    method=rp_method or "inverse_vol",
                ),
            )
        elif strategy == "Strategic":
            # Strategic with tilts
            tilts = Tilts(
                regions={
                    "US": us_tilt or 0,
                    "Emerging Markets": em_tilt or 0,
                }
            )
            request = PortfolioRequest(
                strategy=strategy_enum,
                risk_profile=RiskProfile(risk_profile),
                constraints=constraints,
                tilts=tilts,
            )
        else:
            # All other predefined strategies
            request = PortfolioRequest(
                strategy=strategy_enum,
                constraints=constraints,
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
    # Portfolio Save/Load Callbacks
    # =========================================================================

    @app.callback(
        Output("save-load-collapse", "is_open"),
        Input("toggle-save-load-btn", "n_clicks"),
        State("save-load-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_save_load(n_clicks, is_open):
        """Toggle the save/load section."""
        return not is_open

    @app.callback(
        [Output("save-portfolio-feedback", "children"),
         Output("load-portfolio-dropdown", "options"),
         Output("saved-portfolios-list", "children")],
        [Input("save-portfolio-btn", "n_clicks"),
         Input("delete-portfolio-btn", "n_clicks"),
         Input("save-load-collapse", "is_open")],
        [State("save-portfolio-name", "value"),
         State("portfolio-store", "data"),
         State("load-portfolio-dropdown", "value"),
         State("strategy-dropdown", "value")],
        prevent_initial_call=True,
    )
    def handle_save_delete(save_clicks, delete_clicks, is_open,
                           portfolio_name, portfolio_data, selected_portfolio, strategy):
        """Handle saving and deleting portfolios."""
        from dash import ctx
        from services.portfolio_storage import (
            save_portfolio, delete_portfolio, list_portfolios, get_portfolio_names
        )

        feedback = ""
        triggered = ctx.triggered_id

        # Handle save
        if triggered == "save-portfolio-btn" and save_clicks:
            if not portfolio_data:
                feedback = dbc.Alert("Generate a portfolio first", color="warning", duration=3000)
            elif not portfolio_name:
                feedback = dbc.Alert("Enter a portfolio name", color="warning", duration=3000)
            else:
                result = save_portfolio(
                    name=portfolio_name,
                    holdings=portfolio_data.get("holdings", []),
                    notes=portfolio_data.get("notes", []),
                    strategy=strategy,
                )
                color = "success" if result["success"] else "danger"
                feedback = dbc.Alert(result["message"], color=color, duration=3000)

        # Handle delete
        elif triggered == "delete-portfolio-btn" and delete_clicks:
            if not selected_portfolio:
                feedback = dbc.Alert("Select a portfolio to delete", color="warning", duration=3000)
            else:
                result = delete_portfolio(selected_portfolio)
                color = "success" if result["success"] else "danger"
                feedback = dbc.Alert(result["message"], color=color, duration=3000)

        # Refresh dropdown options
        options = [{"label": name, "value": name} for name in get_portfolio_names()]

        # Build saved portfolios list
        portfolios = list_portfolios()
        if portfolios:
            portfolio_list = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Name"),
                        html.Th("Holdings"),
                        html.Th("Strategy"),
                        html.Th("Updated"),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(p["name"]),
                        html.Td(p["holdings_count"]),
                        html.Td(p["strategy"] or "-"),
                        html.Td(p["updated_at"][:10] if p["updated_at"] else "-"),
                    ])
                    for p in portfolios
                ]),
            ], striped=True, hover=True, size="sm")
        else:
            portfolio_list = html.P("No saved portfolios", className="text-muted")

        return feedback, options, portfolio_list

    @app.callback(
        [Output("portfolio-store", "data", allow_duplicate=True),
         Output("portfolio-display", "children", allow_duplicate=True),
         Output("holdings-chart", "figure", allow_duplicate=True),
         Output("load-portfolio-feedback", "children")],
        Input("load-portfolio-btn", "n_clicks"),
        State("load-portfolio-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_saved_portfolio(n_clicks, portfolio_name):
        """Load a saved portfolio."""
        from services.portfolio_storage import load_portfolio

        if not n_clicks or not portfolio_name:
            return no_update, no_update, no_update, dbc.Alert(
                "Select a portfolio to load", color="warning", duration=3000
            )

        portfolio_data = load_portfolio(portfolio_name)

        if not portfolio_data:
            return no_update, no_update, no_update, dbc.Alert(
                f"Portfolio '{portfolio_name}' not found", color="danger", duration=3000
            )

        holdings = portfolio_data.get("holdings", [])
        notes = portfolio_data.get("notes", [])
        strategy = portfolio_data.get("strategy", "Unknown")

        # Create display
        display = html.Div([
            html.H5(f"Portfolio: {portfolio_name}"),
            html.P(f"Strategy: {strategy}", className="text-muted"),
            html.Hr(),
            html.Ul([
                html.Li(f"{h['ticker']}: {h['weight']:.1%} - {h.get('rationale', '')}")
                for h in holdings
            ]),
            html.Hr(),
            html.Small([html.P(note) for note in notes], className="text-muted"),
        ])

        # Create chart
        chart = create_holdings_bar(holdings)

        # Store data
        store_data = {
            "holdings": holdings,
            "notes": notes,
        }

        feedback = dbc.Alert(f"Loaded '{portfolio_name}'", color="success", duration=3000)

        return store_data, display, chart, feedback

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

    # =========================================================================
    # Strategy Comparison Callbacks
    # =========================================================================

    @app.callback(
        [Output("comparison-store", "data"),
         Output("comparison-metrics-table", "children"),
         Output("comparison-equity-chart", "figure"),
         Output("comparison-drawdown-chart", "figure"),
         Output("comparison-loading", "children")],
        Input("run-comparison-btn", "n_clicks"),
        [State("compare-date-range", "start_date"),
         State("compare-date-range", "end_date"),
         State("compare-rebalance-dropdown", "value"),
         State("compare-strategies-checklist", "value")],
        prevent_initial_call=True,
    )
    def run_strategy_comparison(n_clicks, start_date, end_date, rebalance, strategies):
        """Run comparison of multiple strategies."""
        from ui.charts import create_comparison_equity_chart, create_comparison_drawdown_chart

        if not n_clicks or not strategies:
            return no_update, no_update, no_update, no_update, "Select at least one strategy"

        # Parse dates
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date.split("T")[0])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date.split("T")[0])

        # Map strategy names to enums
        strategy_map = {
            "Strategic": AllocationStrategy.STRATEGIC,
            "Risk Parity": AllocationStrategy.RISK_PARITY,
            "Equal Weight": AllocationStrategy.EQUAL_WEIGHT,
            "60/40": AllocationStrategy.CLASSIC_60_40,
            "Growth": AllocationStrategy.GROWTH,
            "Conservative": AllocationStrategy.CONSERVATIVE,
            "Aggressive": AllocationStrategy.AGGRESSIVE,
            "Income": AllocationStrategy.INCOME,
            "All Weather": AllocationStrategy.ALL_WEATHER,
        }

        results = {}
        equity_curves = {}
        drawdown_curves = {}
        metrics_data = []

        for strategy_name in strategies:
            strategy_enum = strategy_map.get(strategy_name)
            if not strategy_enum:
                continue

            # Build portfolio for this strategy
            if strategy_name == "Risk Parity":
                request = PortfolioRequest(
                    strategy=strategy_enum,
                    risk_parity_config=RiskParityConfig(
                        universe="balanced",
                        method="inverse_vol",
                    ),
                )
            else:
                request = PortfolioRequest(strategy=strategy_enum)

            portfolio = build_portfolio(request)

            if not portfolio.holdings:
                continue

            # Run backtest
            holdings = [
                PortfolioHolding(ticker=h.ticker, weight=h.weight)
                for h in portfolio.holdings
            ]

            backtest_request = BacktestRequest(
                portfolio=holdings,
                start_date=start_date,
                end_date=end_date,
                rebalance=RebalanceFrequency(rebalance),
                benchmark=BenchmarkConfig(ticker="SPY"),
                costs=CostConfig(enabled=True),
            )

            try:
                result = run_backtest(backtest_request)

                # Store results
                results[strategy_name] = result
                equity_curves[strategy_name] = result.equity_curve
                drawdown_curves[strategy_name] = result.drawdown_curve

                # Collect metrics
                metrics_data.append({
                    "Strategy": strategy_name,
                    "Total Return": f"{result.metrics.total_return:.1%}",
                    "CAGR": f"{result.metrics.cagr:.1%}",
                    "Volatility": f"{result.metrics.volatility:.1%}",
                    "Sharpe": f"{result.metrics.sharpe_ratio:.2f}",
                    "Sortino": f"{result.metrics.sortino_ratio:.2f}",
                    "Max DD": f"{result.metrics.max_drawdown:.1%}",
                })
            except Exception as e:
                # Skip failed backtests
                continue

        if not results:
            return no_update, "No valid results", {}, {}, "Backtest failed for all strategies"

        # Create metrics table
        metrics_df = pd.DataFrame(metrics_data)

        # Sort by Sharpe ratio (descending)
        metrics_df["_sharpe_num"] = metrics_df["Sharpe"].apply(lambda x: float(x))
        metrics_df = metrics_df.sort_values("_sharpe_num", ascending=False)
        metrics_df = metrics_df.drop("_sharpe_num", axis=1)

        table = dbc.Table.from_dataframe(
            metrics_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="small",
        )

        # Create charts
        equity_chart = create_comparison_equity_chart(equity_curves)
        drawdown_chart = create_comparison_drawdown_chart(drawdown_curves)

        # Store data for export
        store_data = {
            "metrics": metrics_data,
            "strategies": strategies,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        return store_data, table, equity_chart, drawdown_chart, f"Compared {len(results)} strategies"

    # =========================================================================
    # Advanced Analytics Callbacks
    # =========================================================================

    @app.callback(
        [Output("advanced-analytics-store", "data"),
         Output("monte-carlo-chart", "figure"),
         Output("monte-carlo-stats", "children"),
         Output("rolling-returns-chart", "figure"),
         Output("risk-metrics-display", "children"),
         Output("calendar-heatmap", "figure"),
         Output("dividend-summary", "children"),
         Output("dividend-chart", "figure"),
         Output("contribution-chart", "figure"),
         Output("advanced-loading", "children")],
        Input("run-advanced-btn", "n_clicks"),
        [State("portfolio-store", "data"),
         State("advanced-date-range", "start_date"),
         State("advanced-date-range", "end_date"),
         State("mc-years-slider", "value"),
         State("mc-initial-input", "value")],
        prevent_initial_call=True,
    )
    def run_advanced_analytics(n_clicks, portfolio_data, start_date, end_date,
                               mc_years, initial_investment):
        """Run all advanced analytics."""
        from engines.advanced_analytics import (
            run_monte_carlo_simulation,
            calculate_rolling_returns,
            calculate_advanced_risk_metrics,
            calculate_calendar_returns,
            analyze_dividend_yield,
            calculate_contribution_analysis,
        )
        from ui.charts import (
            create_monte_carlo_chart,
            create_rolling_returns_chart,
            create_calendar_heatmap,
            create_yield_chart,
            create_contribution_chart,
        )

        if not n_clicks or not portfolio_data:
            return (no_update,) * 10

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

        # Run Monte Carlo
        mc_result = run_monte_carlo_simulation(
            portfolio, start_date, end_date,
            projection_years=mc_years,
            initial_investment=initial_investment or 10000,
        )
        mc_chart = create_monte_carlo_chart(mc_result)

        # Monte Carlo stats display
        if "statistics" in mc_result:
            stats = mc_result["statistics"]
            mc_stats = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Median Final Value", className="text-muted"),
                            html.H4(f"${stats['median_final']:,.0f}"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("5th Percentile", className="text-muted"),
                            html.H4(f"${stats['p5_final']:,.0f}", className="text-danger"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("95th Percentile", className="text-muted"),
                            html.H4(f"${stats['p95_final']:,.0f}", className="text-success"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Prob. of Loss", className="text-muted"),
                            html.H4(f"{stats['prob_loss']:.1%}"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Prob. of 2x", className="text-muted"),
                            html.H4(f"{stats['prob_double']:.1%}"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Ann. Return", className="text-muted"),
                            html.H4(f"{mc_result['historical_stats']['annualized_return']:.1%}"),
                        ]),
                    ], className="text-center"),
                ], md=2),
            ])
        else:
            mc_stats = html.P("Unable to calculate statistics", className="text-muted")

        # Rolling Returns
        rolling_result = calculate_rolling_returns(portfolio, start_date, end_date)
        rolling_chart = create_rolling_returns_chart(rolling_result)

        # Risk Metrics
        risk_result = calculate_advanced_risk_metrics(portfolio, start_date, end_date)
        if "error" not in risk_result:
            risk_display = html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Value at Risk (VaR)"),
                            dbc.CardBody([
                                html.P([html.Strong("Daily VaR (95%): "),
                                       f"{risk_result['var']['daily']:.2%}"]),
                                html.P([html.Strong("Annual VaR: "),
                                       f"{risk_result['var']['annual']:.2%}"]),
                            ]),
                        ]),
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Expected Shortfall (CVaR)"),
                            dbc.CardBody([
                                html.P([html.Strong("Daily CVaR: "),
                                       f"{risk_result['cvar']['daily']:.2%}"]),
                                html.P([html.Strong("Annual CVaR: "),
                                       f"{risk_result['cvar']['annual']:.2%}"]),
                            ]),
                        ]),
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Distribution"),
                            dbc.CardBody([
                                html.P([html.Strong("Skewness: "),
                                       f"{risk_result['skewness']:.2f}"]),
                                html.P([html.Strong("Kurtosis: "),
                                       f"{risk_result['kurtosis']:.2f}"]),
                            ]),
                        ]),
                    ], md=4),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Benchmark Relative"),
                            dbc.CardBody([
                                html.P([html.Strong("Beta: "),
                                       f"{risk_result['beta']:.2f}" if risk_result['beta'] else "N/A"]),
                                html.P([html.Strong("Alpha: "),
                                       f"{risk_result['alpha']:.2%}" if risk_result['alpha'] else "N/A"]),
                                html.P([html.Strong("R-squared: "),
                                       f"{risk_result['r_squared']:.2%}" if risk_result['r_squared'] else "N/A"]),
                            ]),
                        ]),
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Tracking"),
                            dbc.CardBody([
                                html.P([html.Strong("Tracking Error: "),
                                       f"{risk_result['tracking_error']:.2%}" if risk_result['tracking_error'] else "N/A"]),
                                html.P([html.Strong("Information Ratio: "),
                                       f"{risk_result['information_ratio']:.2f}" if risk_result['information_ratio'] else "N/A"]),
                            ]),
                        ]),
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Risk-Adjusted"),
                            dbc.CardBody([
                                html.P([html.Strong("Downside Dev: "),
                                       f"{risk_result['downside_deviation']:.2%}"]),
                                html.P([html.Strong("Calmar Ratio: "),
                                       f"{risk_result['calmar_ratio']:.2f}"]),
                            ]),
                        ]),
                    ], md=4),
                ]),
            ])
        else:
            risk_display = html.P("Unable to calculate risk metrics", className="text-muted")

        # Calendar Returns
        calendar_result = calculate_calendar_returns(portfolio, start_date, end_date)
        calendar_chart = create_calendar_heatmap(calendar_result)

        # Dividend Analysis
        dividend_result = analyze_dividend_yield(portfolio)
        dividend_summary = html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{dividend_result['portfolio_yield']:.2%}",
                           className="text-primary"),
                    html.P("Portfolio Yield", className="text-muted mb-0"),
                ]),
            ], className="text-center mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"${dividend_result['annual_income_per_10k']:,.0f}"),
                    html.P("Annual Income per $10K", className="text-muted mb-0"),
                ]),
            ], className="text-center mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"${dividend_result['monthly_income_per_10k']:,.0f}"),
                    html.P("Monthly Income per $10K", className="text-muted mb-0"),
                ]),
            ], className="text-center"),
        ])
        dividend_chart = create_yield_chart(dividend_result)

        # Contribution Analysis
        contribution_result = calculate_contribution_analysis(portfolio, start_date, end_date)
        contribution_chart = create_contribution_chart(contribution_result)

        # Store data
        store_data = {
            "monte_carlo": mc_result.get("statistics", {}),
            "risk_metrics": risk_result if "error" not in risk_result else {},
            "dividend": dividend_result,
        }

        return (store_data, mc_chart, mc_stats, rolling_chart, risk_display,
                calendar_chart, dividend_summary, dividend_chart, contribution_chart,
                "Analysis complete")
