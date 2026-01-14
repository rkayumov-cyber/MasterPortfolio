"""Dash callbacks for the ETF Portfolio Tool."""

import json
from datetime import date, timedelta

import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, html, no_update, ALL

from domain.schemas import (
    AllocationStrategy,
    BacktestRequest,
    BenchmarkConfig,
    Constraints,
    CostConfig,
    OptimizationObjective,
    OptimizerConfig,
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
    create_allocation_comparison_chart,
    create_allocation_pie,
    create_correlation_heatmap,
    create_drawdown_chart,
    create_efficient_frontier_chart,
    create_equity_curve_chart,
    create_fear_greed_gauge,
    create_fred_combined_chart,
    create_holdings_bar,
    create_optimal_weights_chart,
    create_optimizer_comparison_chart,
    create_regime_gauge,
    create_sentiment_bars,
    create_stress_test_bars,
)
from ui.layouts import create_metrics_row

# Market Regime imports
from engines.regime_detector import detect_regime, get_regime_summary
from engines.regime_tilts import (
    generate_regime_recommendation,
    get_recommendation_summary,
    create_default_portfolio,
    apply_regime_tilts,
)
from services.sentiment_client import (
    fetch_fear_greed_index,
    fetch_market_analyst_consensus,
    get_aggregate_sentiment,
)
from services.pdf_processor import (
    process_research_pdf,
    is_pdf_support_available,
    get_sentiment_score,
)


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

    # =========================================================================
    # Rebalancing Calculator Callbacks
    # =========================================================================

    @app.callback(
        Output("current-holdings-inputs", "children"),
        Input("portfolio-store", "data"),
    )
    def generate_holdings_inputs(portfolio_data):
        """Generate input fields for current holdings."""
        if not portfolio_data or not portfolio_data.get("holdings"):
            return html.P("Generate a portfolio first", className="text-muted")

        inputs = []
        for holding in portfolio_data["holdings"]:
            ticker = holding["ticker"]
            inputs.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Label(ticker, className="small"),
                    ], width=3),
                    dbc.Col([
                        dbc.Input(
                            id={"type": "holding-shares", "ticker": ticker},
                            type="number",
                            value=0,
                            min=0,
                            step=0.01,
                            size="sm",
                            placeholder="Shares",
                        ),
                    ], width=9),
                ], className="mb-1")
            )

        return html.Div(inputs)

    @app.callback(
        [Output("rebalance-summary", "children"),
         Output("rebalance-trades-table", "children"),
         Output("expense-analysis", "children"),
         Output("rebalance-store", "data")],
        Input("calculate-rebalance-btn", "n_clicks"),
        [State("portfolio-store", "data"),
         State("rebalance-portfolio-value", "value"),
         State("rebalance-cash", "value"),
         State("rebalance-min-trade", "value"),
         State({"type": "holding-shares", "ticker": ALL}, "value"),
         State({"type": "holding-shares", "ticker": ALL}, "id")],
        prevent_initial_call=True,
    )
    def calculate_rebalance(n_clicks, portfolio_data, portfolio_value, cash,
                           min_trade, shares_values, shares_ids):
        """Calculate rebalancing trades and expense analysis."""
        from engines.rebalancing import (
            calculate_rebalance_trades,
            analyze_portfolio_expenses,
        )

        if not n_clicks or not portfolio_data:
            return (
                html.P("Generate a portfolio first", className="text-muted"),
                html.P("No trades to display", className="text-muted"),
                html.P("No expense data", className="text-muted"),
                None,
            )

        # Build portfolio
        holdings = [
            PortfolioHolding(ticker=h["ticker"], weight=h["weight"])
            for h in portfolio_data["holdings"]
        ]
        portfolio = Portfolio(holdings=holdings)

        # Parse current holdings
        current_holdings = {}
        if shares_values and shares_ids:
            for shares, id_dict in zip(shares_values, shares_ids):
                if shares and shares > 0:
                    ticker = id_dict.get("ticker")
                    if ticker:
                        current_holdings[ticker] = float(shares)

        # Calculate rebalance
        result = calculate_rebalance_trades(
            portfolio=portfolio,
            portfolio_value=portfolio_value or 10000,
            current_holdings=current_holdings if current_holdings else None,
            cash_balance=cash or 0,
            min_trade_value=min_trade or 25,
        )

        # Build summary cards
        summary = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${result.total_buys:,.0f}", className="text-success"),
                        html.P("Total Buys", className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${result.total_sells:,.0f}", className="text-danger"),
                        html.P("Total Sells", className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${abs(result.net_cash_flow):,.0f}",
                               className="text-primary"),
                        html.P("Net " + ("Need" if result.net_cash_flow > 0 else "Excess"),
                               className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{result.turnover:.1%}"),
                        html.P("Turnover", className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{result.num_trades}"),
                        html.P("Trades", className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${result.new_cash_balance:,.0f}",
                               className="text-info" if result.new_cash_balance >= 0 else "text-warning"),
                        html.P("Cash After", className="text-muted mb-0 small"),
                    ]),
                ], className="text-center"),
            ], md=2),
        ])

        # Build trades table
        if result.trades:
            trades_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Action"),
                        html.Th("Ticker"),
                        html.Th("Shares"),
                        html.Th("Price"),
                        html.Th("Trade Value"),
                        html.Th("Target Weight"),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            dbc.Badge(t.action, color="success" if t.action == "BUY" else "danger")
                        ),
                        html.Td([
                            html.Strong(t.ticker),
                            html.Br(),
                            html.Small(t.name[:25] + "..." if len(t.name) > 25 else t.name,
                                      className="text-muted"),
                        ]),
                        html.Td(f"{t.shares_to_trade:.2f}"),
                        html.Td(f"${t.price:.2f}"),
                        html.Td(f"${t.trade_value:,.0f}"),
                        html.Td(f"{t.target_weight:.1%}"),
                    ])
                    for t in result.trades
                ]),
            ], striped=True, hover=True, size="sm", responsive=True)
        else:
            trades_table = html.P("Portfolio is already balanced (no trades needed)",
                                 className="text-success")

        # Calculate expense analysis
        expense_result = analyze_portfolio_expenses(portfolio, portfolio_value or 10000)

        expense_display = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{expense_result.weighted_expense_ratio:.3%}",
                                   className="text-primary"),
                            html.P("Weighted Expense Ratio", className="text-muted mb-0"),
                        ]),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"${expense_result.annual_cost_per_10k:.0f}",
                                   className="text-warning"),
                            html.P(f"Annual Cost on ${portfolio_value or 10000:,}",
                                  className="text-muted mb-0"),
                        ]),
                    ], className="text-center"),
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"${expense_result.annual_cost_per_10k * 10:.0f}"),
                            html.P("10-Year Cost (Cumulative)", className="text-muted mb-0"),
                        ]),
                    ], className="text-center"),
                ], md=4),
            ], className="mb-3"),

            html.H6("Expense Breakdown by Holding", className="mt-3"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Ticker"),
                        html.Th("Weight"),
                        html.Th("Expense Ratio"),
                        html.Th("Annual Cost"),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(h["ticker"]),
                        html.Td(f"{h['weight']:.1%}"),
                        html.Td(f"{h['expense_ratio']:.2%}"),
                        html.Td(f"${h['annual_cost']:.2f}"),
                    ])
                    for h in expense_result.holdings_expenses
                ]),
            ], striped=True, size="sm"),

            dbc.Alert([
                html.Strong("Cheapest: "),
                f"{expense_result.cheapest_holding.get('ticker', 'N/A')} ",
                f"({expense_result.cheapest_holding.get('expense_ratio', 0):.2%})",
                html.Br(),
                html.Strong("Most Expensive: "),
                f"{expense_result.most_expensive_holding.get('ticker', 'N/A')} ",
                f"({expense_result.most_expensive_holding.get('expense_ratio', 0):.2%})",
            ], color="info", className="mt-3"),
        ])

        # Store data
        store_data = {
            "trades": [
                {
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": t.shares_to_trade,
                    "value": t.trade_value,
                }
                for t in result.trades
            ],
            "summary": {
                "total_buys": result.total_buys,
                "total_sells": result.total_sells,
                "turnover": result.turnover,
            },
            "expenses": {
                "weighted_ratio": expense_result.weighted_expense_ratio,
                "annual_cost": expense_result.annual_cost_per_10k,
            },
        }

        return summary, trades_table, expense_display, store_data

    # =========================================================================
    # ETF Screener Callbacks
    # =========================================================================

    @app.callback(
        [Output("screener-results", "children"),
         Output("screener-count", "children"),
         Output("screener-store", "data")],
        [Input("apply-screener-btn", "n_clicks"),
         Input("reset-screener-btn", "n_clicks")],
        [State("screener-search", "value"),
         State("screener-asset-class", "value"),
         State("screener-region", "value"),
         State("screener-sector", "value"),
         State("screener-max-expense", "value"),
         State("screener-tags", "value"),
         State("screener-exclude-inverse", "value")],
        prevent_initial_call=True,
    )
    def apply_screener_filters(apply_clicks, reset_clicks, search, asset_classes,
                               regions, sectors, max_expense, tags, exclude_inverse):
        """Apply screener filters and display results."""
        from dash import ctx
        from engines.screener import screen_etfs, ScreenerFilters

        triggered = ctx.triggered_id

        # If reset was clicked, use empty filters
        if triggered == "reset-screener-btn":
            filters = ScreenerFilters(exclude_inverse=True)
        else:
            # Build filters from inputs
            filters = ScreenerFilters(
                search_query=search if search else None,
                asset_classes=asset_classes if asset_classes else None,
                regions=regions if regions else None,
                sectors=sectors if sectors else None,
                max_expense_ratio=max_expense if max_expense and max_expense < 1.0 else None,
                tags=tags if tags else None,
                exclude_inverse="exclude" in (exclude_inverse or []),
            )

        # Run screener
        result = screen_etfs(filters)

        # Build results table
        if result.etfs:
            table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Ticker"),
                        html.Th("Name"),
                        html.Th("Asset Class"),
                        html.Th("Region"),
                        html.Th("Expense"),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td([
                            dbc.Button(
                                etf.ticker,
                                id={"type": "etf-ticker-btn", "ticker": etf.ticker},
                                color="link",
                                size="sm",
                                className="p-0",
                            ),
                        ]),
                        html.Td(etf.name[:40] + "..." if len(etf.name) > 40 else etf.name,
                               className="small"),
                        html.Td(dbc.Badge(etf.asset_class.value, color="primary"),
                               className="small"),
                        html.Td(etf.region.value, className="small"),
                        html.Td(f"{etf.expense_ratio:.2f}%" if etf.expense_ratio else "-",
                               className="small"),
                    ])
                    for etf in result.etfs[:50]  # Limit to 50 results
                ]),
            ], striped=True, hover=True, size="sm", responsive=True)

            if result.total_count > 50:
                results_display = html.Div([
                    table,
                    html.P(f"Showing 50 of {result.total_count} results",
                          className="text-muted small mt-2"),
                ])
            else:
                results_display = table
        else:
            results_display = html.P("No ETFs match the filters", className="text-muted")

        # Store the filter results
        store_data = {
            "tickers": [etf.ticker for etf in result.etfs],
            "count": result.total_count,
            "filters": result.filters_applied,
        }

        return results_display, str(result.total_count), store_data

    @app.callback(
        Output("etf-details-display", "children"),
        Input({"type": "etf-ticker-btn", "ticker": ALL}, "n_clicks"),
        State({"type": "etf-ticker-btn", "ticker": ALL}, "id"),
        prevent_initial_call=True,
    )
    def show_etf_details(clicks, ids):
        """Display details for selected ETF."""
        from dash import ctx
        from engines.screener import get_etf_details, find_similar_etfs, get_low_cost_alternatives

        # Check which button was clicked
        if not ctx.triggered or not any(clicks):
            return html.P("Click on an ETF ticker to see details", className="text-muted")

        # Find the triggered button
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return no_update

        ticker = triggered_id.get("ticker")
        if not ticker:
            return no_update

        # Get ETF details
        details = get_etf_details(ticker)
        if not details:
            return html.P(f"Details not found for {ticker}", className="text-danger")

        # Get similar ETFs
        similar = find_similar_etfs(ticker, limit=3)

        # Get low-cost alternatives
        alternatives = get_low_cost_alternatives(ticker)[:3]

        # Build details display
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4([
                        details["ticker"],
                        dbc.Badge(details["asset_class"], color="primary", className="ms-2"),
                    ]),
                    html.P(details["name"], className="text-muted"),
                ], md=8),
                dbc.Col([
                    html.H5(f"{details['expense_ratio']:.2f}%" if details['expense_ratio'] else "N/A",
                           className="text-end"),
                    html.P("Expense Ratio", className="text-muted text-end small"),
                ], md=4),
            ]),

            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.Strong("Region: "),
                    details["region"],
                ], md=4),
                dbc.Col([
                    html.Strong("Sector: "),
                    details["sector"],
                ], md=4),
                dbc.Col([
                    html.Strong("Inception: "),
                    str(details.get("inception_year") or "N/A"),
                ], md=4),
            ], className="mb-2"),

            dbc.Row([
                dbc.Col([
                    html.Strong("AUM: "),
                    f"${details.get('aum_billions') or 0}B",
                ], md=4),
                dbc.Col([
                    html.Strong("Avg Volume: "),
                    f"{details.get('avg_volume_millions') or 0}M",
                ], md=4),
                dbc.Col([
                    html.Strong("Yield: "),
                    f"{details.get('dividend_yield') or 0:.1f}%",
                ], md=4),
            ], className="mb-2"),

            # Tags
            html.Div([
                dbc.Badge(tag, color="secondary", className="me-1")
                for tag in details.get("tags", [])
            ], className="mb-3"),

            # Similar ETFs
            html.Hr(),
            html.H6("Similar ETFs"),
            html.Div([
                dbc.Badge(
                    f"{etf.ticker} ({etf.expense_ratio:.2f}%)" if etf.expense_ratio else etf.ticker,
                    color="info",
                    className="me-1",
                )
                for etf in similar
            ]) if similar else html.P("No similar ETFs found", className="text-muted small"),

            # Low-cost alternatives
            html.Hr() if alternatives else "",
            html.H6("Lower-Cost Alternatives") if alternatives else "",
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Ticker"),
                        html.Th("Expense"),
                        html.Th("Savings"),
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(alt["ticker"]),
                        html.Td(f"{alt['expense_ratio']:.2f}%"),
                        html.Td(f"{alt['savings_bps']:.2f} bps", className="text-success"),
                    ])
                    for alt in alternatives
                ]),
            ], size="sm", striped=True) if alternatives else "",
        ])

    # =========================================================================
    # Portfolio Import Callbacks
    # =========================================================================

    @app.callback(
        Output("import-collapse", "is_open"),
        Input("toggle-import-btn", "n_clicks"),
        State("import-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_import_section(n_clicks, is_open):
        """Toggle the import section."""
        return not is_open

    @app.callback(
        [Output("import-status", "children"),
         Output("import-preview", "children"),
         Output("use-import-btn", "style"),
         Output("clear-import-btn", "style"),
         Output("csv-upload", "contents", allow_duplicate=True)],
        [Input("csv-upload", "contents"),
         Input("clear-import-btn", "n_clicks")],
        [State("csv-upload", "filename")],
        prevent_initial_call=True,
    )
    def handle_csv_upload(contents, clear_clicks, filename):
        """Handle CSV file upload and parse contents."""
        from dash import ctx
        from services.portfolio_import import parse_uploaded_file, convert_to_portfolio_weights

        triggered = ctx.triggered_id
        show_btn = {"display": "inline-block"}
        hide_btn = {"display": "none"}

        # Handle clear button
        if triggered == "clear-import-btn":
            return "", "", hide_btn, hide_btn, None

        if not contents:
            return "", "", hide_btn, hide_btn, no_update

        # Parse the uploaded file
        result = parse_uploaded_file(contents, filename or "upload.csv")

        if not result.success:
            status = dbc.Alert(
                [html.Strong("Import Error: "), result.error],
                color="danger",
            )
            return status, "", hide_btn, hide_btn, no_update

        # Convert to weights
        holdings, total_value, warnings = convert_to_portfolio_weights(result.holdings)

        if not holdings:
            status = dbc.Alert(
                "Could not calculate portfolio weights",
                color="danger",
            )
            return status, "", hide_btn, hide_btn, no_update

        # Build status message
        status_items = [
            f"Found {len(result.holdings)} holdings",
            f"Total value: ${total_value:,.0f}",
            f"Matched ETFs: {result.matched_etfs}/{len(result.holdings)}",
        ]
        if result.unmatched_tickers:
            status_items.append(f"Unmatched: {len(result.unmatched_tickers)}")

        status = dbc.Alert(
            [
                html.Strong("Import Successful! "),
                " | ".join(status_items),
            ],
            color="success",
        )

        # Build warnings if any
        warning_display = ""
        all_warnings = result.warnings + warnings
        if all_warnings:
            warning_display = dbc.Alert(
                [html.Ul([html.Li(w, className="small") for w in all_warnings])],
                color="warning",
                className="mt-2",
            )

        # Build preview table
        preview_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Ticker"),
                    html.Th("Weight"),
                    html.Th("Value"),
                    html.Th("Shares"),
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(h["ticker"]),
                    html.Td(f"{h['weight']:.1%}"),
                    html.Td(f"${h['value']:,.0f}"),
                    html.Td(f"{h['shares']:.2f}"),
                ])
                for h in holdings[:10]
            ]),
        ], striped=True, hover=True, size="sm")

        if len(holdings) > 10:
            preview_note = html.P(
                f"Showing 10 of {len(holdings)} holdings",
                className="text-muted small"
            )
        else:
            preview_note = ""

        preview = html.Div([
            html.H6("Preview"),
            preview_table,
            preview_note,
            warning_display,
            # Store the parsed data in a hidden div for the use button
            dcc.Store(id="import-data-store", data={
                "holdings": holdings,
                "total_value": total_value,
            }),
        ])

        return status, preview, show_btn, show_btn, no_update

    @app.callback(
        [Output("portfolio-store", "data", allow_duplicate=True),
         Output("portfolio-display", "children", allow_duplicate=True),
         Output("holdings-chart", "figure", allow_duplicate=True),
         Output("import-status", "children", allow_duplicate=True),
         Output("import-preview", "children", allow_duplicate=True),
         Output("use-import-btn", "style", allow_duplicate=True),
         Output("clear-import-btn", "style", allow_duplicate=True)],
        Input("use-import-btn", "n_clicks"),
        State("import-data-store", "data"),
        prevent_initial_call=True,
    )
    def use_imported_portfolio(n_clicks, import_data):
        """Use the imported portfolio as the current portfolio."""
        if not n_clicks or not import_data:
            return (no_update,) * 7

        holdings = import_data.get("holdings", [])
        total_value = import_data.get("total_value", 0)

        if not holdings:
            return (no_update,) * 7

        # Create display
        display = html.Div([
            html.H5(f"Imported Portfolio ({len(holdings)} holdings)"),
            html.P(f"Total Value: ${total_value:,.0f}", className="text-muted"),
            html.Hr(),
            html.Ul([
                html.Li(f"{h['ticker']}: {h['weight']:.1%} - {h.get('rationale', '')}")
                for h in holdings
            ]),
        ])

        # Create chart
        chart = create_holdings_bar(holdings)

        # Store data
        store_data = {
            "holdings": holdings,
            "notes": [f"Imported from CSV. Total value: ${total_value:,.0f}"],
        }

        # Clear import UI
        success_msg = dbc.Alert(
            "Portfolio imported successfully! You can now backtest or analyze it.",
            color="success",
        )

        hide_btn = {"display": "none"}

        return store_data, display, chart, success_msg, "", hide_btn, hide_btn

    # =========================================================================
    # Portfolio Optimizer Callbacks
    # =========================================================================

    @app.callback(
        Output("etf-count-feedback", "children"),
        Input("optimizer-etf-dropdown", "value"),
    )
    def update_etf_count(selected_etfs):
        """Show feedback on ETF selection count."""
        if not selected_etfs:
            return dbc.Alert(
                "Select 2-8 ETFs to optimize",
                color="info",
                className="py-1 mb-0",
            )

        count = len(selected_etfs)
        if count < 2:
            return dbc.Alert(
                f"Select at least 2 ETFs ({count} selected)",
                color="warning",
                className="py-1 mb-0",
            )
        elif count > 8:
            return dbc.Alert(
                f"Maximum 8 ETFs allowed ({count} selected)",
                color="danger",
                className="py-1 mb-0",
            )
        else:
            return dbc.Alert(
                f"{count} ETFs selected",
                color="success",
                className="py-1 mb-0",
            )

    @app.callback(
        Output("search-space-estimate", "children"),
        [Input("optimizer-etf-dropdown", "value"),
         Input("optimizer-weight-step", "value"),
         Input("optimizer-min-weight", "value"),
         Input("optimizer-max-weight", "value")],
    )
    def estimate_search_space(etfs, step, min_w, max_w):
        """Estimate and display search space size."""
        if not etfs or len(etfs) < 2:
            return ""

        from engines.optimizer import estimate_search_space_size

        n = len(etfs)
        try:
            size = estimate_search_space_size(n, step, min_w, max_w)

            if size > 100000:
                warning = " (may take a while)"
                color = "warning"
            elif size > 10000:
                warning = ""
                color = "info"
            else:
                warning = ""
                color = "success"

            return dbc.Alert(
                f"Search space: {size:,} combinations{warning}",
                color=color,
                className="py-1 mb-0",
            )
        except Exception:
            return ""

    @app.callback(
        [Output("optimizer-store", "data"),
         Output("optimal-portfolio-summary", "children"),
         Output("optimizer-metrics-cards", "children"),
         Output("efficient-frontier-chart", "figure"),
         Output("optimal-weights-chart", "figure"),
         Output("optimizer-comparison-chart", "figure"),
         Output("all-portfolios-table", "children"),
         Output("use-optimal-portfolio-btn", "disabled"),
         Output("optimizer-loading", "children")],
        Input("run-optimizer-btn", "n_clicks"),
        [State("optimizer-etf-dropdown", "value"),
         State("optimizer-objective", "value"),
         State("optimizer-date-range", "start_date"),
         State("optimizer-date-range", "end_date"),
         State("optimizer-weight-step", "value"),
         State("optimizer-min-weight", "value"),
         State("optimizer-max-weight", "value")],
        prevent_initial_call=True,
    )
    def run_optimization_callback(n_clicks, etfs, objective, start_date, end_date,
                                  weight_step, min_weight, max_weight):
        """Run portfolio optimization."""
        from engines.optimizer import run_optimization
        from domain.schemas import OptimizationObjective, OptimizerConfig
        from ui.charts import (
            create_efficient_frontier_chart,
            create_optimal_weights_chart,
            create_optimizer_comparison_chart,
        )

        if not n_clicks or not etfs or len(etfs) < 2:
            return (no_update,) * 9

        if len(etfs) > 8:
            error_msg = html.P("Maximum 8 ETFs allowed", className="text-danger")
            return None, error_msg, "", {}, {}, {}, "", True, ""

        # Parse dates
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date.split("T")[0])
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date.split("T")[0])

        # Map objective string to enum
        objective_map = {
            "MAX_SHARPE": OptimizationObjective.MAX_SHARPE,
            "MAX_CAGR": OptimizationObjective.MAX_CAGR,
            "MIN_VOLATILITY": OptimizationObjective.MIN_VOLATILITY,
        }
        objective_enum = objective_map.get(objective, OptimizationObjective.MAX_SHARPE)

        try:
            config = OptimizerConfig(
                tickers=etfs,
                objective=objective_enum,
                weight_step=weight_step,
                min_weight=min_weight,
                max_weight=max_weight,
                start_date=start_date,
                end_date=end_date,
            )

            result = run_optimization(config)

            # Build optimal portfolio summary
            best = result.best_portfolio
            summary = html.Div([
                html.H5(f"Objective: {result.objective.value}"),
                html.P(
                    f"Searched {result.search_space_size:,} combinations in "
                    f"{result.computation_time_seconds:.1f} seconds"
                ),
                html.Hr(),
                html.H6("Optimal Allocation:"),
                html.Ul([
                    html.Li(f"{ticker}: {weight:.1%}")
                    for ticker, weight in sorted(
                        best.weights.items(),
                        key=lambda x: -x[1]
                    )
                    if weight > 0.001
                ]),
            ])

            # Metrics cards
            metrics_row = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{best.sharpe_ratio:.2f}", className="text-primary mb-0"),
                            html.Small("Sharpe Ratio", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                f"{best.cagr:.1%}",
                                className="text-success mb-0" if best.cagr > 0 else "text-danger mb-0"
                            ),
                            html.Small("CAGR", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{best.volatility:.1%}", className="text-info mb-0"),
                            html.Small("Volatility", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(
                                f"{best.total_return:.1%}",
                                className="text-success mb-0" if best.total_return > 0 else "text-danger mb-0"
                            ),
                            html.Small("Total Return", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{best.max_drawdown:.1%}", className="text-danger mb-0"),
                            html.Small("Max Drawdown", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{result.search_space_size:,}", className="text-secondary mb-0"),
                            html.Small("Combinations", className="text-muted"),
                        ]),
                    ], className="text-center"),
                ], md=2),
            ])

            # Prepare data for efficient frontier chart
            all_portfolios_data = [
                {
                    "volatility": p.volatility,
                    "cagr": p.cagr,
                    "sharpe_ratio": p.sharpe_ratio,
                }
                for p in result.all_portfolios
            ]

            optimal_point = {
                "volatility": best.volatility,
                "return": best.cagr,
                "sharpe": best.sharpe_ratio,
            }
            frontier_chart = create_efficient_frontier_chart(
                all_portfolios_data,
                result.efficient_frontier,
                optimal_point=optimal_point,
            )

            # Weights pie chart
            weights_chart = create_optimal_weights_chart(best.weights)

            # Comparison chart - run backtest on optimal portfolio
            holdings = [
                PortfolioHolding(ticker=t, weight=w)
                for t, w in best.weights.items()
                if w > 0.001
            ]
            backtest_request = BacktestRequest(
                portfolio=holdings,
                start_date=start_date,
                end_date=end_date,
                rebalance=RebalanceFrequency.QUARTERLY,
                benchmark=BenchmarkConfig(ticker="SPY"),
                costs=CostConfig(enabled=False),
            )
            backtest_result = run_backtest(backtest_request)
            comparison_chart = create_optimizer_comparison_chart(
                backtest_result.equity_curve,
            )

            # All portfolios table (show top 50 by Sharpe)
            sorted_portfolios = sorted(
                result.all_portfolios,
                key=lambda p: p.sharpe_ratio,
                reverse=True,
            )[:50]

            table_rows = []
            for i, p in enumerate(sorted_portfolios):
                weights_str = ", ".join(
                    f"{t}:{w:.0%}"
                    for t, w in sorted(p.weights.items(), key=lambda x: -x[1])
                    if w > 0.001
                )
                if len(weights_str) > 60:
                    weights_str = weights_str[:60] + "..."
                table_rows.append(
                    html.Tr([
                        html.Td(str(i + 1)),
                        html.Td(f"{p.sharpe_ratio:.2f}"),
                        html.Td(f"{p.cagr:.1%}"),
                        html.Td(f"{p.volatility:.1%}"),
                        html.Td(f"{p.max_drawdown:.1%}"),
                        html.Td(weights_str, className="small"),
                    ])
                )

            all_table = dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Rank"),
                        html.Th("Sharpe"),
                        html.Th("CAGR"),
                        html.Th("Vol"),
                        html.Th("Max DD"),
                        html.Th("Weights"),
                    ])
                ]),
                html.Tbody(table_rows),
            ], striped=True, bordered=True, hover=True, responsive=True, size="sm")

            # Store data
            store_data = {
                "best_portfolio": {
                    "weights": best.weights,
                    "sharpe_ratio": best.sharpe_ratio,
                    "cagr": best.cagr,
                    "volatility": best.volatility,
                    "total_return": best.total_return,
                    "max_drawdown": best.max_drawdown,
                },
                "objective": objective,
                "search_space_size": result.search_space_size,
                "computation_time": result.computation_time_seconds,
            }

            loading_msg = f"Optimization complete ({result.computation_time_seconds:.1f}s)"

            return (
                store_data,
                summary,
                metrics_row,
                frontier_chart,
                weights_chart,
                comparison_chart,
                all_table,
                False,  # Enable use button
                loading_msg,
            )

        except Exception as e:
            error_msg = html.Div([
                html.P(f"Optimization failed: {str(e)}", className="text-danger"),
            ])
            return None, error_msg, "", {}, {}, {}, "", True, str(e)

    @app.callback(
        [Output("portfolio-store", "data", allow_duplicate=True),
         Output("portfolio-display", "children", allow_duplicate=True),
         Output("holdings-chart", "figure", allow_duplicate=True),
         Output("use-portfolio-feedback", "children")],
        Input("use-optimal-portfolio-btn", "n_clicks"),
        State("optimizer-store", "data"),
        prevent_initial_call=True,
    )
    def use_optimal_portfolio(n_clicks, optimizer_data):
        """Transfer optimal portfolio to main portfolio store."""
        if not n_clicks or not optimizer_data:
            return (no_update,) * 4

        best = optimizer_data.get("best_portfolio", {})
        weights = best.get("weights", {})

        if not weights:
            return (no_update,) * 4

        # Build holdings data
        holdings_data = [
            {"ticker": ticker, "weight": weight, "rationale": "Optimized allocation"}
            for ticker, weight in weights.items()
            if weight > 0.001
        ]

        # Create display
        display = html.Div([
            html.H5(f"Optimized Portfolio ({len(holdings_data)} holdings)"),
            html.P(
                f"Sharpe: {best['sharpe_ratio']:.2f} | "
                f"CAGR: {best['cagr']:.1%} | "
                f"Vol: {best['volatility']:.1%}",
                className="text-muted"
            ),
            html.Hr(),
            html.Ul([
                html.Li(f"{h['ticker']}: {h['weight']:.1%}")
                for h in sorted(holdings_data, key=lambda x: -x['weight'])
            ]),
        ])

        # Create chart
        chart = create_holdings_bar(holdings_data)

        # Store data
        store_data = {
            "holdings": holdings_data,
            "notes": [
                f"Optimized for {optimizer_data.get('objective', 'MAX_SHARPE')}",
                f"Sharpe Ratio: {best['sharpe_ratio']:.2f}",
                f"Expected CAGR: {best['cagr']:.1%}",
            ],
        }

        feedback = dbc.Alert(
            "Portfolio transferred! Go to Backtest tab to analyze.",
            color="success",
            duration=5000,
        )

        return store_data, display, chart, feedback

    # =========================================================================
    # Theme Selector Callback (Clientside)
    # =========================================================================

    # Apply theme when selector changes
    app.clientside_callback(
        """
        function(theme) {
            // Default to bloomberg if no theme selected
            var activeTheme = theme || 'theme-bloomberg';

            // Remove all theme classes from body
            var themeClasses = ['theme-light', 'theme-dark', 'theme-bloomberg',
                               'theme-modern', 'theme-professional'];
            themeClasses.forEach(function(cls) {
                document.body.classList.remove(cls);
            });

            // Add the selected theme class
            document.body.classList.add(activeTheme);

            // Save to localStorage for persistence
            try {
                localStorage.setItem('etf-portfolio-theme', activeTheme);
            } catch(e) {}

            return activeTheme;
        }
        """,
        Output("theme-output", "children"),
        Input("theme-selector", "value"),
    )

    # =========================================================================
    # Data Download Callbacks
    # =========================================================================

    @app.callback(
        Output("download-etf-count", "children"),
        Input("download-etf-dropdown", "value"),
    )
    def update_download_etf_count(tickers):
        """Show selected ETF count."""
        if not tickers:
            return html.Span("No ETFs selected", className="text-muted")

        count = len(tickers)
        return html.Span(
            f"{count} ETF{'s' if count > 1 else ''} selected",
            className="text-success" if count > 0 else "text-muted"
        )

    @app.callback(
        [Output("daily-options", "style"),
         Output("intraday-options", "style")],
        Input("download-data-type", "value"),
    )
    def toggle_data_type_options(data_type):
        """Show/hide options based on data type."""
        if data_type == "daily":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("intraday-limit-warning", "children"),
        Input("download-interval", "value"),
    )
    def update_intraday_warning(interval):
        """Show intraday data limitations."""
        from services.data_client import INTRADAY_LIMITS
        max_days = INTRADAY_LIMITS.get(interval, 60)
        return f"Note: {interval} data available for last {max_days} days only (yfinance limit)"

    @app.callback(
        [Output("download-summary", "children"),
         Output("download-preview-table", "children"),
         Output("download-price-chart", "figure"),
         Output("download-data-store", "data"),
         Output("download-csv-btn", "disabled"),
         Output("fetch-loading", "children")],
        Input("fetch-data-btn", "n_clicks"),
        [State("download-etf-dropdown", "value"),
         State("download-data-type", "value"),
         State("download-date-range", "start_date"),
         State("download-date-range", "end_date"),
         State("download-interval", "value"),
         State("download-period", "value")],
        prevent_initial_call=True,
    )
    def fetch_download_data(n_clicks, tickers, data_type, start_date, end_date,
                           interval, period):
        """Fetch data and update preview."""
        import plotly.graph_objects as go
        from services.data_client import fetch_intraday_prices, get_aligned_prices
        from ui.charts import apply_bb_layout

        if not tickers:
            return (
                html.P("Please select at least one ETF", className="text-warning"),
                None,
                go.Figure(),
                None,
                True,
                ""
            )

        try:
            if data_type == "daily":
                # Fetch daily data
                start = date.fromisoformat(start_date[:10]) if isinstance(start_date, str) else start_date
                end = date.fromisoformat(end_date[:10]) if isinstance(end_date, str) else end_date

                prices_df = get_aligned_prices(tickers, start, end, use_cache=False)

                if prices_df is None or prices_df.empty:
                    return (
                        html.P("No data available for selected ETFs and date range",
                               className="text-warning"),
                        None,
                        go.Figure(),
                        None,
                        True,
                        ""
                    )

                # Reset index for display
                display_df = prices_df.reset_index()
                display_df.columns = ["Date"] + list(prices_df.columns)

                # Summary
                summary = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5(f"{len(tickers)}", className="text-primary mb-0"),
                            html.Small("ETFs"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(f"{len(prices_df):,}", className="text-primary mb-0"),
                            html.Small("Rows"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(str(prices_df.index.min()), className="text-primary mb-0"),
                            html.Small("Start Date"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(str(prices_df.index.max()), className="text-primary mb-0"),
                            html.Small("End Date"),
                        ], className="text-center"),
                    ]),
                ])

                # Preview table
                preview_data = display_df.head(100).copy()
                preview_data["Date"] = preview_data["Date"].astype(str)
                for col in preview_data.columns[1:]:
                    preview_data[col] = preview_data[col].round(2)

                table = dbc.Table.from_dataframe(
                    preview_data,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                )

                # Chart
                fig = go.Figure()
                for ticker in tickers[:5]:  # Limit to 5 for chart clarity
                    if ticker in prices_df.columns:
                        fig.add_trace(go.Scatter(
                            x=prices_df.index,
                            y=prices_df[ticker],
                            name=ticker,
                            mode="lines",
                        ))
                fig.update_layout(
                    title="Price History",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    hovermode="x unified",
                )
                apply_bb_layout(fig)

                # Store data for download
                store_data = {
                    "type": "daily",
                    "data": display_df.to_dict("records"),
                    "columns": list(display_df.columns),
                }

                return summary, table, fig, store_data, False, ""

            else:
                # Fetch intraday data
                intraday_df = fetch_intraday_prices(tickers, period, interval)

                if intraday_df is None or intraday_df.empty:
                    return (
                        html.P("No intraday data available. Try a different period or interval.",
                               className="text-warning"),
                        None,
                        go.Figure(),
                        None,
                        True,
                        ""
                    )

                # Summary
                summary = html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.H5(f"{len(tickers)}", className="text-primary mb-0"),
                            html.Small("ETFs"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(f"{len(intraday_df):,}", className="text-primary mb-0"),
                            html.Small("Rows"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(interval, className="text-primary mb-0"),
                            html.Small("Interval"),
                        ], className="text-center"),
                        dbc.Col([
                            html.H5(period, className="text-primary mb-0"),
                            html.Small("Period"),
                        ], className="text-center"),
                    ]),
                ])

                # Preview table
                preview_data = intraday_df.head(100).copy()
                preview_data["Datetime"] = preview_data["Datetime"].astype(str)
                for col in ["Open", "High", "Low", "Close"]:
                    if col in preview_data.columns:
                        preview_data[col] = preview_data[col].round(2)

                table = dbc.Table.from_dataframe(
                    preview_data,
                    striped=True,
                    bordered=True,
                    hover=True,
                    size="sm",
                )

                # Chart - pivot to wide format for plotting
                fig = go.Figure()
                for ticker in tickers[:5]:
                    ticker_data = intraday_df[intraday_df["Ticker"] == ticker]
                    if not ticker_data.empty:
                        fig.add_trace(go.Scatter(
                            x=ticker_data["Datetime"],
                            y=ticker_data["Close"],
                            name=ticker,
                            mode="lines",
                        ))
                fig.update_layout(
                    title=f"Intraday Prices ({interval})",
                    xaxis_title="Datetime",
                    yaxis_title="Price",
                    hovermode="x unified",
                )
                apply_bb_layout(fig)

                # Store data for download
                export_df = intraday_df.copy()
                export_df["Datetime"] = export_df["Datetime"].astype(str)
                store_data = {
                    "type": "intraday",
                    "data": export_df.to_dict("records"),
                    "columns": list(export_df.columns),
                }

                return summary, table, fig, store_data, False, ""

        except Exception as e:
            return (
                html.P(f"Error fetching data: {str(e)}", className="text-danger"),
                None,
                go.Figure(),
                None,
                True,
                ""
            )

    @app.callback(
        Output("download-intraday-csv", "data"),
        Input("download-csv-btn", "n_clicks"),
        State("download-data-store", "data"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks, store_data):
        """Download data as CSV."""
        if not store_data or not store_data.get("data"):
            return no_update

        df = pd.DataFrame(store_data["data"])
        data_type = store_data.get("type", "data")
        filename = f"etf_{data_type}_{date.today().isoformat()}.csv"

        return dcc.send_data_frame(df.to_csv, filename, index=False)

    # =========================================================================
    # Market Regime Callbacks
    # =========================================================================

    @app.callback(
        [
            Output("regime-gauge", "figure"),
            Output("regime-summary", "children"),
            Output("regime-indicators-display", "children"),
            Output("fear-greed-gauge", "figure"),
            Output("regime-state-store", "data"),
            Output("regime-badge", "children"),
            Output("regime-badge", "color"),
            # FRED Economic Indicators outputs
            Output("yield-curve-display", "children"),
            Output("yield-curve-signal", "children"),
            Output("credit-spread-display", "children"),
            Output("credit-spread-signal", "children"),
            Output("unemployment-display", "children"),
            Output("unemployment-signal", "children"),
            Output("claims-display", "children"),
            Output("claims-signal", "children"),
            Output("consumer-sentiment-display", "children"),
            Output("consumer-sentiment-signal", "children"),
            Output("fed-funds-display", "children"),
            Output("fed-funds-signal", "children"),
            Output("economic-signal-badge", "children"),
            Output("economic-signal-badge", "color"),
            Output("economic-summary", "children"),
        ],
        [
            Input("refresh-regime-btn", "n_clicks"),
            Input("main-tabs", "active_tab"),
        ],
        prevent_initial_call=True,
    )
    def update_regime_indicators(n_clicks, active_tab):
        """Update market regime indicators when tab is active or refresh clicked."""
        import plotly.graph_objects as go

        # Only update when on the regime tab
        if active_tab != "tab-regime":
            return [no_update] * 22

        # Detect market regime (now includes economic data)
        regime_state = detect_regime(include_economic=True)

        if regime_state is None:
            empty_fig = go.Figure()
            return (
                empty_fig,
                html.P("Unable to fetch market data", className="text-danger"),
                html.P("Error connecting to data source", className="text-muted"),
                empty_fig,
                None,
                "Error",
                "danger",
                # Economic indicators - default values
                "N/A", "Data unavailable",
                "N/A", "Data unavailable",
                "N/A", "Data unavailable",
                "N/A", "Data unavailable",
                "N/A", "Data unavailable",
                "N/A", "Data unavailable",
                "Unknown", "secondary",
                "Unable to fetch FRED economic data",
            )

        # Get regime summary
        summary = get_regime_summary(regime_state)

        # Create regime gauge - normalize score to -1 to +1
        # With economic data, max score is ~4.5, so normalize differently
        max_score = 4.5 if regime_state.indicators.economic else 3.0
        regime_value = regime_state.score / max_score
        regime_value = max(-1, min(1, regime_value))  # Clamp to [-1, 1]

        regime_gauge = create_regime_gauge(
            regime_value=regime_value,
            confidence=regime_state.confidence,
            regime_label=regime_state.regime.value,
        )

        # Create regime summary
        regime_summary = html.Div([
            html.P([
                html.Strong("Score: "),
                html.Span(
                    summary["score"],
                    className="text-success" if regime_state.score > 0 else "text-danger" if regime_state.score < 0 else "text-warning",
                ),
            ], className="mb-1"),
        ])

        # Create indicators display
        indicators = regime_state.indicators
        indicators_display = html.Div([
            dbc.Table([
                html.Tbody([
                    html.Tr([
                        html.Td("VIX Level"),
                        html.Td(f"{indicators.vix_level:.1f}", className="text-end"),
                    ]),
                    html.Tr([
                        html.Td("VIX Percentile"),
                        html.Td(f"{indicators.vix_percentile:.0f}th", className="text-end"),
                    ]),
                    html.Tr([
                        html.Td("Volatility Regime"),
                        html.Td(indicators.volatility_regime.value, className="text-end"),
                    ]),
                    html.Tr([
                        html.Td("SPY Price"),
                        html.Td(f"${indicators.spy_price:,.2f}", className="text-end"),
                    ]),
                    html.Tr([
                        html.Td("SPY vs 200 SMA"),
                        html.Td(
                            f"{indicators.spy_vs_200sma:+.1%}",
                            className="text-end " + ("text-success" if indicators.spy_vs_200sma > 0 else "text-danger"),
                        ),
                    ]),
                    html.Tr([
                        html.Td("SPY vs 50 SMA"),
                        html.Td(
                            f"{indicators.spy_vs_50sma:+.1%}",
                            className="text-end " + ("text-success" if indicators.spy_vs_50sma > 0 else "text-danger"),
                        ),
                    ]),
                ]),
            ], size="sm", bordered=True, className="mb-3"),

            # Signals (limit to technical signals only in this view)
            html.H6("Technical Signals", className="mt-2"),
            html.Ul([
                html.Li(signal, className="small")
                for signal in regime_state.signals[:4]
                if not any(kw in signal.lower() for kw in ["yield", "credit", "labor", "consumer", "fed"])
            ], className="small"),
        ])

        # Fetch Fear & Greed
        fg_data = fetch_fear_greed_index()
        if fg_data and fg_data.raw_value is not None:
            fg_gauge = create_fear_greed_gauge(
                score=fg_data.raw_value,
                label=fg_data.label,
            )
        else:
            fg_gauge = create_fear_greed_gauge(score=50, label="Unavailable")

        # Determine badge color
        if regime_state.regime.value == "Bull / Risk-On":
            badge_color = "success"
        elif regime_state.regime.value == "Bear / Risk-Off":
            badge_color = "danger"
        else:
            badge_color = "warning"

        # === FRED Economic Indicators ===
        econ = indicators.economic
        if econ:
            # Yield Curve
            yield_curve_val = f"{econ.yield_curve_spread:.2f}%" if econ.yield_curve_spread is not None else "N/A"
            yield_curve_sig = econ.yield_curve_signal or "N/A"
            yield_curve_class = _get_signal_class(econ.yield_curve_signal)

            # Credit Spreads
            if econ.credit_spread_ig is not None or econ.credit_spread_hy is not None:
                ig_str = f"IG: {econ.credit_spread_ig:.0f}" if econ.credit_spread_ig else ""
                hy_str = f"HY: {econ.credit_spread_hy:.0f}" if econ.credit_spread_hy else ""
                credit_val = f"{ig_str} {hy_str}".strip()
            else:
                credit_val = "N/A"
            credit_sig = econ.credit_signal or "N/A"
            credit_class = _get_signal_class(econ.credit_signal)

            # Unemployment
            unemp_val = f"{econ.unemployment_rate:.1f}%" if econ.unemployment_rate is not None else "N/A"
            unemp_sig = econ.labor_signal or "N/A"
            unemp_class = _get_signal_class(econ.labor_signal)

            # Initial Claims
            claims_val = f"{econ.initial_claims:,}" if econ.initial_claims is not None else "N/A"
            claims_sig = econ.labor_signal or "N/A"

            # Consumer Sentiment
            sentiment_val = f"{econ.consumer_sentiment:.0f}" if econ.consumer_sentiment is not None else "N/A"
            sentiment_sig = econ.sentiment_signal or "N/A"
            sentiment_class = _get_signal_class(econ.sentiment_signal)

            # Fed Funds
            fed_val = f"{econ.fed_funds_rate:.2f}%" if econ.fed_funds_rate is not None else "N/A"
            fed_sig = econ.fed_stance or "N/A"

            # Economic signal badge
            econ_score = econ.economic_score
            if econ_score > 0.5:
                econ_badge = "Bullish"
                econ_badge_color = "success"
            elif econ_score < -0.5:
                econ_badge = "Bearish"
                econ_badge_color = "danger"
            else:
                econ_badge = "Neutral"
                econ_badge_color = "warning"

            # Economic summary
            econ_signals = econ.economic_signals[:3] if econ.economic_signals else []
            econ_summary = html.Div([
                html.Span(f"Economic Score: {econ_score:+.2f} | ", className="fw-bold"),
                html.Span(" | ".join(econ_signals) if econ_signals else "No significant signals"),
            ])
        else:
            # Fallback values when no economic data
            yield_curve_val = "N/A"
            yield_curve_sig = "No FRED API key"
            credit_val = "N/A"
            credit_sig = "Set FRED_API_KEY"
            unemp_val = "N/A"
            unemp_sig = "in environment"
            claims_val = "N/A"
            claims_sig = ""
            sentiment_val = "N/A"
            sentiment_sig = ""
            fed_val = "N/A"
            fed_sig = ""
            econ_badge = "No Data"
            econ_badge_color = "secondary"
            econ_summary = html.Span(
                "Set FRED_API_KEY environment variable to enable economic indicators",
                className="text-muted",
            )

        # Store regime state (include economic data)
        store_data = {
            "regime": regime_state.regime.value,
            "confidence": regime_state.confidence,
            "score": regime_state.score,
            "indicators": {
                "vix_level": indicators.vix_level,
                "vix_percentile": indicators.vix_percentile,
                "spy_price": indicators.spy_price,
                "spy_vs_200sma": indicators.spy_vs_200sma,
                "spy_vs_50sma": indicators.spy_vs_50sma,
            },
            "economic": {
                "yield_curve_spread": econ.yield_curve_spread if econ else None,
                "credit_spread_ig": econ.credit_spread_ig if econ else None,
                "credit_spread_hy": econ.credit_spread_hy if econ else None,
                "unemployment_rate": econ.unemployment_rate if econ else None,
                "consumer_sentiment": econ.consumer_sentiment if econ else None,
                "fed_funds_rate": econ.fed_funds_rate if econ else None,
                "economic_score": econ.economic_score if econ else 0,
            } if econ else None,
        }

        return (
            regime_gauge,
            regime_summary,
            indicators_display,
            fg_gauge,
            store_data,
            regime_state.regime.value,
            badge_color,
            # Economic indicators
            yield_curve_val,
            yield_curve_sig,
            credit_val,
            credit_sig,
            unemp_val,
            unemp_sig,
            claims_val,
            claims_sig,
            sentiment_val,
            sentiment_sig,
            fed_val,
            fed_sig,
            econ_badge,
            econ_badge_color,
            econ_summary,
        )

    def _get_signal_class(signal: str) -> str:
        """Get CSS class for signal badge."""
        if signal in ("bullish", "Bullish"):
            return "text-success"
        elif signal in ("bearish", "Bearish"):
            return "text-danger"
        elif signal in ("cautious", "Cautious"):
            return "text-warning"
        return "text-muted"

    @app.callback(
        Output("fred-charts-collapse", "is_open"),
        Output("toggle-fred-charts-btn", "children"),
        Input("toggle-fred-charts-btn", "n_clicks"),
        State("fred-charts-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_fred_charts(n_clicks, is_open):
        """Toggle the FRED historical charts section."""
        new_state = not is_open
        if new_state:
            btn_content = [
                html.I(className="bi bi-graph-up me-2"),
                "Hide Historical Charts",
            ]
        else:
            btn_content = [
                html.I(className="bi bi-graph-up me-2"),
                "Show Historical Charts",
            ]
        return new_state, btn_content

    @app.callback(
        Output("fred-combined-chart", "figure"),
        Input("fred-charts-collapse", "is_open"),
        State("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def update_fred_charts(is_open, main_tab):
        """Update FRED historical charts when expanded."""
        import plotly.graph_objects as go

        if not is_open or main_tab != "tab-regime":
            return no_update

        try:
            from services.fred_client import get_fred_historical_data

            # Fetch historical data (2 years)
            historical_data = get_fred_historical_data(lookback_years=2)

            if not historical_data:
                # Return empty chart with message
                fig = go.Figure()
                fig.add_annotation(
                    text="Unable to fetch FRED historical data.<br>Please set FRED_API_KEY environment variable.",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#1a1a2e",
                )
                return fig

            # Create combined chart
            return create_fred_combined_chart(historical_data)

        except Exception as e:
            print(f"Error fetching FRED historical data: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading charts: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            fig.update_layout(
                height=400,
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#1a1a2e",
            )
            return fig

    @app.callback(
        Output("analyst-ratings-content", "children"),
        Input("market-views-tabs", "active_tab"),
        State("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def update_analyst_ratings(views_tab, main_tab):
        """Update analyst ratings when tab is selected."""
        if main_tab != "tab-regime" or views_tab != "analyst-tab":
            return no_update

        # Fetch analyst ratings
        ratings, aggregate = fetch_market_analyst_consensus()

        if not ratings:
            return html.P("Unable to fetch analyst ratings", className="text-muted")

        # Create ratings table
        rows = []
        for ticker, rating in ratings.items():
            upside_class = "text-success" if rating.upside_pct and rating.upside_pct > 0 else "text-danger"
            rows.append(html.Tr([
                html.Td(html.Strong(ticker)),
                html.Td(f"{rating.buy_pct:.0f}%", className="text-success"),
                html.Td(f"{rating.hold_pct:.0f}%", className="text-warning"),
                html.Td(f"{rating.sell_pct:.0f}%", className="text-danger"),
                html.Td(
                    f"{rating.upside_pct:+.1f}%" if rating.upside_pct else "N/A",
                    className=upside_class,
                ),
            ]))

        return html.Div([
            # Aggregate summary
            dbc.Alert([
                html.Strong("Market Consensus: "),
                html.Span(
                    aggregate.label,
                    className="text-success" if aggregate.label == "Bullish" else "text-danger" if aggregate.label == "Bearish" else "text-warning",
                ),
                html.Span(f" (Score: {aggregate.score:+.2f})", className="text-muted"),
            ], color="light", className="mb-3"),

            # Ratings table
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Ticker"),
                        html.Th("Buy"),
                        html.Th("Hold"),
                        html.Th("Sell"),
                        html.Th("Upside"),
                    ]),
                ]),
                html.Tbody(rows),
            ], size="sm", striped=True, hover=True),
        ])

    @app.callback(
        Output("sentiment-chart", "figure"),
        Input("market-views-tabs", "active_tab"),
        State("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def update_sentiment_chart(views_tab, main_tab):
        """Update sentiment chart when tab is selected."""
        import plotly.graph_objects as go

        if main_tab != "tab-regime" or views_tab != "sentiment-tab":
            return no_update

        # Fetch aggregate sentiment
        sentiments = get_aggregate_sentiment()

        if not sentiments:
            return go.Figure()

        # Format for chart
        sentiment_data = []
        for key, sent in sentiments.items():
            sentiment_data.append({
                "source": sent.source.value,
                "score": sent.score,
                "label": sent.label,
            })

        return create_sentiment_bars(sentiment_data)

    @app.callback(
        Output("pdf-analysis-result", "children"),
        Output("pdf-research-store", "data"),
        Input("pdf-upload", "contents"),
        State("pdf-upload", "filename"),
        prevent_initial_call=True,
    )
    def process_uploaded_pdf(contents, filename):
        """Process uploaded research PDF."""
        if contents is None:
            return no_update, no_update

        if not is_pdf_support_available():
            return (
                dbc.Alert(
                    "PDF processing is not available. Install PyMuPDF: pip install PyMuPDF",
                    color="warning",
                ),
                None,
            )

        # Process PDF
        summary = process_research_pdf(contents, filename or "research.pdf")

        if summary is None:
            return (
                dbc.Alert("Error processing PDF", color="danger"),
                None,
            )

        # Create result display
        sentiment_class = {
            "bullish": "success",
            "bearish": "danger",
            "neutral": "warning",
        }.get(summary.sentiment, "secondary")

        result = html.Div([
            dbc.Alert([
                html.Strong("File: "),
                html.Span(summary.filename),
            ], color="light", className="mb-2"),

            dbc.Row([
                dbc.Col([
                    html.P([
                        html.Strong("Sentiment: "),
                        dbc.Badge(
                            summary.sentiment.title(),
                            color=sentiment_class,
                        ),
                    ]),
                ], md=4),
                dbc.Col([
                    html.P([
                        html.Strong("Words: "),
                        html.Span(f"{summary.word_count:,}"),
                    ]),
                ], md=4),
                dbc.Col([
                    html.P([
                        html.Strong("Keywords: "),
                        html.Span(", ".join(summary.keywords[:5]) or "None"),
                    ]),
                ], md=4),
            ], className="mb-2"),

            html.Div([
                html.Strong("Preview:"),
                html.P(summary.preview, className="small text-muted border p-2 rounded"),
            ]),
        ])

        # Store for later use
        store_data = {
            "filename": summary.filename,
            "sentiment": summary.sentiment,
            "score": get_sentiment_score(summary.sentiment),
        }

        return result, store_data

    @app.callback(
        [
            Output("regime-recommendations", "children"),
            Output("allocation-comparison-chart", "figure"),
        ],
        [
            Input("regime-state-store", "data"),
            Input("portfolio-store", "data"),
        ],
    )
    def update_regime_recommendations(regime_data, portfolio_data):
        """Update regime recommendations based on current regime and portfolio."""
        import plotly.graph_objects as go
        from domain.schemas import MarketRegime, RegimeState, RegimeIndicators, VolatilityRegime

        empty_fig = go.Figure()

        if not regime_data:
            return (
                html.P("Click 'Refresh Data' to detect market regime", className="text-muted"),
                empty_fig,
            )

        # Reconstruct regime state
        try:
            indicators = RegimeIndicators(
                vix_level=regime_data["indicators"]["vix_level"],
                vix_percentile=regime_data["indicators"]["vix_percentile"],
                spy_price=regime_data["indicators"]["spy_price"],
                spy_200sma=regime_data["indicators"]["spy_price"],  # Approximate
                spy_50sma=regime_data["indicators"]["spy_price"],   # Approximate
                spy_vs_200sma=regime_data["indicators"]["spy_vs_200sma"],
                spy_vs_50sma=regime_data["indicators"]["spy_vs_50sma"],
                volatility_regime=VolatilityRegime.NORMAL,
            )

            regime_state = RegimeState(
                regime=MarketRegime(regime_data["regime"]),
                confidence=regime_data["confidence"],
                score=regime_data["score"],
                indicators=indicators,
                signals=[],
            )
        except Exception:
            return (
                html.P("Error processing regime data", className="text-danger"),
                empty_fig,
            )

        # Get or create portfolio
        if portfolio_data and portfolio_data.get("holdings"):
            holdings = [
                PortfolioHolding(
                    ticker=h["ticker"],
                    weight=h["weight"],
                )
                for h in portfolio_data["holdings"]
            ]
            portfolio = Portfolio(holdings=holdings)
        else:
            portfolio = create_default_portfolio()

        # Generate recommendations
        recommendation = generate_regime_recommendation(portfolio, regime_state)
        summary = get_recommendation_summary(recommendation)

        # Create recommendations display
        changes_rows = []
        for change in summary["allocation_changes"]:
            direction_icon = "↑" if change["direction"] == "up" else "↓" if change["direction"] == "down" else "→"
            direction_class = "text-success" if change["direction"] == "up" else "text-danger" if change["direction"] == "down" else "text-muted"
            changes_rows.append(html.Tr([
                html.Td(change["asset_class"]),
                html.Td(change["current"]),
                html.Td(change["recommended"]),
                html.Td(
                    f"{direction_icon} {change['change']}",
                    className=direction_class,
                ),
            ]))

        recommendations_content = html.Div([
            # Allocation changes table
            html.H6("Allocation Adjustments"),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Asset Class"),
                        html.Th("Current"),
                        html.Th("Target"),
                        html.Th("Change"),
                    ]),
                ]),
                html.Tbody(changes_rows),
            ], size="sm", bordered=True, className="mb-3"),

            # ETF suggestions
            html.H6("Suggested ETFs"),
            html.Ul([
                html.Li([
                    html.Strong(etf["ticker"]),
                    html.Span(f": {etf['reason']}", className="small text-muted"),
                ])
                for etf in summary["etf_suggestions"][:4]
            ], className="small mb-3"),

            # Rationale
            html.H6("Rationale"),
            html.Ul([
                html.Li(r, className="small text-muted")
                for r in summary["rationale"][:3]
            ]),
        ])

        # Create allocation comparison chart
        comparison_chart = create_allocation_comparison_chart(
            current=recommendation.current_allocation,
            recommended=recommendation.recommended_allocation,
        )

        return recommendations_content, comparison_chart

    @app.callback(
        Output("portfolio-store", "data", allow_duplicate=True),
        Input("apply-regime-tilts-btn", "n_clicks"),
        [
            State("regime-state-store", "data"),
            State("portfolio-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def apply_tilts_to_portfolio(n_clicks, regime_data, portfolio_data):
        """Apply regime tilts to the current portfolio."""
        from domain.schemas import MarketRegime, RegimeState, RegimeIndicators, VolatilityRegime

        if not n_clicks or not regime_data:
            return no_update

        # Reconstruct regime state
        try:
            indicators = RegimeIndicators(
                vix_level=regime_data["indicators"]["vix_level"],
                vix_percentile=regime_data["indicators"]["vix_percentile"],
                spy_price=regime_data["indicators"]["spy_price"],
                spy_200sma=regime_data["indicators"]["spy_price"],
                spy_50sma=regime_data["indicators"]["spy_price"],
                spy_vs_200sma=regime_data["indicators"]["spy_vs_200sma"],
                spy_vs_50sma=regime_data["indicators"]["spy_vs_50sma"],
                volatility_regime=VolatilityRegime.NORMAL,
            )

            regime_state = RegimeState(
                regime=MarketRegime(regime_data["regime"]),
                confidence=regime_data["confidence"],
                score=regime_data["score"],
                indicators=indicators,
                signals=[],
            )
        except Exception:
            return no_update

        # Get or create portfolio
        if portfolio_data and portfolio_data.get("holdings"):
            holdings = [
                PortfolioHolding(
                    ticker=h["ticker"],
                    weight=h["weight"],
                )
                for h in portfolio_data["holdings"]
            ]
            portfolio = Portfolio(holdings=holdings)
        else:
            portfolio = create_default_portfolio()

        # Apply tilts
        tilted_portfolio = apply_regime_tilts(portfolio, regime_state)

        # Return updated portfolio data
        return {
            "holdings": [
                {"ticker": h.ticker, "weight": h.weight}
                for h in tilted_portfolio.holdings
            ],
            "notes": tilted_portfolio.notes,
        }
