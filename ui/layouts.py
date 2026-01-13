"""Dash layout components for the ETF Portfolio Tool."""

from datetime import date, timedelta

import dash_bootstrap_components as dbc
from dash import dcc, html

from data.etf_universe import ETF_UNIVERSE
from domain.schemas import RebalanceFrequency, RiskProfile


def create_app_layout() -> dbc.Container:
    """Create the main application layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("ETF Portfolio Tool", className="text-primary mb-0"),
                html.P("Portfolio Construction, Backtesting & Analysis",
                       className="text-muted"),
            ]),
        ], className="my-4"),

        # Main tabs
        dbc.Tabs([
            dbc.Tab(create_portfolio_tab(), label="Portfolio Builder", tab_id="tab-portfolio"),
            dbc.Tab(create_backtest_tab(), label="Backtest", tab_id="tab-backtest"),
            dbc.Tab(create_analytics_tab(), label="Analytics", tab_id="tab-analytics"),
            dbc.Tab(create_advanced_analytics_tab(), label="Advanced", tab_id="tab-advanced"),
            dbc.Tab(create_diversification_tab(), label="Diversification", tab_id="tab-diversification"),
            dbc.Tab(create_comparison_tab(), label="Compare Strategies", tab_id="tab-compare"),
            dbc.Tab(create_hedges_tab(), label="Hedges", tab_id="tab-hedges"),
            dbc.Tab(create_export_tab(), label="Export", tab_id="tab-export"),
        ], id="main-tabs", active_tab="tab-portfolio"),

        # Stores for app state
        dcc.Store(id="portfolio-store", storage_type="memory"),
        dcc.Store(id="backtest-store", storage_type="memory"),
        dcc.Store(id="hedge-store", storage_type="memory"),
        dcc.Store(id="comparison-store", storage_type="memory"),
        dcc.Store(id="advanced-analytics-store", storage_type="memory"),

    ], fluid=True)


def create_portfolio_tab() -> dbc.Container:
    """Create the Portfolio Builder tab."""
    return dbc.Container([
        dbc.Row([
            # Left panel - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Settings"),
                    dbc.CardBody([
                        # Allocation Strategy
                        html.Label("Allocation Strategy", className="fw-bold"),
                        dcc.Dropdown(
                            id="strategy-dropdown",
                            options=[
                                {"label": "Strategic (Baseline + Tilts)", "value": "Strategic"},
                                {"label": "60/40 (Classic Balanced)", "value": "60/40"},
                                {"label": "Growth (Equity Heavy)", "value": "Growth"},
                                {"label": "Conservative (Bond Heavy)", "value": "Conservative"},
                                {"label": "Aggressive (100% Equity)", "value": "Aggressive"},
                                {"label": "Income (Dividend Focus)", "value": "Income"},
                                {"label": "All Weather (Balanced)", "value": "All Weather"},
                                {"label": "Risk Parity", "value": "Risk Parity"},
                                {"label": "Equal Weight", "value": "Equal Weight"},
                            ],
                            value="Strategic",
                            clearable=False,
                            className="mb-3",
                        ),

                        # Risk Parity Options (shown conditionally)
                        html.Div(
                            id="risk-parity-options",
                            children=[
                                html.Label("Risk Parity Universe", className="fw-bold"),
                                dcc.Dropdown(
                                    id="rp-universe-dropdown",
                                    options=[
                                        {"label": "Balanced (Stocks, Bonds, Gold, REITs)", "value": "balanced"},
                                        {"label": "Equity Focused", "value": "equity_focused"},
                                        {"label": "Defensive", "value": "defensive"},
                                        {"label": "Global", "value": "global"},
                                    ],
                                    value="balanced",
                                    clearable=False,
                                    className="mb-3",
                                ),
                                html.Label("Risk Parity Method", className="fw-bold"),
                                dcc.Dropdown(
                                    id="rp-method-dropdown",
                                    options=[
                                        {"label": "Inverse Volatility", "value": "inverse_vol"},
                                        {"label": "Equal Risk Contribution", "value": "equal_risk_contribution"},
                                    ],
                                    value="inverse_vol",
                                    clearable=False,
                                    className="mb-3",
                                ),
                            ],
                            style={"display": "none"},
                        ),

                        # Risk Profile (for Strategic strategy)
                        html.Div(
                            id="strategic-options",
                            children=[
                                html.Label("Risk Profile", className="fw-bold"),
                                dcc.Dropdown(
                                    id="risk-profile-dropdown",
                                    options=[
                                        {"label": "Conservative", "value": "Conservative"},
                                        {"label": "Moderate", "value": "Moderate"},
                                        {"label": "Aggressive", "value": "Aggressive"},
                                    ],
                                    value="Moderate",
                                    clearable=False,
                                    className="mb-3",
                                ),
                            ],
                        ),

                        # Constraints
                        html.Label("Max Weight per ETF", className="fw-bold"),
                        dcc.Slider(
                            id="max-weight-slider",
                            min=0.05,
                            max=0.50,
                            step=0.05,
                            value=0.25,
                            marks={i/100: f"{i}%" for i in range(5, 55, 10)},
                            className="mb-4",
                        ),

                        # Region tilts (Strategic only)
                        html.Div(
                            id="tilts-options",
                            children=[
                                html.Label("Region Tilts", className="fw-bold"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Small("US"),
                                        dcc.Slider(
                                            id="us-tilt-slider",
                                            min=-0.10,
                                            max=0.10,
                                            step=0.02,
                                            value=0,
                                            marks={-0.10: "-10%", 0: "0%", 0.10: "+10%"},
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        html.Small("Emerging Markets"),
                                        dcc.Slider(
                                            id="em-tilt-slider",
                                            min=-0.10,
                                            max=0.10,
                                            step=0.02,
                                            value=0,
                                            marks={-0.10: "-10%", 0: "0%", 0.10: "+10%"},
                                        ),
                                    ], width=6),
                                ], className="mb-3"),
                            ],
                        ),

                        # Excluded tickers
                        html.Label("Exclude Tickers", className="fw-bold"),
                        dcc.Dropdown(
                            id="excluded-tickers-dropdown",
                            options=[{"label": etf.ticker, "value": etf.ticker}
                                    for etf in ETF_UNIVERSE],
                            multi=True,
                            placeholder="Select tickers to exclude...",
                            className="mb-4",
                        ),

                        # Generate button
                        dbc.Button(
                            "Generate Portfolio",
                            id="generate-portfolio-btn",
                            color="primary",
                            size="lg",
                            className="w-100",
                        ),
                    ]),
                ], className="mb-3"),
            ], md=4),

            # Right panel - Portfolio display
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Generated Portfolio"),
                    dbc.CardBody([
                        html.Div(id="portfolio-display"),
                        dcc.Graph(id="holdings-chart"),
                    ]),
                ], className="mb-3"),

                # Save/Load section
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Save & Load Portfolios"),
                        dbc.Button(
                            "Expand",
                            id="toggle-save-load-btn",
                            color="link",
                            size="sm",
                            className="float-end p-0",
                        ),
                    ]),
                    dbc.Collapse([
                        dbc.CardBody([
                            dbc.Row([
                                # Save section
                                dbc.Col([
                                    html.Label("Save Current Portfolio", className="fw-bold"),
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="save-portfolio-name",
                                            placeholder="Enter portfolio name...",
                                            type="text",
                                        ),
                                        dbc.Button(
                                            "Save",
                                            id="save-portfolio-btn",
                                            color="success",
                                        ),
                                    ], className="mb-2"),
                                    html.Div(id="save-portfolio-feedback"),
                                ], md=6),

                                # Load section
                                dbc.Col([
                                    html.Label("Load Saved Portfolio", className="fw-bold"),
                                    dbc.InputGroup([
                                        dcc.Dropdown(
                                            id="load-portfolio-dropdown",
                                            placeholder="Select portfolio...",
                                            className="flex-grow-1",
                                            style={"minWidth": "200px"},
                                        ),
                                        dbc.Button(
                                            "Load",
                                            id="load-portfolio-btn",
                                            color="primary",
                                        ),
                                        dbc.Button(
                                            "Delete",
                                            id="delete-portfolio-btn",
                                            color="danger",
                                            outline=True,
                                        ),
                                    ], className="mb-2"),
                                    html.Div(id="load-portfolio-feedback"),
                                ], md=6),
                            ]),

                            html.Hr(),

                            # Saved portfolios list
                            html.Label("Saved Portfolios", className="fw-bold"),
                            html.Div(id="saved-portfolios-list"),
                        ]),
                    ], id="save-load-collapse", is_open=False),
                ]),
            ], md=8),
        ]),
    ], fluid=True, className="py-3")


def create_backtest_tab() -> dbc.Container:
    """Create the Backtest tab."""
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 3)

    return dbc.Container([
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Backtest Settings"),
                    dbc.CardBody([
                        # Date range
                        html.Label("Date Range", className="fw-bold"),
                        dcc.DatePickerRange(
                            id="backtest-date-range",
                            start_date=default_start,
                            end_date=default_end,
                            max_date_allowed=default_end,
                            className="mb-3",
                        ),

                        # Rebalancing
                        html.Label("Rebalancing Frequency", className="fw-bold"),
                        dcc.Dropdown(
                            id="rebalance-dropdown",
                            options=[
                                {"label": "None (Buy & Hold)", "value": "None"},
                                {"label": "Monthly", "value": "Monthly"},
                                {"label": "Quarterly", "value": "Quarterly"},
                                {"label": "Annual", "value": "Annual"},
                            ],
                            value="Quarterly",
                            clearable=False,
                            className="mb-3",
                        ),

                        # Benchmark
                        html.Label("Benchmark", className="fw-bold"),
                        dcc.Dropdown(
                            id="benchmark-dropdown",
                            options=[
                                {"label": "SPY (S&P 500)", "value": "SPY"},
                                {"label": "VTI (Total Market)", "value": "VTI"},
                                {"label": "AGG (Aggregate Bond)", "value": "AGG"},
                                {"label": "EFA (International)", "value": "EFA"},
                            ],
                            value="SPY",
                            clearable=False,
                            className="mb-3",
                        ),

                        # Costs
                        html.Label("Include Transaction Costs", className="fw-bold"),
                        dbc.Switch(
                            id="costs-switch",
                            value=True,
                            className="mb-3",
                        ),

                        # Run button
                        dbc.Button(
                            "Run Backtest",
                            id="run-backtest-btn",
                            color="success",
                            size="lg",
                            className="w-100",
                        ),

                        # Loading indicator
                        dbc.Spinner(
                            html.Div(id="backtest-loading"),
                            color="primary",
                            type="border",
                            size="sm",
                        ),
                    ]),
                ]),
            ], md=3),

            # Results
            dbc.Col([
                # Metrics cards
                html.Div(id="metrics-cards", className="mb-3"),

                # Charts
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="equity-curve-chart"),
                    ]),
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id="drawdown-chart"),
                    ]),
                ]),
            ], md=9),
        ]),
    ], fluid=True, className="py-3")


def create_analytics_tab() -> dbc.Container:
    """Create the Analytics tab."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Portfolio Metrics"),
                    dbc.CardBody([
                        html.Div(id="detailed-metrics"),
                    ]),
                ]),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Benchmark Comparison"),
                    dbc.CardBody([
                        html.Div(id="benchmark-comparison"),
                    ]),
                ]),
            ], md=8),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Rolling Metrics"),
                    dbc.CardBody([
                        html.P("Run a backtest to see rolling metrics",
                               id="rolling-placeholder",
                               className="text-muted"),
                    ]),
                ]),
            ]),
        ]),
    ], fluid=True, className="py-3")


def create_advanced_analytics_tab() -> dbc.Container:
    """Create the Advanced Analytics tab with Monte Carlo, Risk Metrics, etc."""
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)

    return dbc.Container([
        # Controls row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Settings"),
                    dbc.CardBody([
                        html.Label("Historical Period", className="fw-bold"),
                        dcc.DatePickerRange(
                            id="advanced-date-range",
                            start_date=default_start,
                            end_date=default_end,
                            max_date_allowed=default_end,
                            className="mb-3",
                        ),

                        html.Label("Monte Carlo Projection Years", className="fw-bold"),
                        dcc.Slider(
                            id="mc-years-slider",
                            min=5,
                            max=30,
                            step=5,
                            value=10,
                            marks={i: f"{i}Y" for i in range(5, 35, 5)},
                            className="mb-3",
                        ),

                        html.Label("Initial Investment ($)", className="fw-bold"),
                        dcc.Input(
                            id="mc-initial-input",
                            type="number",
                            value=10000,
                            min=1000,
                            step=1000,
                            className="form-control mb-3",
                        ),

                        dbc.Button(
                            "Run Advanced Analytics",
                            id="run-advanced-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mb-2",
                        ),

                        dbc.Spinner(
                            html.Div(id="advanced-loading"),
                            color="primary",
                            type="border",
                            size="sm",
                        ),
                    ]),
                ]),
            ], md=3),

            # Results
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id="monte-carlo-chart"),
                                html.Div(id="monte-carlo-stats", className="mt-3"),
                            ]),
                        ]),
                    ], label="Monte Carlo", tab_id="mc-tab"),

                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id="rolling-returns-chart"),
                            ]),
                        ]),
                    ], label="Rolling Returns", tab_id="rolling-tab"),

                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(id="risk-metrics-display"),
                            ]),
                        ]),
                    ], label="Risk Metrics", tab_id="risk-tab"),

                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id="calendar-heatmap"),
                            ]),
                        ]),
                    ], label="Calendar Returns", tab_id="calendar-tab"),

                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Dividend Analysis"),
                                    dbc.CardBody([
                                        html.Div(id="dividend-summary"),
                                    ]),
                                ]),
                            ], md=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        dcc.Graph(id="dividend-chart"),
                                    ]),
                                ]),
                            ], md=8),
                        ]),
                    ], label="Income", tab_id="income-tab"),

                    dbc.Tab([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id="contribution-chart"),
                            ]),
                        ]),
                    ], label="Contribution", tab_id="contribution-tab"),

                ], id="advanced-tabs", active_tab="mc-tab"),
            ], md=9),
        ]),
    ], fluid=True, className="py-3")


def create_comparison_tab() -> dbc.Container:
    """Create the Strategy Comparison tab."""
    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 5)

    return dbc.Container([
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Comparison Settings"),
                    dbc.CardBody([
                        html.Label("Date Range", className="fw-bold"),
                        dcc.DatePickerRange(
                            id="compare-date-range",
                            start_date=default_start,
                            end_date=default_end,
                            max_date_allowed=default_end,
                            className="mb-3",
                        ),

                        html.Label("Rebalancing Frequency", className="fw-bold"),
                        dcc.Dropdown(
                            id="compare-rebalance-dropdown",
                            options=[
                                {"label": "Monthly", "value": "Monthly"},
                                {"label": "Quarterly", "value": "Quarterly"},
                                {"label": "Annual", "value": "Annual"},
                            ],
                            value="Quarterly",
                            clearable=False,
                            className="mb-3",
                        ),

                        html.Label("Strategies to Compare", className="fw-bold"),
                        dcc.Checklist(
                            id="compare-strategies-checklist",
                            options=[
                                {"label": " Strategic", "value": "Strategic"},
                                {"label": " 60/40", "value": "60/40"},
                                {"label": " Growth", "value": "Growth"},
                                {"label": " Conservative", "value": "Conservative"},
                                {"label": " Aggressive", "value": "Aggressive"},
                                {"label": " Income", "value": "Income"},
                                {"label": " All Weather", "value": "All Weather"},
                                {"label": " Risk Parity", "value": "Risk Parity"},
                                {"label": " Equal Weight", "value": "Equal Weight"},
                            ],
                            value=["60/40", "Growth", "Conservative", "All Weather"],
                            className="mb-3",
                            inputClassName="me-1",
                            labelClassName="d-block mb-1",
                        ),

                        dbc.Button(
                            "Run Comparison",
                            id="run-comparison-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mb-2",
                        ),

                        dbc.Spinner(
                            html.Div(id="comparison-loading"),
                            color="primary",
                            type="border",
                            size="sm",
                        ),
                    ]),
                ]),
            ], md=3),

            # Results
            dbc.Col([
                # Metrics comparison table
                dbc.Card([
                    dbc.CardHeader("Performance Metrics Comparison"),
                    dbc.CardBody([
                        html.Div(id="comparison-metrics-table"),
                    ]),
                ], className="mb-3"),

                # Equity curves overlay
                dbc.Card([
                    dbc.CardHeader("Equity Curves"),
                    dbc.CardBody([
                        dcc.Graph(id="comparison-equity-chart"),
                    ]),
                ], className="mb-3"),

                # Drawdown comparison
                dbc.Card([
                    dbc.CardHeader("Drawdown Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="comparison-drawdown-chart"),
                    ]),
                ]),
            ], md=9),
        ]),
    ], fluid=True, className="py-3")


def create_diversification_tab() -> dbc.Container:
    """Create the Diversification tab."""
    return dbc.Container([
        dbc.Row([
            # Allocation charts
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Asset Class Allocation"),
                    dbc.CardBody([
                        dcc.Graph(id="asset-class-pie"),
                    ]),
                ]),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Region Allocation"),
                    dbc.CardBody([
                        dcc.Graph(id="region-pie"),
                    ]),
                ]),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Sector Allocation"),
                    dbc.CardBody([
                        dcc.Graph(id="sector-pie"),
                    ]),
                ]),
            ], md=4),
        ], className="mb-3"),

        dbc.Row([
            # Concentration metrics
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Concentration Metrics"),
                    dbc.CardBody([
                        html.Div(id="concentration-metrics"),
                    ]),
                ]),
            ], md=4),
            # Correlation heatmap
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Correlation Matrix"),
                    dbc.CardBody([
                        dcc.Graph(id="correlation-heatmap"),
                    ]),
                ]),
            ], md=8),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stress Test Results"),
                    dbc.CardBody([
                        dcc.Graph(id="stress-test-chart"),
                    ]),
                ]),
            ]),
        ]),
    ], fluid=True, className="py-3")


def create_hedges_tab() -> dbc.Container:
    """Create the Hedges tab."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hedging Settings"),
                    dbc.CardBody([
                        html.Label("Aggressiveness", className="fw-bold"),
                        dcc.Dropdown(
                            id="hedge-aggressiveness",
                            options=[
                                {"label": "Conservative", "value": "conservative"},
                                {"label": "Moderate", "value": "moderate"},
                                {"label": "Aggressive", "value": "aggressive"},
                            ],
                            value="moderate",
                            clearable=False,
                            className="mb-3",
                        ),

                        dbc.Button(
                            "Get Recommendations",
                            id="get-hedges-btn",
                            color="warning",
                            className="w-100",
                        ),
                    ]),
                ]),
            ], md=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Hedge Recommendations"),
                    dbc.CardBody([
                        html.Div(id="hedge-recommendations"),
                    ]),
                ], className="mb-3"),

                dbc.Card([
                    dbc.CardHeader("Apply Hedge"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="hedge-select-dropdown",
                            placeholder="Select a hedge to apply...",
                            className="mb-3",
                        ),
                        dbc.Button(
                            "Apply Selected Hedge",
                            id="apply-hedge-btn",
                            color="primary",
                            className="w-100",
                            disabled=True,
                        ),
                    ]),
                ]),
            ], md=9),
        ]),
    ], fluid=True, className="py-3")


def create_export_tab() -> dbc.Container:
    """Create the Export tab."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Export Options"),
                    dbc.CardBody([
                        html.H5("Portfolio"),
                        dbc.Button(
                            "Download Portfolio (CSV)",
                            id="export-portfolio-csv-btn",
                            color="primary",
                            className="me-2 mb-2",
                        ),
                        dbc.Button(
                            "Download Portfolio (JSON)",
                            id="export-portfolio-json-btn",
                            color="secondary",
                            className="mb-2",
                        ),

                        html.Hr(),

                        html.H5("Backtest Results"),
                        dbc.Button(
                            "Download Results (CSV)",
                            id="export-backtest-csv-btn",
                            color="primary",
                            className="me-2 mb-2",
                        ),
                        dbc.Button(
                            "Download Results (JSON)",
                            id="export-backtest-json-btn",
                            color="secondary",
                            className="mb-2",
                        ),

                        # Download components
                        dcc.Download(id="download-portfolio-csv"),
                        dcc.Download(id="download-portfolio-json"),
                        dcc.Download(id="download-backtest-csv"),
                        dcc.Download(id="download-backtest-json"),
                    ]),
                ]),
            ], md=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Export Summary"),
                    dbc.CardBody([
                        html.Div(id="export-summary"),
                    ]),
                ]),
            ], md=6),
        ]),
    ], fluid=True, className="py-3")


def create_metric_card(label: str, value: float, format_str: str = ".1%") -> dbc.Card:
    """Create a single metric card."""
    if format_str == ".1%":
        formatted_value = f"{value:.1%}"
    elif format_str == ".2f":
        formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)

    # Color based on value
    if "return" in label.lower() or "cagr" in label.lower():
        color = "success" if value > 0 else "danger"
    elif "drawdown" in label.lower():
        color = "danger"
    else:
        color = "primary"

    return dbc.Card([
        dbc.CardBody([
            html.H6(label, className="text-muted mb-1"),
            html.H4(formatted_value, className=f"text-{color} mb-0"),
        ]),
    ], className="text-center")


def create_metrics_row(metrics: dict) -> dbc.Row:
    """Create a row of metric cards."""
    cards = [
        ("Total Return", metrics.get("total_return", 0), ".1%"),
        ("CAGR", metrics.get("cagr", 0), ".1%"),
        ("Volatility", metrics.get("volatility", 0), ".1%"),
        ("Sharpe", metrics.get("sharpe_ratio", 0), ".2f"),
        ("Sortino", metrics.get("sortino_ratio", 0), ".2f"),
        ("Max Drawdown", metrics.get("max_drawdown", 0), ".1%"),
    ]

    return dbc.Row([
        dbc.Col(create_metric_card(label, value, fmt), md=2)
        for label, value, fmt in cards
    ])
