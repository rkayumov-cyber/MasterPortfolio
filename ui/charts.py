"""Chart builders for Plotly visualizations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_equity_curve_chart(
    equity_data: list[dict],
    title: str = "Portfolio vs Benchmark",
) -> go.Figure:
    """Create equity curve line chart."""
    df = pd.DataFrame(equity_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["portfolio_value"],
        mode="lines",
        name="Portfolio",
        line=dict(color="#2E86AB", width=2),
    ))

    if "benchmark_value" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["benchmark_value"],
            mode="lines",
            name="Benchmark",
            line=dict(color="#A23B72", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_drawdown_chart(
    drawdown_data: list[dict],
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """Create drawdown area chart."""
    df = pd.DataFrame(drawdown_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["drawdown"],
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#E74C3C", width=1),
        fillcolor="rgba(231, 76, 60, 0.3)",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_allocation_pie(
    allocation_data: dict[str, float],
    title: str = "Allocation",
) -> go.Figure:
    """Create allocation pie chart."""
    labels = list(allocation_data.keys())
    values = list(allocation_data.values())

    colors = px.colors.qualitative.Set2

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors[:len(labels)],
        textinfo="label+percent",
        textposition="outside",
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return fig


def create_holdings_bar(
    holdings: list[dict],
    title: str = "Portfolio Holdings",
) -> go.Figure:
    """Create horizontal bar chart of holdings."""
    # Sort by weight descending
    sorted_holdings = sorted(holdings, key=lambda x: x["weight"], reverse=True)

    tickers = [h["ticker"] for h in sorted_holdings]
    weights = [h["weight"] for h in sorted_holdings]

    fig = go.Figure(data=[go.Bar(
        x=weights,
        y=tickers,
        orientation="h",
        marker_color="#2E86AB",
        text=[f"{w:.1%}" for w in weights],
        textposition="outside",
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Weight",
        xaxis_tickformat=".0%",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=50, t=50, b=50),
        height=max(300, len(holdings) * 30),
    )

    return fig


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        margin=dict(l=80, r=30, t=50, b=80),
        height=max(400, len(corr_matrix) * 50),
    )

    return fig


def create_stress_test_bars(
    stress_results: list[dict],
    title: str = "Stress Test Results",
) -> go.Figure:
    """Create horizontal bar chart for stress test results."""
    scenarios = [r["scenario"] for r in stress_results]
    portfolio_impacts = [r["portfolio_impact"] for r in stress_results]
    benchmark_impacts = [r["benchmark_impact"] for r in stress_results]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=scenarios,
        x=portfolio_impacts,
        orientation="h",
        name="Portfolio",
        marker_color="#2E86AB",
    ))

    fig.add_trace(go.Bar(
        y=scenarios,
        x=benchmark_impacts,
        orientation="h",
        name="Benchmark",
        marker_color="#A23B72",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Impact",
        xaxis_tickformat=".1%",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=150, r=50, t=80, b=50),
        height=max(300, len(scenarios) * 50),
    )

    return fig


def create_metrics_cards(metrics: dict) -> list[dict]:
    """
    Format metrics for display cards.

    Returns list of {label, value, format} dicts.
    """
    cards = [
        {
            "label": "Total Return",
            "value": metrics.get("total_return", 0),
            "format": ".1%",
        },
        {
            "label": "CAGR",
            "value": metrics.get("cagr", 0),
            "format": ".1%",
        },
        {
            "label": "Volatility",
            "value": metrics.get("volatility", 0),
            "format": ".1%",
        },
        {
            "label": "Sharpe Ratio",
            "value": metrics.get("sharpe_ratio", 0),
            "format": ".2f",
        },
        {
            "label": "Sortino Ratio",
            "value": metrics.get("sortino_ratio", 0),
            "format": ".2f",
        },
        {
            "label": "Max Drawdown",
            "value": metrics.get("max_drawdown", 0),
            "format": ".1%",
        },
    ]

    return cards


def create_rolling_chart(
    dates: list,
    portfolio_values: list,
    benchmark_values: list = None,
    title: str = "Rolling Metric",
    y_format: str = ".1%",
) -> go.Figure:
    """Create rolling metric line chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode="lines",
        name="Portfolio",
        line=dict(color="#2E86AB", width=2),
    ))

    if benchmark_values:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode="lines",
            name="Benchmark",
            line=dict(color="#A23B72", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_tickformat=y_format,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_comparison_equity_chart(
    strategy_curves: dict[str, list[dict]],
    title: str = "Strategy Comparison - Equity Curves",
) -> go.Figure:
    """Create overlay chart of multiple strategy equity curves."""
    fig = go.Figure()

    # Color palette for strategies
    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#44AF69", "#FCAB10", "#2D3047", "#93B7BE",
    ]

    for i, (strategy_name, equity_data) in enumerate(strategy_curves.items()):
        if not equity_data:
            continue
        df = pd.DataFrame(equity_data)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["portfolio_value"],
            mode="lines",
            name=strategy_name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $10,000",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        yaxis_tickformat="$,.0f",
    )

    return fig


def create_comparison_drawdown_chart(
    strategy_drawdowns: dict[str, list[dict]],
    title: str = "Strategy Comparison - Drawdowns",
) -> go.Figure:
    """Create overlay chart of multiple strategy drawdowns."""
    fig = go.Figure()

    # Color palette for strategies
    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#44AF69", "#FCAB10", "#2D3047", "#93B7BE",
    ]

    for i, (strategy_name, drawdown_data) in enumerate(strategy_drawdowns.items()):
        if not drawdown_data:
            continue
        df = pd.DataFrame(drawdown_data)
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            mode="lines",
            name=strategy_name,
            line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_hedge_recommendations_table(recommendations: list[dict]) -> go.Figure:
    """Create table figure for hedge recommendations."""
    if not recommendations:
        return go.Figure()

    headers = ["Instrument", "Risk Targeted", "Suggested Weight", "Rationale"]

    cells = [
        [r["instrument"] for r in recommendations],
        [r["risk"] for r in recommendations],
        [f"{r['weight']:.1%}" for r in recommendations],
        [r["rationale"][:50] + "..." if len(r["rationale"]) > 50 else r["rationale"]
         for r in recommendations],
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#2E86AB",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cells,
            fill_color="white",
            font=dict(size=11),
            align="left",
            height=30,
        ),
    )])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(200, len(recommendations) * 40 + 50),
    )

    return fig
