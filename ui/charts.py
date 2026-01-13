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


def create_monte_carlo_chart(
    simulation_data: dict,
    title: str = "Monte Carlo Simulation",
) -> go.Figure:
    """Create Monte Carlo simulation fan chart."""
    if "error" in simulation_data:
        return go.Figure()

    dates = simulation_data.get("dates", [])
    paths = simulation_data.get("paths_sampled", {})

    fig = go.Figure()

    # Add percentile bands (filled areas)
    if "p5" in paths and "p95" in paths:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=paths["p95"] + paths["p5"][::-1],
            fill="toself",
            fillcolor="rgba(46, 134, 171, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th-95th Percentile",
            showlegend=True,
        ))

    if "p25" in paths and "p75" in paths:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=paths["p75"] + paths["p25"][::-1],
            fill="toself",
            fillcolor="rgba(46, 134, 171, 0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25th-75th Percentile",
            showlegend=True,
        ))

    # Add median line
    if "p50" in paths:
        fig.add_trace(go.Scatter(
            x=dates,
            y=paths["p50"],
            mode="lines",
            name="Median (50th)",
            line=dict(color="#2E86AB", width=3),
        ))

    # Add percentile lines
    percentile_styles = {
        "p5": {"color": "#E74C3C", "dash": "dot", "name": "5th Percentile"},
        "p95": {"color": "#27AE60", "dash": "dot", "name": "95th Percentile"},
    }

    for key, style in percentile_styles.items():
        if key in paths:
            fig.add_trace(go.Scatter(
                x=dates,
                y=paths[key],
                mode="lines",
                name=style["name"],
                line=dict(color=style["color"], width=1.5, dash=style["dash"]),
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_rolling_returns_chart(
    rolling_data: dict,
    title: str = "Rolling Returns",
) -> go.Figure:
    """Create rolling returns line chart."""
    if "error" in rolling_data:
        return go.Figure()

    dates = rolling_data.get("dates", [])
    rolling_returns = rolling_data.get("rolling_returns", {})

    fig = go.Figure()

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, (window, values) in enumerate(rolling_returns.items()):
        # Align dates with values
        plot_dates = dates[-len(values):] if len(dates) > len(values) else dates

        fig.add_trace(go.Scatter(
            x=plot_dates,
            y=values,
            mode="lines",
            name=f"{window} Rolling",
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return fig


def create_calendar_heatmap(
    calendar_data: dict,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    """Create calendar returns heatmap."""
    if "error" in calendar_data:
        return go.Figure()

    heatmap_data = calendar_data.get("heatmap_data", [])
    months = calendar_data.get("months", [])

    if not heatmap_data:
        return go.Figure()

    # Build z matrix
    years = [d["year"] for d in heatmap_data]
    z = []
    text = []

    for row in heatmap_data:
        row_values = []
        row_text = []
        for month in months:
            val = row.get(month)
            if val is not None:
                row_values.append(val)
                row_text.append(f"{val:.1%}")
            else:
                row_values.append(None)
                row_text.append("")
        z.append(row_values)
        text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=months,
        y=years,
        colorscale=[
            [0, "#E74C3C"],      # Red for negative
            [0.5, "#FFFFFF"],    # White for zero
            [1, "#27AE60"],      # Green for positive
        ],
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title="Return",
            tickformat=".0%",
        ),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=50, r=80, t=50, b=50),
        height=max(300, len(years) * 35),
    )

    return fig


def create_contribution_chart(
    contribution_data: dict,
    title: str = "Return Contribution by Holding",
) -> go.Figure:
    """Create contribution waterfall/bar chart."""
    if "error" in contribution_data:
        return go.Figure()

    contributions = contribution_data.get("contributions", [])

    if not contributions:
        return go.Figure()

    tickers = [c["ticker"] for c in contributions]
    values = [c["contribution"] for c in contributions]
    colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in values]

    fig = go.Figure(data=[go.Bar(
        x=tickers,
        y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    )])

    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    fig.update_layout(
        title=title,
        xaxis_title="Holding",
        yaxis_title="Contribution to Return",
        yaxis_tickformat=".1%",
        margin=dict(l=50, r=30, t=50, b=50),
        height=400,
    )

    return fig


def create_yield_chart(
    dividend_data: dict,
    title: str = "Dividend Yield Breakdown",
) -> go.Figure:
    """Create dividend yield breakdown chart."""
    if "error" in dividend_data:
        return go.Figure()

    holdings = dividend_data.get("holdings_yield", [])

    if not holdings:
        return go.Figure()

    # Filter to non-zero yields
    holdings = [h for h in holdings if h["yield"] > 0]

    if not holdings:
        return go.Figure()

    tickers = [h["ticker"] for h in holdings]
    yields = [h["yield"] for h in holdings]
    contributions = [h["contribution"] for h in holdings]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Individual Yields", "Contribution to Portfolio Yield"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Individual yields
    fig.add_trace(go.Bar(
        x=tickers,
        y=yields,
        marker_color="#2E86AB",
        text=[f"{y:.1%}" for y in yields],
        textposition="outside",
        name="Yield",
    ), row=1, col=1)

    # Contributions
    fig.add_trace(go.Bar(
        x=tickers,
        y=contributions,
        marker_color="#A23B72",
        text=[f"{c:.2%}" for c in contributions],
        textposition="outside",
        name="Contribution",
    ), row=1, col=2)

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=50, r=30, t=70, b=50),
        height=400,
    )

    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=1, col=2)

    return fig
