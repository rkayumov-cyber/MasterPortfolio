"""Chart builders for Plotly visualizations - Bloomberg Style."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# Bloomberg Color Palette
# ============================================================================
BB_COLORS = {
    "orange": "#FF6600",
    "yellow": "#FFCC00",
    "green": "#00CC66",
    "red": "#FF3333",
    "blue": "#3399FF",
    "white": "#FFFFFF",
    "black": "#000000",
    "text": "#E0E0E0",
    "muted": "#888888",
    "bg": "#000000",
    "dark": "#0A0A0A",
    "paper": "#0A0A0A",
    "grid": "#333333",
}

# Bloomberg-style color sequence for multiple series
BB_COLOR_SEQUENCE = [
    "#FF6600",  # Orange
    "#FFCC00",  # Yellow
    "#00CC66",  # Green
    "#3399FF",  # Blue
    "#FF3333",  # Red
    "#CC99FF",  # Purple
    "#00CCCC",  # Cyan
    "#FF9966",  # Light orange
    "#99FF66",  # Light green
]

# Bloomberg chart layout template
BB_LAYOUT = dict(
    paper_bgcolor=BB_COLORS["bg"],
    plot_bgcolor=BB_COLORS["bg"],
    font=dict(
        family="Consolas, Monaco, monospace",
        size=12,
        color=BB_COLORS["text"],
    ),
    title=dict(
        font=dict(
            size=14,
            color=BB_COLORS["orange"],
        ),
        x=0,
        xanchor="left",
    ),
    xaxis=dict(
        gridcolor=BB_COLORS["grid"],
        linecolor=BB_COLORS["grid"],
        tickfont=dict(color=BB_COLORS["muted"]),
        title_font=dict(color=BB_COLORS["text"]),
    ),
    yaxis=dict(
        gridcolor=BB_COLORS["grid"],
        linecolor=BB_COLORS["grid"],
        tickfont=dict(color=BB_COLORS["muted"]),
        title_font=dict(color=BB_COLORS["text"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=BB_COLORS["text"]),
    ),
    hoverlabel=dict(
        bgcolor=BB_COLORS["paper"],
        font_size=12,
        font_family="Consolas, monospace",
    ),
)


def apply_bb_layout(fig: go.Figure) -> go.Figure:
    """Apply Bloomberg-style layout to a figure."""
    fig.update_layout(**BB_LAYOUT)
    return fig


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
        line=dict(color=BB_COLORS["orange"], width=2),
    ))

    if "benchmark_value" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["benchmark_value"],
            mode="lines",
            name="Benchmark",
            line=dict(color=BB_COLORS["yellow"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


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
        line=dict(color=BB_COLORS["red"], width=1),
        fillcolor="rgba(255, 51, 51, 0.3)",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


def create_allocation_pie(
    allocation_data: dict[str, float],
    title: str = "Allocation",
) -> go.Figure:
    """Create allocation pie chart."""
    labels = list(allocation_data.keys())
    values = list(allocation_data.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=BB_COLOR_SEQUENCE[:len(labels)],
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return apply_bb_layout(fig)


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
        marker_color=BB_COLORS["orange"],
        text=[f"{w:.1%}" for w in weights],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
    )])

    fig.update_layout(
        title=title,
        xaxis_title="Weight",
        xaxis_tickformat=".0%",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=50, t=50, b=50),
        height=max(300, len(holdings) * 30),
    )

    return apply_bb_layout(fig)


def create_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create correlation heatmap."""
    # Bloomberg-style colorscale: red (negative) -> black (zero) -> green (positive)
    bb_colorscale = [
        [0, BB_COLORS["red"]],
        [0.5, "#1a1a1a"],
        [1, BB_COLORS["green"]],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=bb_colorscale,
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10, "color": BB_COLORS["text"]},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        margin=dict(l=80, r=30, t=50, b=80),
        height=max(400, len(corr_matrix) * 50),
    )

    return apply_bb_layout(fig)


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
        marker_color=BB_COLORS["orange"],
    ))

    fig.add_trace(go.Bar(
        y=scenarios,
        x=benchmark_impacts,
        orientation="h",
        name="Benchmark",
        marker_color=BB_COLORS["yellow"],
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

    return apply_bb_layout(fig)


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
        line=dict(color=BB_COLORS["orange"], width=2),
    ))

    if benchmark_values:
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode="lines",
            name="Benchmark",
            line=dict(color=BB_COLORS["yellow"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_tickformat=y_format,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


def create_comparison_equity_chart(
    strategy_curves: dict[str, list[dict]],
    title: str = "Strategy Comparison - Equity Curves",
) -> go.Figure:
    """Create overlay chart of multiple strategy equity curves."""
    fig = go.Figure()

    for i, (strategy_name, equity_data) in enumerate(strategy_curves.items()):
        if not equity_data:
            continue
        df = pd.DataFrame(equity_data)
        color = BB_COLOR_SEQUENCE[i % len(BB_COLOR_SEQUENCE)]

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
            bgcolor="rgba(0,0,0,0.8)",
        ),
        margin=dict(l=50, r=30, t=50, b=50),
        yaxis_tickformat="$,.0f",
    )

    return apply_bb_layout(fig)


def create_comparison_drawdown_chart(
    strategy_drawdowns: dict[str, list[dict]],
    title: str = "Strategy Comparison - Drawdowns",
) -> go.Figure:
    """Create overlay chart of multiple strategy drawdowns."""
    fig = go.Figure()

    for i, (strategy_name, drawdown_data) in enumerate(strategy_drawdowns.items()):
        if not drawdown_data:
            continue
        df = pd.DataFrame(drawdown_data)
        color = BB_COLOR_SEQUENCE[i % len(BB_COLOR_SEQUENCE)]

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
            bgcolor="rgba(0,0,0,0.8)",
        ),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


def create_hedge_recommendations_table(recommendations: list[dict]) -> go.Figure:
    """Create table figure for hedge recommendations."""
    if not recommendations:
        fig = go.Figure()
        return apply_bb_layout(fig)

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
            fill_color=BB_COLORS["orange"],
            font=dict(color=BB_COLORS["black"], size=12, family="Consolas, monospace"),
            align="left",
        ),
        cells=dict(
            values=cells,
            fill_color=BB_COLORS["dark"],
            font=dict(size=11, color=BB_COLORS["text"], family="Consolas, monospace"),
            align="left",
            height=30,
            line_color=BB_COLORS["grid"],
        ),
    )])

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(200, len(recommendations) * 40 + 50),
    )

    return apply_bb_layout(fig)


def create_monte_carlo_chart(
    simulation_data: dict,
    title: str = "Monte Carlo Simulation",
) -> go.Figure:
    """Create Monte Carlo simulation fan chart."""
    if "error" in simulation_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    dates = simulation_data.get("dates", [])
    paths = simulation_data.get("paths_sampled", {})

    fig = go.Figure()

    # Add percentile bands (filled areas)
    if "p5" in paths and "p95" in paths:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=paths["p95"] + paths["p5"][::-1],
            fill="toself",
            fillcolor="rgba(255, 102, 0, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th-95th Percentile",
            showlegend=True,
        ))

    if "p25" in paths and "p75" in paths:
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=paths["p75"] + paths["p25"][::-1],
            fill="toself",
            fillcolor="rgba(255, 102, 0, 0.2)",
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
            line=dict(color=BB_COLORS["orange"], width=3),
        ))

    # Add percentile lines
    percentile_styles = {
        "p5": {"color": BB_COLORS["red"], "dash": "dot", "name": "5th Percentile"},
        "p95": {"color": BB_COLORS["green"], "dash": "dot", "name": "95th Percentile"},
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

    return apply_bb_layout(fig)


def create_rolling_returns_chart(
    rolling_data: dict,
    title: str = "Rolling Returns",
) -> go.Figure:
    """Create rolling returns line chart."""
    if "error" in rolling_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    dates = rolling_data.get("dates", [])
    rolling_returns = rolling_data.get("rolling_returns", {})

    fig = go.Figure()

    for i, (window, values) in enumerate(rolling_returns.items()):
        # Align dates with values
        plot_dates = dates[-len(values):] if len(dates) > len(values) else dates

        fig.add_trace(go.Scatter(
            x=plot_dates,
            y=values,
            mode="lines",
            name=f"{window} Rolling",
            line=dict(color=BB_COLOR_SEQUENCE[i % len(BB_COLOR_SEQUENCE)], width=2),
        ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color=BB_COLORS["muted"], opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


def create_calendar_heatmap(
    calendar_data: dict,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    """Create calendar returns heatmap."""
    if "error" in calendar_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    heatmap_data = calendar_data.get("heatmap_data", [])
    months = calendar_data.get("months", [])

    if not heatmap_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

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

    # Bloomberg colorscale: red -> black -> green
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=months,
        y=years,
        colorscale=[
            [0, BB_COLORS["red"]],
            [0.5, "#1a1a1a"],
            [1, BB_COLORS["green"]],
        ],
        zmid=0,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 10, "color": BB_COLORS["text"]},
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title=dict(
                text="Return",
                font=dict(color=BB_COLORS["text"]),
            ),
            tickformat=".0%",
            tickfont=dict(color=BB_COLORS["text"]),
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

    return apply_bb_layout(fig)


def create_contribution_chart(
    contribution_data: dict,
    title: str = "Return Contribution by Holding",
) -> go.Figure:
    """Create contribution waterfall/bar chart."""
    if "error" in contribution_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    contributions = contribution_data.get("contributions", [])

    if not contributions:
        fig = go.Figure()
        return apply_bb_layout(fig)

    tickers = [c["ticker"] for c in contributions]
    values = [c["contribution"] for c in contributions]
    colors = [BB_COLORS["green"] if v >= 0 else BB_COLORS["red"] for v in values]

    fig = go.Figure(data=[go.Bar(
        x=tickers,
        y=values,
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
    )])

    fig.add_hline(y=0, line_dash="solid", line_color=BB_COLORS["muted"], opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Holding",
        yaxis_title="Contribution to Return",
        yaxis_tickformat=".1%",
        margin=dict(l=50, r=30, t=50, b=50),
        height=400,
    )

    return apply_bb_layout(fig)


def create_yield_chart(
    dividend_data: dict,
    title: str = "Dividend Yield Breakdown",
) -> go.Figure:
    """Create dividend yield breakdown chart."""
    if "error" in dividend_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    holdings = dividend_data.get("holdings_yield", [])

    if not holdings:
        fig = go.Figure()
        return apply_bb_layout(fig)

    # Filter to non-zero yields
    holdings = [h for h in holdings if h["yield"] > 0]

    if not holdings:
        fig = go.Figure()
        return apply_bb_layout(fig)

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
        marker_color=BB_COLORS["orange"],
        text=[f"{y:.1%}" for y in yields],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
        name="Yield",
    ), row=1, col=1)

    # Contributions
    fig.add_trace(go.Bar(
        x=tickers,
        y=contributions,
        marker_color=BB_COLORS["yellow"],
        text=[f"{c:.2%}" for c in contributions],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
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

    # Update subplot title colors
    fig.update_annotations(font=dict(color=BB_COLORS["orange"]))

    return apply_bb_layout(fig)


# ============================================================================
# Optimizer Charts
# ============================================================================


def create_efficient_frontier_chart(
    all_portfolios: list[dict],
    frontier_data: list[dict],
    optimal_point: dict = None,
    title: str = "Efficient Frontier",
) -> go.Figure:
    """
    Create efficient frontier scatter plot.

    X-axis: Volatility
    Y-axis: Expected Return (CAGR)
    Color: Sharpe Ratio

    Args:
        all_portfolios: List of all evaluated portfolios
        frontier_data: Points on the efficient frontier
        optimal_point: The optimal portfolio to highlight
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Bloomberg-style colorscale for Sharpe ratio
    bb_sharpe_colorscale = [
        [0, BB_COLORS["red"]],
        [0.5, BB_COLORS["yellow"]],
        [1, BB_COLORS["green"]],
    ]

    # All portfolios as background scatter
    if all_portfolios:
        vols = [p["volatility"] for p in all_portfolios]
        returns = [p["cagr"] for p in all_portfolios]
        sharpes = [p["sharpe_ratio"] for p in all_portfolios]

        fig.add_trace(go.Scatter(
            x=vols,
            y=returns,
            mode="markers",
            name="All Portfolios",
            marker=dict(
                size=6,
                color=sharpes,
                colorscale=bb_sharpe_colorscale,
                colorbar=dict(
                    title=dict(
                        text="Sharpe",
                        font=dict(color=BB_COLORS["text"]),
                    ),
                    x=1.02,
                    tickfont=dict(color=BB_COLORS["text"]),
                ),
                showscale=True,
                opacity=0.7,
            ),
            hovertemplate=(
                "<b>Volatility:</b> %{x:.2%}<br>"
                "<b>Return:</b> %{y:.2%}<br>"
                "<b>Sharpe:</b> %{marker.color:.2f}<extra></extra>"
            ),
        ))

    # Efficient frontier line
    if frontier_data:
        frontier_vols = [p["volatility"] for p in frontier_data]
        frontier_returns = [p["return"] for p in frontier_data]

        fig.add_trace(go.Scatter(
            x=frontier_vols,
            y=frontier_returns,
            mode="lines+markers",
            name="Efficient Frontier",
            marker=dict(size=8, color=BB_COLORS["orange"]),
            line=dict(color=BB_COLORS["orange"], width=2),
            hovertemplate=(
                "<b>Frontier Point</b><br>"
                "<b>Volatility:</b> %{x:.2%}<br>"
                "<b>Return:</b> %{y:.2%}<extra></extra>"
            ),
        ))

    # Highlight optimal point
    if optimal_point:
        fig.add_trace(go.Scatter(
            x=[optimal_point["volatility"]],
            y=[optimal_point["return"]],
            mode="markers",
            name="Optimal Portfolio",
            marker=dict(
                size=20,
                color=BB_COLORS["yellow"],
                symbol="star",
                line=dict(width=2, color=BB_COLORS["black"]),
            ),
            hovertemplate=(
                "<b>OPTIMAL</b><br>"
                "<b>Volatility:</b> %{x:.2%}<br>"
                "<b>Return:</b> %{y:.2%}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Volatility (Annualized)",
        yaxis_title="Expected Return (CAGR)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=80, t=50, b=60),
    )

    return apply_bb_layout(fig)


def create_optimal_weights_chart(
    weights: dict[str, float],
    title: str = "Optimal Portfolio Allocation",
) -> go.Figure:
    """
    Create pie chart for optimal portfolio weights.

    Args:
        weights: Dict mapping ticker to weight
        title: Chart title

    Returns:
        Plotly figure
    """
    # Filter out zero/tiny weights
    filtered = {k: v for k, v in weights.items() if v > 0.001}

    if not filtered:
        fig = go.Figure()
        return apply_bb_layout(fig)

    labels = list(filtered.keys())
    values = list(filtered.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=BB_COLOR_SEQUENCE[:len(labels)],
        textinfo="label+percent",
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return apply_bb_layout(fig)


def create_optimizer_comparison_chart(
    equity_data: list[dict],
    title: str = "Optimal Portfolio vs Benchmark",
) -> go.Figure:
    """
    Create equity curve comparison for optimized portfolio vs benchmark.

    Args:
        equity_data: List of {date, portfolio_value, benchmark_value}
        title: Chart title

    Returns:
        Plotly figure
    """
    if not equity_data:
        fig = go.Figure()
        return apply_bb_layout(fig)

    df = pd.DataFrame(equity_data)

    fig = go.Figure()

    # Optimal portfolio
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["portfolio_value"],
        mode="lines",
        name="Optimal Portfolio",
        line=dict(color=BB_COLORS["green"], width=2),
    ))

    # Benchmark
    if "benchmark_value" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["benchmark_value"],
            mode="lines",
            name="Benchmark (SPY)",
            line=dict(color=BB_COLORS["yellow"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Growth of $10,000",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    return apply_bb_layout(fig)


def create_optimization_scatter_matrix(
    portfolios: list[dict],
    title: str = "Portfolio Metrics Relationships",
) -> go.Figure:
    """
    Create scatter matrix showing relationships between portfolio metrics.

    Args:
        portfolios: List of portfolio dicts with metrics
        title: Chart title

    Returns:
        Plotly figure
    """
    if not portfolios:
        fig = go.Figure()
        return apply_bb_layout(fig)

    df = pd.DataFrame(portfolios)

    # Bloomberg-style colorscale
    bb_sharpe_colorscale = [
        [0, BB_COLORS["red"]],
        [0.5, BB_COLORS["yellow"]],
        [1, BB_COLORS["green"]],
    ]

    fig = px.scatter_matrix(
        df,
        dimensions=["volatility", "cagr", "sharpe_ratio", "max_drawdown"],
        color="sharpe_ratio",
        color_continuous_scale=bb_sharpe_colorscale,
        title=title,
        labels={
            "volatility": "Vol",
            "cagr": "CAGR",
            "sharpe_ratio": "Sharpe",
            "max_drawdown": "Max DD",
        },
    )

    fig.update_layout(
        margin=dict(l=50, r=50, t=80, b=50),
        height=600,
    )

    fig.update_traces(diagonal_visible=False)

    return apply_bb_layout(fig)


# =============================================================================
# Market Regime Charts
# =============================================================================


def create_regime_gauge(
    regime_value: float,
    confidence: float,
    regime_label: str,
    title: str = "Market Regime",
) -> go.Figure:
    """
    Create market regime gauge indicator.

    Args:
        regime_value: Value from -1 (bearish) to +1 (bullish)
        confidence: Confidence level 0-1
        regime_label: Text label for current regime
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=regime_value,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "suffix": "",
            "font": {"size": 24, "color": BB_COLORS["text"]},
            "valueformat": "+.2f",
        },
        gauge={
            "axis": {
                "range": [-1, 1],
                "tickwidth": 1,
                "tickcolor": BB_COLORS["text"],
                "tickvals": [-1, -0.5, 0, 0.5, 1],
                "ticktext": ["Bear", "", "Neutral", "", "Bull"],
            },
            "bar": {"color": BB_COLORS["orange"], "thickness": 0.75},
            "bgcolor": BB_COLORS["bg"],
            "borderwidth": 2,
            "bordercolor": BB_COLORS["grid"],
            "steps": [
                {"range": [-1, -0.5], "color": BB_COLORS["red"]},
                {"range": [-0.5, -0.15], "color": "#CC4444"},
                {"range": [-0.15, 0.15], "color": BB_COLORS["yellow"]},
                {"range": [0.15, 0.5], "color": "#44AA44"},
                {"range": [0.5, 1], "color": BB_COLORS["green"]},
            ],
            "threshold": {
                "line": {"color": BB_COLORS["white"], "width": 4},
                "thickness": 0.8,
                "value": regime_value,
            },
        },
    ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=30),
        height=250,
        annotations=[
            dict(
                text=f"<b>{regime_label}</b><br>Confidence: {confidence:.0%}",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=14, color=BB_COLORS["text"]),
                xanchor="center",
            ),
        ],
    )

    return apply_bb_layout(fig)


def create_fear_greed_gauge(
    score: float,
    label: str,
    title: str = "Fear & Greed Index",
) -> go.Figure:
    """
    Create Fear & Greed Index gauge.

    Args:
        score: Score from 0 (extreme fear) to 100 (extreme greed)
        label: Text label (e.g., "Greed", "Fear")
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "font": {"size": 32, "color": BB_COLORS["text"]},
            "valueformat": ".0f",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": BB_COLORS["text"],
                "tickvals": [0, 25, 50, 75, 100],
                "ticktext": ["Fear", "", "Neutral", "", "Greed"],
            },
            "bar": {"color": BB_COLORS["orange"], "thickness": 0.75},
            "bgcolor": BB_COLORS["bg"],
            "borderwidth": 2,
            "bordercolor": BB_COLORS["grid"],
            "steps": [
                {"range": [0, 25], "color": BB_COLORS["red"]},
                {"range": [25, 45], "color": "#CC6644"},
                {"range": [45, 55], "color": BB_COLORS["yellow"]},
                {"range": [55, 75], "color": "#66AA66"},
                {"range": [75, 100], "color": BB_COLORS["green"]},
            ],
            "threshold": {
                "line": {"color": BB_COLORS["white"], "width": 4},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=60, b=30),
        height=250,
        annotations=[
            dict(
                text=f"<b>{label}</b>",
                x=0.5,
                y=-0.1,
                showarrow=False,
                font=dict(size=14, color=BB_COLORS["text"]),
                xanchor="center",
            ),
        ],
    )

    return apply_bb_layout(fig)


def create_sentiment_bars(
    sentiment_data: list[dict],
    title: str = "Sentiment by Source",
) -> go.Figure:
    """
    Create horizontal bar chart for sentiment from multiple sources.

    Args:
        sentiment_data: List of {source, score, label} dicts
            score ranges from -1 (bearish) to +1 (bullish)
        title: Chart title

    Returns:
        Plotly figure
    """
    if not sentiment_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=BB_COLORS["muted"]),
        )
        return apply_bb_layout(fig)

    sources = [d["source"] for d in sentiment_data]
    scores = [d["score"] for d in sentiment_data]
    labels = [d.get("label", "") for d in sentiment_data]

    colors = []
    for score in scores:
        if score > 0.3:
            colors.append(BB_COLORS["green"])
        elif score < -0.3:
            colors.append(BB_COLORS["red"])
        else:
            colors.append(BB_COLORS["yellow"])

    fig = go.Figure(data=[go.Bar(
        y=sources,
        x=scores,
        orientation="h",
        marker_color=colors,
        text=[f"{s:+.0%} ({l})" for s, l in zip(scores, labels)],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
        hovertemplate="<b>%{y}</b><br>Score: %{x:+.2f}<extra></extra>",
    )])

    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[-1.2, 1.2],
            title="Bearish ← Sentiment → Bullish",
            zeroline=True,
            zerolinecolor=BB_COLORS["muted"],
            zerolinewidth=2,
        ),
        yaxis=dict(title=""),
        margin=dict(l=120, r=80, t=50, b=50),
        height=max(200, len(sources) * 50 + 100),
    )

    return apply_bb_layout(fig)


def create_allocation_comparison_chart(
    current: dict[str, float],
    recommended: dict[str, float],
    title: str = "Current vs Recommended Allocation",
) -> go.Figure:
    """
    Create grouped bar chart comparing current vs recommended allocation.

    Args:
        current: Current allocation {asset_class: weight}
        recommended: Recommended allocation {asset_class: weight}
        title: Chart title

    Returns:
        Plotly figure
    """
    asset_classes = list(current.keys())

    fig = go.Figure()

    # Current allocation bars
    fig.add_trace(go.Bar(
        name="Current",
        x=asset_classes,
        y=[current.get(ac, 0) for ac in asset_classes],
        marker_color=BB_COLORS["muted"],
        text=[f"{current.get(ac, 0):.0%}" for ac in asset_classes],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
    ))

    # Recommended allocation bars
    fig.add_trace(go.Bar(
        name="Recommended",
        x=asset_classes,
        y=[recommended.get(ac, 0) for ac in asset_classes],
        marker_color=BB_COLORS["orange"],
        text=[f"{recommended.get(ac, 0):.0%}" for ac in asset_classes],
        textposition="outside",
        textfont=dict(color=BB_COLORS["text"]),
    ))

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Asset Class",
        yaxis_title="Allocation",
        yaxis_tickformat=".0%",
        yaxis_range=[0, max(max(current.values()), max(recommended.values())) * 1.2],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=80, b=50),
        height=300,
    )

    return apply_bb_layout(fig)


# =============================================================================
# FRED Economic Indicator Charts
# =============================================================================


def create_fred_yield_curve_chart(df: "pd.DataFrame") -> go.Figure:
    """
    Create yield curve spread historical chart.

    Args:
        df: DataFrame with 'date' and 'value' columns

    Returns:
        Plotly figure
    """
    import pandas as pd

    fig = go.Figure()

    # Add zero line for reference (inversion threshold)
    fig.add_hline(y=0, line_dash="dash", line_color=BB_COLORS["red"], opacity=0.7)

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name="10Y-2Y Spread",
        line=dict(color=BB_COLORS["orange"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 140, 0, 0.2)",
    ))

    # Add annotation for current value
    if len(df) > 0:
        current = df["value"].iloc[-1]
        status = "Normal" if current > 0 else "INVERTED"
        color = BB_COLORS["green"] if current > 0 else BB_COLORS["red"]

        fig.add_annotation(
            x=df["date"].iloc[-1],
            y=current,
            text=f"{current:.2f}% ({status})",
            showarrow=True,
            arrowhead=2,
            arrowcolor=color,
            font=dict(color=color, size=11),
            bgcolor=BB_COLORS["background"],
        )

    fig.update_layout(
        title="Yield Curve (10Y - 2Y Treasury Spread)",
        xaxis_title="",
        yaxis_title="Spread (%)",
        height=280,
        margin=dict(l=50, r=30, t=40, b=30),
        showlegend=False,
    )

    return apply_bb_layout(fig)


def create_fred_credit_spreads_chart(
    ig_df: "pd.DataFrame" = None,
    hy_df: "pd.DataFrame" = None,
) -> go.Figure:
    """
    Create credit spreads historical chart.

    Args:
        ig_df: Investment grade spread DataFrame
        hy_df: High yield spread DataFrame

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    if ig_df is not None and not ig_df.empty:
        fig.add_trace(go.Scatter(
            x=ig_df["date"],
            y=ig_df["value"],
            mode="lines",
            name="Investment Grade",
            line=dict(color=BB_COLORS["blue"], width=2),
        ))

    if hy_df is not None and not hy_df.empty:
        fig.add_trace(go.Scatter(
            x=hy_df["date"],
            y=hy_df["value"],
            mode="lines",
            name="High Yield",
            line=dict(color=BB_COLORS["orange"], width=2),
        ))

    # Add stress threshold lines
    fig.add_hline(y=200, line_dash="dot", line_color=BB_COLORS["yellow"], opacity=0.5,
                  annotation_text="IG Stress", annotation_position="right")
    fig.add_hline(y=500, line_dash="dot", line_color=BB_COLORS["red"], opacity=0.5,
                  annotation_text="HY Stress", annotation_position="right")

    fig.update_layout(
        title="Credit Spreads (Option-Adjusted)",
        xaxis_title="",
        yaxis_title="Spread (bps)",
        height=280,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return apply_bb_layout(fig)


def create_fred_unemployment_chart(
    rate_df: "pd.DataFrame" = None,
    claims_df: "pd.DataFrame" = None,
) -> go.Figure:
    """
    Create unemployment rate and claims historical chart.

    Args:
        rate_df: Unemployment rate DataFrame
        claims_df: Initial claims DataFrame

    Returns:
        Plotly figure with dual y-axes
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if rate_df is not None and not rate_df.empty:
        fig.add_trace(
            go.Scatter(
                x=rate_df["date"],
                y=rate_df["value"],
                mode="lines",
                name="Unemployment Rate",
                line=dict(color=BB_COLORS["red"], width=2),
            ),
            secondary_y=False,
        )

    if claims_df is not None and not claims_df.empty:
        fig.add_trace(
            go.Scatter(
                x=claims_df["date"],
                y=claims_df["value"] / 1000,  # Convert to thousands
                mode="lines",
                name="Initial Claims (K)",
                line=dict(color=BB_COLORS["orange"], width=1.5),
                opacity=0.7,
            ),
            secondary_y=True,
        )

    fig.update_yaxes(title_text="Unemployment Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Initial Claims (K)", secondary_y=True)

    fig.update_layout(
        title="Labor Market Indicators",
        xaxis_title="",
        height=280,
        margin=dict(l=50, r=50, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return apply_bb_layout(fig)


def create_fred_sentiment_chart(df: "pd.DataFrame") -> go.Figure:
    """
    Create consumer sentiment historical chart.

    Args:
        df: DataFrame with 'date' and 'value' columns

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Historical average line
    avg = df["value"].mean() if len(df) > 0 else 80

    fig.add_hline(y=avg, line_dash="dash", line_color=BB_COLORS["muted"], opacity=0.7,
                  annotation_text=f"Avg: {avg:.0f}", annotation_position="right")

    # Sentiment line with color gradient based on level
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name="Consumer Sentiment",
        line=dict(color=BB_COLORS["cyan"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 255, 0.1)",
    ))

    # Threshold zones
    fig.add_hline(y=60, line_dash="dot", line_color=BB_COLORS["red"], opacity=0.4)
    fig.add_hline(y=100, line_dash="dot", line_color=BB_COLORS["green"], opacity=0.4)

    fig.update_layout(
        title="Consumer Sentiment (U. of Michigan)",
        xaxis_title="",
        yaxis_title="Index",
        height=280,
        margin=dict(l=50, r=30, t=40, b=30),
        showlegend=False,
    )

    return apply_bb_layout(fig)


def create_fred_fed_funds_chart(df: "pd.DataFrame") -> go.Figure:
    """
    Create Fed Funds rate historical chart.

    Args:
        df: DataFrame with 'date' and 'value' columns

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["value"],
        mode="lines",
        name="Fed Funds Rate",
        line=dict(color=BB_COLORS["green"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 255, 0, 0.1)",
    ))

    # Add current value annotation
    if len(df) > 0:
        current = df["value"].iloc[-1]
        fig.add_annotation(
            x=df["date"].iloc[-1],
            y=current,
            text=f"{current:.2f}%",
            showarrow=True,
            arrowhead=2,
            arrowcolor=BB_COLORS["green"],
            font=dict(color=BB_COLORS["green"], size=11),
            bgcolor=BB_COLORS["background"],
        )

    fig.update_layout(
        title="Federal Funds Rate",
        xaxis_title="",
        yaxis_title="Rate (%)",
        height=280,
        margin=dict(l=50, r=30, t=40, b=30),
        showlegend=False,
    )

    return apply_bb_layout(fig)


def create_fred_combined_chart(historical_data: dict) -> go.Figure:
    """
    Create a combined multi-panel chart for all FRED indicators.

    Args:
        historical_data: Dict from get_fred_historical_data()

    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Yield Curve (10Y-2Y)",
            "Credit Spreads",
            "Unemployment Rate",
            "Consumer Sentiment",
            "Fed Funds Rate",
            "Initial Claims",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 1. Yield Curve (row 1, col 1)
    if "yield_curve" in historical_data:
        df = historical_data["yield_curve"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["orange"], width=1.5),
                      name="Yield Curve", showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color=BB_COLORS["red"],
                     opacity=0.5, row=1, col=1)

    # 2. Credit Spreads (row 1, col 2)
    if "credit_spread_ig" in historical_data:
        df = historical_data["credit_spread_ig"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["blue"], width=1.5),
                      name="IG Spread"),
            row=1, col=2
        )
    if "credit_spread_hy" in historical_data:
        df = historical_data["credit_spread_hy"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["orange"], width=1.5),
                      name="HY Spread"),
            row=1, col=2
        )

    # 3. Unemployment (row 2, col 1)
    if "unemployment_rate" in historical_data:
        df = historical_data["unemployment_rate"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["red"], width=1.5),
                      name="Unemployment", showlegend=False),
            row=2, col=1
        )

    # 4. Consumer Sentiment (row 2, col 2)
    if "consumer_sentiment" in historical_data:
        df = historical_data["consumer_sentiment"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["cyan"], width=1.5),
                      name="Sentiment", showlegend=False),
            row=2, col=2
        )

    # 5. Fed Funds (row 3, col 1)
    if "fed_funds" in historical_data:
        df = historical_data["fed_funds"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"], mode="lines",
                      line=dict(color=BB_COLORS["green"], width=1.5),
                      name="Fed Funds", showlegend=False),
            row=3, col=1
        )

    # 6. Initial Claims (row 3, col 2)
    if "initial_claims" in historical_data:
        df = historical_data["initial_claims"]
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["value"] / 1000, mode="lines",
                      line=dict(color=BB_COLORS["orange"], width=1.5),
                      name="Claims (K)", showlegend=False),
            row=3, col=2
        )

    fig.update_layout(
        height=700,
        margin=dict(l=50, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title="FRED Economic Indicators - Historical View",
    )

    return apply_bb_layout(fig)
