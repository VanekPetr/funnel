"""Plotly figure builders for the trading bot.

These pure functions were extracted from ``main._TradeBot`` so that the bot's
orchestration logic (backtests, lifecycle analysis, asset selection) stays
separate from the rendering code. Each function only constructs and returns
Plotly figures; none mutate bot state.
"""

from itertools import cycle
from math import ceil

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# Categorical colour ramp reused for composition bar charts.
_COMPOSITION_COLORS = (
    px.colors.sequential.turbid
    + px.colors.sequential.Brwnyl
    + px.colors.sequential.YlOrBr
    + px.colors.sequential.gray
    + px.colors.sequential.Mint
    + px.colors.sequential.dense
    + px.colors.sequential.Plasma
    + px.colors.sequential.Viridis
    + px.colors.sequential.Cividis
)


def plot_backtest(
    performance: pd.DataFrame,
    performance_benchmark: pd.DataFrame,
    composition: pd.DataFrame,
    names: list[str],
    tickers: list[str],
) -> tuple[go.Figure, go.Figure]:
    """Create performance and composition plots for backtest results.

    Args:
        performance: DataFrame with portfolio values over time.
        performance_benchmark: DataFrame with benchmark values over time.
        composition: DataFrame with portfolio weights per asset over time.
        names: Human-readable asset names.
        tickers: Ticker symbols corresponding to ``names``.

    Returns:
        Tuple of (performance line chart, composition stacked-bar chart).
    """
    performance.index = pd.to_datetime(performance.index.values, utc=True)

    # ** PERFORMANCE GRAPH **
    try:
        df_to_plot = pd.concat([performance, performance_benchmark], axis=1)
    except Exception:  # noqa: BLE001  # fall back to legacy date handling on any concat failure
        logger.warning("⚠️ Old data format.")
        performance.index = [date.date() for date in performance.index]  # needed for old data
        df_to_plot = pd.concat([performance, performance_benchmark], axis=1)

    fig_performance = px.line(
        df_to_plot,
        x=df_to_plot.index,
        y=df_to_plot.columns,
        title="Comparison of different strategies",
        color_discrete_map={"Portfolio_Value": "#21304f", "Benchmark_Value": "#f58f02"},
    )

    # ** COMPOSITION GRAPH ** — change ISIN to NAMES in allocation df
    composition_names = []
    for ticker in composition.columns:
        ticker_index = list(tickers).index(ticker)
        composition_names.append(list(names)[ticker_index])
    composition.columns = composition_names
    composition = composition.loc[:, (composition != 0).any(axis=0)]

    data = []
    for idx_color, isin in enumerate(composition.columns):
        data.append(
            go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=_COMPOSITION_COLORS[idx_color % len(_COMPOSITION_COLORS)],
            )
        )

    fig_composition = go.Figure(data=data, layout=go.Layout(barmode="stack"))
    fig_composition.update_layout(
        title="Portfolio Composition",
        xaxis_title="Number of the Investment Period",
        yaxis_title="Composition",
        legend_title="Name of the Fund",
    )
    fig_composition.layout.yaxis.tickformat = ",.1%"

    return fig_performance, fig_composition


def _density_figure(portfolio_performance_dict: dict) -> go.Figure:
    """Build the terminal-wealth density (KDE) figure across glide paths."""
    colors = [
        "#99A4AE",  # gray50
        "#3b4956",  # dark
        "#b7ada5",  # secondary
        "#4099da",  # blue
        "#8ecdc8",  # aqua
        "#e85757",  # coral
        "#fdd779",  # sun
        "#644c76",  # eggplant
        "#D8D1CA",  # warmGray50
    ]
    color_cycle = cycle(colors)  # To cycle through colors
    fig = go.Figure()
    max_density_across_all_datasets = 0  # Initialize max density tracker

    for label, df in portfolio_performance_dict.items():
        # Generate a range of values to evaluate the KDE
        x = np.linspace(df["Terminal Wealth"].min(), df["Terminal Wealth"].max(), 1000)

        # KDE per dataset; handle insufficient variance (singular covariance)
        try:
            density = gaussian_kde(df["Terminal Wealth"])(x)
        except np.linalg.LinAlgError:
            logger.warning(
                f"KDE failed for '{label}' due to insufficient variance in terminal wealth. "
                "Using histogram-based density estimation."
            )
            hist, bin_edges = np.histogram(df["Terminal Wealth"], bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            density = np.interp(x, bin_centers, hist)

        max_density_across_all_datasets = max(max_density_across_all_datasets, max(density))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=density,
                mode="lines",
                name=label,
                line={"width": 2.5, "color": next(color_cycle)},
            )
        )

    # Dashed vertical line at x=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0,
        y1=max_density_across_all_datasets,
        line={"color": "Black", "width": 3, "dash": "dash"},
    )
    fig.update_layout(
        title_text="Density function(s) of the end portfolio value for various glide paths.",
        title_font={"size": 24},
        xaxis_title="Target date portfolio value",
        xaxis_title_font={"size": 18},
        xaxis_tickfont={"size": 16},
        yaxis_title="Density",
        yaxis_title_font={"size": 18},
        yaxis_tickfont={"size": 16},
        legend_title="Risb Budget glide path",
        legend_title_font={"size": 18},
        legend_font={"size": 16},
        template="plotly_white",
    )
    return fig


def _composition_subplots(
    filtered_compositions: dict[str, pd.DataFrame],
    tickers: list,
    names: list,
) -> tuple[dict[str, go.Figure], go.Figure]:
    """Build per-portfolio composition figures plus a combined subplot grid."""
    num_portfolios = len(filtered_compositions)
    cols = 2 if num_portfolios > 1 else 1
    rows = ceil(num_portfolios / cols)  # rows needed for all compositions

    fig_subplots = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Portfolio Composition: {name}" for name in filtered_compositions],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    composition_figures: dict[str, go.Figure] = {}
    tickers_in_legend: set = set()
    current_plot = 1  # current plot index, to compute row and col

    for portfolio_name, composition in filtered_compositions.items():
        composition_names = []
        for ticker in composition.columns[:-1]:
            ticker_index = list(tickers).index(ticker)
            composition_names.append(list(names)[ticker_index])
        if "Cash" not in composition_names:
            composition_names.append("Cash")
        composition.columns = composition_names
        composition = composition.loc[:, (composition != 0).any(axis=0)]

        individual_fig = go.Figure()
        for idx_color, isin in enumerate(composition.columns):
            show_legend = isin not in tickers_in_legend
            tickers_in_legend.add(isin)
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=_COMPOSITION_COLORS[idx_color % len(_COMPOSITION_COLORS)],
                showlegend=show_legend,
            )
            row, col = divmod(current_plot - 1, cols)
            fig_subplots.add_trace(trace, row=row + 1, col=col + 1)
            individual_fig.add_trace(trace)

        individual_fig.update_layout(
            title=f"Portfolio Composition: {portfolio_name}",
            plot_bgcolor="white",
            barmode="stack",
        )
        individual_fig["layout"]["yaxis"].tickformat = ",.1%"
        composition_figures[portfolio_name] = individual_fig
        current_plot += 1

    fig_subplots.update_layout(
        title="Portfolio Compositions",
        height=500 * rows,
        width=1000 * cols,
        plot_bgcolor="white",
        barmode="stack",
    )
    for i in range(1, cols * rows + 1):
        fig_subplots["layout"][f"yaxis{i}"].tickformat = ",.1%"

    return composition_figures, fig_subplots


def plot_portfolio_densities(
    portfolio_performance_dict: dict,
    compositions: dict[str, pd.DataFrame],
    tickers: list,
    names: list,
) -> tuple[go.Figure, dict[str, go.Figure], go.Figure]:
    """Plot lifecycle simulation results: terminal-wealth densities and compositions.

    Returns:
        Tuple of (density figure, per-portfolio composition figures, combined subplots).
    """
    fig = _density_figure(portfolio_performance_dict)
    filtered_compositions = {name: comp for name, comp in compositions.items() if "reverse" not in name}
    composition_figures, fig_subplots = _composition_subplots(filtered_compositions, tickers, names)
    return fig, composition_figures, fig_subplots


def _annotate_dots_data(
    data: pd.DataFrame,
    ml: str,
    ml_subset: list | pd.DataFrame | None,
    fund_set: list,
    top_performers: list,
    optimal_portfolio: list | None,
    benchmark: list | None,
    names: list,
    tickers: list,
) -> pd.DataFrame:
    """Annotate the stats frame with portfolio/benchmark/ML/fund highlights."""
    if optimal_portfolio:
        data.loc[optimal_portfolio[4]] = optimal_portfolio
    if benchmark:
        data.loc[benchmark[4]] = benchmark

    # Highlight the subset of assets selected by an ML method
    if ml == "MST":
        data.loc[:, "Type"] = "Funds"
        for fund in ml_subset:  # ty: ignore[not-iterable]
            data.loc[fund, "Type"] = "MST subset"
    if ml == "Clustering":
        data.loc[:, "Type"] = ml_subset.loc[:, "Cluster"]  # ty: ignore[unresolved-attribute]

    for fund in fund_set:
        isin_idx = list(names).index(fund)
        data.loc[tickers[isin_idx], "Type"] = str(data.loc[tickers[isin_idx], "Name"])
        data.loc[tickers[isin_idx], "Size"] = 3

    for fund in top_performers:
        isin_idx = list(names).index(fund)
        data.loc[tickers[isin_idx], "Type"] = "Top Performer"
        data.loc[tickers[isin_idx], "Size"] = 3

    return data


def _add_risk_level_markers(fig: go.Figure, data: pd.DataFrame) -> None:
    """Add dashed vertical risk-class boundaries and their annotations."""
    min_risk = data["Standard Deviation of Returns"].min()
    max_risk = data["Standard Deviation of Returns"].max()
    risk_level = {
        "Risk Class 1": 0.005,
        "Risk Class 2": 0.02,
        "Risk Class 3": 0.05,
        "Risk Class 4": 0.10,
        "Risk Class 5": 0.15,
        "Risk Class 6": 0.25,
        "Risk Class 7": max_risk,
    }
    actual_risk_level = set()
    for i in range(1, 8):
        k = "Risk Class " + str(i)
        if (risk_level[k] >= min_risk) and (risk_level[k] <= max_risk):
            actual_risk_level.add(i)

    if max(actual_risk_level) < 7:  # pragma: no cover (dead code - 7 always in set)
        actual_risk_level.add(max(actual_risk_level) + 1)  # Add the final risk level

    for level in actual_risk_level:
        k = "Risk Class " + str(level)
        fig.add_vline(x=risk_level[k], line_width=1, line_dash="dash", line_color="#7c90a0")
        fig.add_annotation(
            x=risk_level[k] - 0.01,
            y=max(data["Average Annual Returns"]),
            text=k,
            textangle=-90,
            showarrow=False,
        )


def dots_figure(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    ml: str = "",
    ml_subset: list | pd.DataFrame | None = None,
    fund_set: list | None = None,
    top_performers: list | None = None,
    optimal_portfolio: list | None = None,
    benchmark: list | None = None,
    names: list,
    tickers: list,
) -> go.Figure:
    """Build the risk/return scatter of all products with risk-class markers."""
    fund_set = fund_set if fund_set else []
    top_performers = top_performers if top_performers else []

    data = _annotate_dots_data(
        data, ml, ml_subset, fund_set, top_performers, optimal_portfolio, benchmark, names, tickers
    )

    fig = px.scatter(
        data,
        x="Standard Deviation of Returns",
        y="Average Annual Returns",
        color="Type",
        size="Size",
        size_max=8,
        hover_name="Name",
        hover_data={"Sharpe Ratio": True, "ISIN": True, "Size": False},
        color_discrete_map={
            "ETF": "#21304f",
            "Mutual Fund": "#f58f02",
            "Funds": "#21304f",
            "MST subset": "#f58f02",
            "Top Performer": "#f58f02",
            "Cluster 1": "#21304f",
            "Cluster 2": "#f58f02",
            "Benchmark Portfolio": "#f58f02",
            "Optimal Portfolio": "olive",
        },
        title="Annual Returns and Standard Deviation of Returns from " + start_date[:10] + " to " + end_date[:10],
    )

    # Axes in percentages
    fig.layout.yaxis.tickformat = ",.1%"
    fig.layout.xaxis.tickformat = ",.1%"

    _add_risk_level_markers(fig, data)

    # Return-level marker
    fig.add_hline(y=0, line_width=1.5, line_color="rgba(233, 30, 99, 0.5)")

    fig.update_annotations(font_color="#000000")
    fig.update_layout(
        xaxis_title="Annualised standard deviation of returns (Risk)",
        yaxis_title="Annualised average returns",
    )
    fig.update_layout(legend={"yanchor": "bottom", "y": 0.01, "xanchor": "left", "x": 0.01})
    return fig
