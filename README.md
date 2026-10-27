# [funnel](https://vanekpetr.github.io/funnel/)

[![PyPI version](https://badge.fury.io/py/ifunnel.svg)](https://badge.fury.io/py/ifunnel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/VanekPetr/funnel/actions/workflows/rhiza_ci.yml/badge.svg)](https://github.com/VanekPetr/funnel/actions/workflows/rhiza_ci.yml)

`ifunnel` is a Python-based investment funnel backend designed for asset selection, portfolio optimization, and lifecycle investing. It provides tools for data preprocessing, risk modeling, and scenario analysis to help build and backtest sophisticated investment strategies.

## Key Features

- **Asset Selection & Clustering**:
    - Hierarchical clustering based on Spearman correlation.
    - Maximum Spanning Tree (MST) for asset relationship visualization.
    - Performance-based asset selection (e.g., Sharpe ratio).
- **Risk Management**:
    - Conditional Value at Risk (CVaR) models.
    - Volatility targeting and risk budget management.
- **Portfolio Optimization**:
    - Mean-Variance Optimization (MVO).
    - Support for various solvers (e.g., Clarabel, Mosek).
    - Transaction cost modeling and weight constraints.
- **Lifecycle Investing**:
    - Glide path creation (linear, concave, convex risk profiles).
    - Lifecycle rebalancing models with withdrawal planning.
- **Scenario Generation**:
    - Monte Carlo simulations.
    - Bootstrapping methods.
    - Ledoit-Wolf and Jorion shrinkage for moment estimation.
- **Data Integration**:
    - Seamless integration with Yahoo Finance and Algostrata data.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- Python >= 3.11

### Set Up Environment

```bash
make install
```

This will create a virtual environment and install all necessary dependencies using `uv`.

### Development

For adding or removing packages:

```bash
uv add requests        # for main dependencies
uv add requests --dev  # for dev dependencies
```

Run tests:

```bash
make test
```

Start Marimo notebooks for interactive analysis:

```bash
make marimo
```

## Usage

`initialize_bot()` loads the ETF returns bundled with the package; pass a parquet
file of your own to analyse other assets. Asset selection narrows the universe
before optimising:

```python
from ifunnel import initialize_bot

bot = initialize_bot()
stats = bot.get_stat(start_date="2015-12-23", end_date="2017-07-01")
print(list(stats.columns[:3]))

_, mst_assets = bot.mst(start_date="2015-12-23", end_date="2017-07-01", n_mst_runs=3)
_, cluster_assets = bot.clustering(start_date="2015-12-23", end_date="2017-07-01", n_clusters=3, n_assets=2)
print(len(mst_assets), len(cluster_assets))
```

```result
['Average Annual Returns', 'Standard Deviation of Returns', 'Sharpe Ratio']
97 6
```

Either subset can then be passed to `bot.backtest(...)` or
`bot.lifecycle_scenario_analysis(...)`; the docstrings of both list every
parameter, and the end of `src/ifunnel/models/main.py` runs a complete example.

## Project Structure

- `src/ifunnel/models/`: Core logic for optimization, clustering, and scenario generation.
- `src/ifunnel/financial_data_preprocessing/`: Tools for fetching and cleaning financial data.
- `book/marimo/`: Interactive notebooks for exploration and demonstration.
- `tests/`: Comprehensive test suite.

## Contributing

- Fork the repository
- Create your feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add some amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
