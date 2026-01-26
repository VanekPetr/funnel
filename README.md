# [funnel](https://tschm.github.io/funnel)

[![PyPI version](https://badge.fury.io/py/funnel.svg)](https://badge.fury.io/py/funnel)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tschm/funnel/actions/workflows/ci.yml/badge.svg)](https://github.com/tschm/funnel/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/tschm/funnel/badge.svg?branch=main)](https://coveralls.io/github/tschm/funnel?branch=main)

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
