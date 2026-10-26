"""Investment Funnel (ifunnel) package for portfolio optimization.

This package provides tools for financial portfolio optimization, including
scenario generation, risk modeling, and optimization algorithms.

The main entry point for using this package is the `initialize_bot` function,
which creates and returns a `TradeBot` instance for portfolio analysis.

Example:
    >>> from ifunnel import initialize_bot
    >>> bot = initialize_bot()
    >>> # Now use the bot for portfolio analysis
"""

import importlib.metadata

from .models.main import TradeBot, initialize_bot

__version__ = importlib.metadata.version("ifunnel")
__all__ = ["TradeBot", "initialize_bot"]
