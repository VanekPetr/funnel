"""This module contains tests for the import of the ifunnel package."""

import subprocess  # nosec B404
import sys

from ifunnel import initialize_bot


def test_bot():
    """Initializing the bot from the bundled data yields a usable instance.

    Asserts that the default (file-less) initialization loads the packaged
    returns data and exposes a valid date range and a non-empty asset universe.
    """
    # Ensure we exercise a fresh load rather than a cached instance from another test
    initialize_bot.cache_clear()

    bot = initialize_bot()

    assert bot.min_date is not None
    assert bot.max_date is not None
    assert bot.min_date <= bot.max_date
    assert len(bot.tickers) > 0
    assert len(bot.tickers) == len(bot.names)


def test_import_leaves_plotly_renderer_unchanged():
    """Importing ifunnel must not change the global plotly renderer.

    Runs in a fresh interpreter, since ifunnel is already imported in this one.
    """
    code = (
        "import plotly.io as pio; pio.renderers.default = 'json'; "
        "import ifunnel; assert pio.renderers.default == 'json', pio.renderers.default"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # nosec B603
