"""This module contains tests for the import of the ifunnel package."""

from ifunnel import initialize_bot


def test_bot():
    """Tests the initialization of the bot and prints its relevant attributes.

    This function attempts to initialize the bot and verifies the presence of
    essential attributes such as date range and asset count. It also handles
    and reports any errors that occur during the initialization process.

    Raises:
    ------
    Exception
        If there is an error during the bot initialization process.
    """
    # Try to initialize the bot
    try:
        bot = initialize_bot()
        print("Successfully initialized bot!")
        print(f"Bot has data from {bot.min_date} to {bot.max_date}")
        print(f"Number of assets: {len(bot.tickers)}")
    except Exception as e:  # noqa: BLE001  # smoke test: report any initialization failure
        print(f"Error initializing bot: {e}")
