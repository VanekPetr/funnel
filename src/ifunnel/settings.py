"""Configuration settings for the ifunnel package.

This module provides configuration settings for the ifunnel package,
including API endpoints and credentials for external data providers.
It uses environment variables loaded from a .env file when available.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings class for the ifunnel package.

    This class defines configuration variables used throughout the application,
    particularly for accessing external data sources like AlgoStrata.
    It uses pydantic_settings to handle environment variable loading and validation.

    Attributes:
        NAME: Name of the application.
        ALGOSTRATA_NAMES_URL: URL for AlgoStrata names API endpoint.
        ALGOSTRATA_PRICES_URL: URL for AlgoStrata prices API endpoint.
        ALGOSTRATA_KEY: API key for AlgoStrata services.

    Example:
        Values are read from the environment (or a ``.env`` file); keyword
        arguments take precedence:

        >>> Settings(ALGOSTRATA_KEY="my-key").ALGOSTRATA_KEY
        'my-key'
    """

    # Would you like to download the latest AlgoStrata's data?
    # TODO write to kourosh@algostrata.dk to get your own API key
    NAME: str = "Investment Funnel Secrets"
    ALGOSTRATA_NAMES_URL: str = "private"
    ALGOSTRATA_PRICES_URL: str = "notgoingtotellyou"
    ALGOSTRATA_KEY: str = "nonono"


settings = Settings()
