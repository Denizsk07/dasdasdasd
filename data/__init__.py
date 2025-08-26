"""Data providers for XAUUSD Trading Bot"""

from .market_provider import (
    market_data,
    get_current_price,
    get_historical_data,
    is_market_open,
    PriceData,
    MarketStatus
)

__all__ = [
    'market_data',
    'get_current_price', 
    'get_historical_data',
    'is_market_open',
    'PriceData',
    'MarketStatus'
]