 """Configuration module for XAUUSD Trading Bot"""

from .settings import (
    settings,
    TELEGRAM_CONFIG,
    TRADING_CONFIG, 
    STORAGE_CONFIG,
    TelegramConfig,
    TradingConfig,
    StrategyWeights
)

__all__ = [
    'settings',
    'TELEGRAM_CONFIG',
    'TRADING_CONFIG',
    'STORAGE_CONFIG',
    'TelegramConfig',
    'TradingConfig',
    'StrategyWeights'
]