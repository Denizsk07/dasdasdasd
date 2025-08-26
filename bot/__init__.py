"""
Telegram Bot Module for XAUUSD Trading Bot
Handles all Telegram communication and message formatting
"""

from .telegram_handler import (
    telegram_bot,
    send_trading_signal,
    send_bot_status,
    TelegramSignalBot
)

from .message_formatter import (
    message_formatter,
    format_trading_signal,
    format_performance_report,
    format_learning_update,
    MessageFormatter,
    MessageTemplate
)

__all__ = [
    'telegram_bot',
    'send_trading_signal',
    'send_bot_status',
    'TelegramSignalBot',
    'message_formatter',
    'format_trading_signal', 
    'format_performance_report',
    'format_learning_update',
    'MessageFormatter',
    'MessageTemplate'
]