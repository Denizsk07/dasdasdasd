 """Trading module for XAUUSD Bot"""

from .risk_manager import (
    risk_manager,
    calculate_trade_risk,
    XAUUSDRiskManager,
    RiskMetrics,
    TradeRisk
)

__all__ = [
    'risk_manager',
    'calculate_trade_risk',
    'XAUUSDRiskManager',
    'RiskMetrics',
    'TradeRisk'
]