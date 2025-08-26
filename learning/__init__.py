 """
Learning System for XAUUSD Trading Bot
Self-learning and optimization components
"""

from .strategy_optimizer import (
    strategy_optimizer,
    optimize_strategy_weights,
    learn_from_performance,
    StrategyOptimizer,
    OptimizationResult
)

__all__ = [
    'strategy_optimizer',
    'optimize_strategy_weights', 
    'learn_from_performance',
    'StrategyOptimizer',
    'OptimizationResult'
]