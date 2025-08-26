"""XAUUSD Risk Manager - Advanced Position Sizing & Risk Control"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('risk_manager')

@dataclass
class RiskMetrics:
    position_size: float
    max_loss_usd: float
    max_profit_usd: float
    risk_reward_ratios: List[float]
    portfolio_risk_pct: float
    margin_required: float

@dataclass
class TradeRisk:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    position_size: float
    risk_amount: float
    risk_pct: float

class XAUUSDRiskManager:
    def __init__(self):
        self.account_balance = 10000  # Default
        self.max_risk_per_trade = settings.trading.risk_percentage / 100
        self.max_daily_risk = 0.06  # 6% max daily risk
        self.daily_risk_used = 0.0
        self.open_positions = []
        self.daily_reset_date = datetime.now().date()
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_pct: Optional[float] = None) -> RiskMetrics:
        """Calculate optimal position size"""
        
        if risk_pct is None:
            risk_pct = self.max_risk_per_trade
        
        # Reset daily risk if new day
        self._reset_daily_risk_if_needed()
        
        # Available risk
        remaining_daily_risk = self.max_daily_risk - self.daily_risk_used
        actual_risk_pct = min(risk_pct, remaining_daily_risk)
        
        risk_amount = self.account_balance * actual_risk_pct
        sl_distance = abs(entry_price - stop_loss)
        
        if sl_distance == 0:
            logger.error("❌ Stop loss distance is zero")
            return self._zero_risk_metrics()
        
        # XAUUSD: 1 lot = 100oz
        position_oz = risk_amount / sl_distance
        position_lots = position_oz / 100
        
        # Apply limits
        position_lots = max(0.01, min(1.0, position_lots))
        
        # Recalculate actual risk with limited position size
        actual_risk = position_lots * 100 * sl_distance
        
        return RiskMetrics(
            position_size=position_lots,
            max_loss_usd=actual_risk,
            max_profit_usd=0,  # Will be calculated with TPs
            risk_reward_ratios=[],
            portfolio_risk_pct=(actual_risk / self.account_balance) * 100,
            margin_required=entry_price * position_lots * 100 * 0.01  # 1% margin
        )
    
    def calculate_take_profit_metrics(self, entry: float, stop_loss: float, 
                                    take_profits: List[float], position_size: float) -> Dict[str, Any]:
        """Calculate TP metrics and R:R ratios"""
        
        sl_distance = abs(entry - stop_loss)
        risk_amount = position_size * 100 * sl_distance
        
        tp_metrics = []
        for i, tp in enumerate(take_profits, 1):
            tp_distance = abs(tp - entry)
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            profit_amount = position_size * 100 * tp_distance
            
            tp_metrics.append({
                'level': i,
                'price': tp,
                'distance': tp_distance,
                'rr_ratio': rr_ratio,
                'profit_usd': profit_amount
            })
        
        return {
            'risk_amount': risk_amount,
            'tp_metrics': tp_metrics,
            'average_rr': np.mean([tp['rr_ratio'] for tp in tp_metrics]),
            'best_rr': max([tp['rr_ratio'] for tp in tp_metrics]) if tp_metrics else 0
        }
    
    def validate_trade_risk(self, trade_risk: TradeRisk) -> Tuple[bool, List[str]]:
        """Validate if trade meets risk criteria"""
        
        issues = []
        
        # Check daily risk limit
        if self.daily_risk_used + trade_risk.risk_pct > self.max_daily_risk:
            issues.append(f"Exceeds daily risk limit ({trade_risk.risk_pct:.1%} + {self.daily_risk_used:.1%} > {self.max_daily_risk:.1%})")
        
        # Check per-trade risk limit
        if trade_risk.risk_pct > self.max_risk_per_trade:
            issues.append(f"Exceeds per-trade risk limit ({trade_risk.risk_pct:.1%} > {self.max_risk_per_trade:.1%})")
        
        # Check minimum R:R ratio
        if trade_risk.take_profits:
            sl_distance = abs(trade_risk.entry_price - trade_risk.stop_loss)
            min_tp_distance = abs(trade_risk.take_profits[0] - trade_risk.entry_price)
            min_rr = min_tp_distance / sl_distance if sl_distance > 0 else 0
            
            if min_rr < 0.8:  # Minimum 0.8:1 R:R
                issues.append(f"Poor risk:reward ratio ({min_rr:.1f}:1 < 0.8:1)")
        
        # Check position size limits
        if trade_risk.position_size < 0.01:
            issues.append("Position size too small (<0.01 lots)")
        elif trade_risk.position_size > 1.0:
            issues.append("Position size too large (>1.0 lots)")
        
        # Check maximum risk amount
        if trade_risk.risk_amount > 500:
            issues.append(f"Risk amount too high (${trade_risk.risk_amount:.2f} > $500)")
        
        return len(issues) == 0, issues
    
    def add_position(self, trade_risk: TradeRisk):
        """Add position to tracking"""
        self.open_positions.append(trade_risk)
        self.daily_risk_used += trade_risk.risk_pct
        logger.info(f"📊 Position added: {trade_risk.risk_pct:.1%} risk, {self.daily_risk_used:.1%} daily total")
    
    def close_position(self, symbol: str, pnl_usd: float):
        """Close position and update risk tracking"""
        for i, pos in enumerate(self.open_positions):
            if pos.symbol == symbol:
                self.open_positions.pop(i)
                logger.info(f"📊 Position closed: {symbol}, P&L: ${pnl_usd:.2f}")
                break
    
    def get_portfolio_risk(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics"""
        
        total_risk = sum(pos.risk_amount for pos in self.open_positions)
        total_exposure = sum(pos.position_size * pos.entry_price * 100 for pos in self.open_positions)
        
        return {
            'open_positions': len(self.open_positions),
            'total_risk_usd': total_risk,
            'total_risk_pct': total_risk / self.account_balance * 100,
            'daily_risk_used': self.daily_risk_used * 100,
            'daily_risk_remaining': (self.max_daily_risk - self.daily_risk_used) * 100,
            'total_exposure_usd': total_exposure,
            'margin_used': total_exposure * 0.01  # 1% margin
        }
    
    def _reset_daily_risk_if_needed(self):
        """Reset daily risk tracking if new day"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_date:
            self.daily_risk_used = 0.0
            self.daily_reset_date = current_date
            logger.info("🔄 Daily risk reset")
    
    def _zero_risk_metrics(self) -> RiskMetrics:
        """Return zero risk metrics"""
        return RiskMetrics(0, 0, 0, [], 0, 0)

# Global risk manager
risk_manager = XAUUSDRiskManager()

def calculate_trade_risk(entry: float, stop_loss: float, take_profits: List[float], 
                        direction: str = 'BUY') -> Dict[str, Any]:
    """Calculate complete trade risk metrics"""
    
    # Position sizing
    risk_metrics = risk_manager.calculate_position_size(entry, stop_loss)
    
    # TP metrics
    tp_analysis = risk_manager.calculate_take_profit_metrics(
        entry, stop_loss, take_profits, risk_metrics.position_size
    )
    
    return {
        'position_size': risk_metrics.position_size,
        'risk_amount': tp_analysis['risk_amount'],
        'risk_pct': risk_metrics.portfolio_risk_pct,
        'tp_analysis': tp_analysis,
        'margin_required': risk_metrics.margin_required
    }