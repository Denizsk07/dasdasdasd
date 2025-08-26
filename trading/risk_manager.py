"""XAUUSD Risk Manager - Advanced Position Sizing & Risk Control"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('risk_manager')

@dataclass
class RiskMetrics:
    """Risk metrics for a trade"""
    position_size: float
    max_loss_usd: float
    max_profit_usd: float
    risk_reward_ratios: List[float]
    portfolio_risk_pct: float
    margin_required: float
    pip_value: float
    trade_value: float

@dataclass
class TradeRisk:
    """Individual trade risk information"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    position_size: float
    risk_amount: float
    risk_pct: float
    trade_id: str
    timestamp: datetime

@dataclass
class PortfolioRisk:
    """Portfolio-wide risk metrics"""
    total_exposure_usd: float
    total_risk_usd: float
    daily_risk_used_pct: float
    open_positions: int
    max_drawdown_usd: float
    value_at_risk_95: float
    correlation_risk: float
    margin_utilization_pct: float

class XAUUSDRiskManager:
    """Advanced risk management for XAUUSD trading"""
    
    def __init__(self):
        # Account settings
        self.account_balance = float(os.getenv('ACCOUNT_BALANCE', '10000'))
        self.max_risk_per_trade = settings.trading.risk_percentage / 100
        self.max_daily_risk = 0.06  # 6% max daily risk
        self.max_portfolio_risk = 0.15  # 15% max total portfolio risk
        
        # XAUUSD specific settings
        self.pip_size = 0.1  # $0.1 per pip for XAUUSD
        self.standard_lot_size = 100  # 100 oz per lot
        self.margin_requirement = 0.01  # 1% margin requirement
        self.max_leverage = 100  # Maximum leverage
        
        # Risk tracking
        self.open_positions: List[TradeRisk] = []
        self.daily_risk_used = 0.0
        self.daily_reset_date = datetime.now().date()
        self.risk_events = []
        
        # Load existing positions if any
        self._load_risk_state()
        
        logger.info(f"🛡️ Risk Manager initialized: ${self.account_balance:,.2f} account")
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_pct: Optional[float] = None, 
                              timeframe: str = 'M15') -> RiskMetrics:
        """Calculate optimal position size with comprehensive risk analysis"""
        
        try:
            # Use default risk if not specified
            if risk_pct is None:
                risk_pct = self.max_risk_per_trade
            
            # Reset daily risk if new day
            self._reset_daily_risk_if_needed()
            
            # Check available daily risk
            available_daily_risk = max(0, self.max_daily_risk - self.daily_risk_used)
            effective_risk_pct = min(risk_pct, available_daily_risk)
            
            if effective_risk_pct <= 0:
                logger.warning("⚠️ Daily risk limit reached")
                return self._zero_risk_metrics()
            
            # Calculate stop loss distance in pips
            sl_distance_usd = abs(entry_price - stop_loss)
            sl_distance_pips = sl_distance_usd / self.pip_size
            
            if sl_distance_usd == 0:
                logger.error("❌ Stop loss distance is zero")
                return self._zero_risk_metrics()
            
            # Calculate risk amount
            risk_amount_usd = self.account_balance * effective_risk_pct
            
            # Calculate position size in lots
            # Risk Amount = Position Size (lots) * Lot Size * SL Distance (USD)
            position_size_lots = risk_amount_usd / (self.standard_lot_size * sl_distance_usd)
            
            # Apply position size limits
            min_position = 0.01  # Minimum 0.01 lots
            max_position = self._calculate_max_position_size(entry_price)
            position_size_lots = max(min_position, min(position_size_lots, max_position))
            
            # Recalculate actual risk with limited position size
            actual_risk_usd = position_size_lots * self.standard_lot_size * sl_distance_usd
            actual_risk_pct = (actual_risk_usd / self.account_balance) * 100
            
            # Calculate pip value for this position
            pip_value = position_size_lots * self.standard_lot_size * self.pip_size
            
            # Calculate trade value and margin
            trade_value = position_size_lots * self.standard_lot_size * entry_price
            margin_required = trade_value * self.margin_requirement
            
            # Calculate max profit potential (using first TP)
            max_profit_usd = 0
            if position_size_lots > 0:
                # Assume first TP is 2:1 R:R minimum
                profit_distance = sl_distance_usd * 2
                max_profit_usd = position_size_lots * self.standard_lot_size * profit_distance
            
            return RiskMetrics(
                position_size=position_size_lots,
                max_loss_usd=actual_risk_usd,
                max_profit_usd=max_profit_usd,
                risk_reward_ratios=[2.0, 3.0, 4.0, 6.0],  # Standard R:R ratios
                portfolio_risk_pct=actual_risk_pct,
                margin_required=margin_required,
                pip_value=pip_value,
                trade_value=trade_value
            )
            
        except Exception as e:
            logger.error(f"❌ Position size calculation failed: {e}")
            return self._zero_risk_metrics()
    
    def calculate_take_profit_levels(self, entry_price: float, stop_loss: float, 
                                   direction: str, position_size: float) -> Dict[str, Any]:
        """Calculate optimal take profit levels with risk-reward analysis"""
        
        try:
            sl_distance = abs(entry_price - stop_loss)
            
            # Standard R:R ratios for XAUUSD
            rr_ratios = [1.5, 2.5, 4.0, 6.0]  # Conservative to aggressive
            
            tp_levels = []
            total_profit_potential = 0
            
            for i, rr in enumerate(rr_ratios, 1):
                if direction.upper() == 'BUY':
                    tp_price = entry_price + (sl_distance * rr)
                else:
                    tp_price = entry_price - (sl_distance * rr)
                
                # Calculate profit for this TP level (assuming 25% position closure each time)
                partial_position = position_size * 0.25
                profit_usd = partial_position * self.standard_lot_size * (sl_distance * rr)
                total_profit_potential += profit_usd
                
                tp_levels.append({
                    'level': f'TP{i}',
                    'price': tp_price,
                    'rr_ratio': rr,
                    'profit_usd': profit_usd,
                    'position_percent': 25,  # Close 25% at each TP
                    'cumulative_profit': total_profit_potential
                })
            
            # Risk metrics
            max_loss = position_size * self.standard_lot_size * sl_distance
            max_gain = total_profit_potential
            
            return {
                'tp_levels': tp_levels,
                'risk_reward_summary': {
                    'max_loss_usd': max_loss,
                    'max_profit_usd': max_gain,
                    'profit_factor': max_gain / max_loss if max_loss > 0 else 0,
                    'average_rr': np.mean(rr_ratios)
                },
                'position_management': {
                    'total_position_size': position_size,
                    'partial_closure_strategy': '25% at each TP level',
                    'remaining_position': '0% after TP4'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ TP calculation failed: {e}")
            return {'error': str(e)}
    
    def validate_trade_risk(self, entry_price: float, stop_loss: float, 
                           take_profits: List[float], direction: str,
                           position_size: float = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Comprehensive trade risk validation"""
        
        issues = []
        risk_analysis = {}
        
        try:
            # Calculate position size if not provided
            if position_size is None:
                risk_metrics = self.calculate_position_size(entry_price, stop_loss)
                position_size = risk_metrics.position_size
                risk_analysis['calculated_position_size'] = position_size
            
            # 1. Basic price validation
            if entry_price <= 0 or stop_loss <= 0:
                issues.append("Invalid entry or stop loss price")
            
            # 2. Stop loss direction validation
            if direction.upper() == 'BUY' and stop_loss >= entry_price:
                issues.append("Buy trade: Stop loss must be below entry price")
            elif direction.upper() == 'SELL' and stop_loss <= entry_price:
                issues.append("Sell trade: Stop loss must be above entry price")
            
            # 3. Take profit validation
            for i, tp in enumerate(take_profits, 1):
                if direction.upper() == 'BUY' and tp <= entry_price:
                    issues.append(f"TP{i}: Take profit must be above entry for BUY trade")
                elif direction.upper() == 'SELL' and tp >= entry_price:
                    issues.append(f"TP{i}: Take profit must be below entry for SELL trade")
            
            # 4. Risk amount validation
            sl_distance = abs(entry_price - stop_loss)
            trade_risk_usd = position_size * self.standard_lot_size * sl_distance
            trade_risk_pct = (trade_risk_usd / self.account_balance) * 100
            
            risk_analysis['trade_risk_usd'] = trade_risk_usd
            risk_analysis['trade_risk_pct'] = trade_risk_pct
            
            if trade_risk_pct > self.max_risk_per_trade * 100:
                issues.append(f"Risk too high: {trade_risk_pct:.2f}% > {self.max_risk_per_trade*100:.1f}%")
            
            # 5. Daily risk limit validation
            self._reset_daily_risk_if_needed()
            if self.daily_risk_used + (trade_risk_pct/100) > self.max_daily_risk:
                issues.append(f"Exceeds daily risk limit: {((self.daily_risk_used + trade_risk_pct/100)*100):.1f}% > {self.max_daily_risk*100:.1f}%")
            
            # 6. Position size validation
            if position_size < 0.01:
                issues.append("Position size too small (minimum 0.01 lots)")
            elif position_size > 5.0:  # Maximum 5 lots for XAUUSD
                issues.append("Position size too large (maximum 5.0 lots)")
            
            # 7. Margin validation
            trade_value = position_size * self.standard_lot_size * entry_price
            margin_required = trade_value * self.margin_requirement
            available_margin = self.account_balance * 0.8  # Use max 80% for margin
            
            risk_analysis['margin_required'] = margin_required
            risk_analysis['available_margin'] = available_margin
            
            if margin_required > available_margin:
                issues.append(f"Insufficient margin: ${margin_required:,.2f} required, ${available_margin:,.2f} available")
            
            # 8. Portfolio risk validation
            total_portfolio_risk = self._calculate_total_portfolio_risk() + trade_risk_usd
            portfolio_risk_pct = (total_portfolio_risk / self.account_balance) * 100
            
            risk_analysis['portfolio_risk_pct'] = portfolio_risk_pct
            
            if portfolio_risk_pct > self.max_portfolio_risk * 100:
                issues.append(f"Portfolio risk too high: {portfolio_risk_pct:.1f}% > {self.max_portfolio_risk*100:.1f}%")
            
            # 9. Risk-Reward validation
            min_tp_distance = abs(take_profits[0] - entry_price) if take_profits else 0
            rr_ratio = min_tp_distance / sl_distance if sl_distance > 0 else 0
            
            risk_analysis['first_tp_rr_ratio'] = rr_ratio
            
            if rr_ratio < 1.2:  # Minimum 1.2:1 R:R
                issues.append(f"Poor risk:reward ratio: {rr_ratio:.1f}:1 < 1.2:1")
            
            # 10. Market volatility adjustment
            volatility_adjustment = self._get_volatility_adjustment()
            risk_analysis['volatility_adjustment'] = volatility_adjustment
            
            if volatility_adjustment > 1.5:
                issues.append("High volatility detected - consider reducing position size")
            
            # 11. Correlation risk (if multiple positions)
            if len(self.open_positions) > 0:
                correlation_risk = self._calculate_correlation_risk()
                risk_analysis['correlation_risk'] = correlation_risk
                
                if correlation_risk > 0.7:
                    issues.append("High correlation risk with existing positions")
            
            # Generate risk score
            risk_score = self._calculate_risk_score(risk_analysis, len(issues))
            risk_analysis['overall_risk_score'] = risk_score
            
            is_valid = len(issues) == 0
            
            return is_valid, issues, risk_analysis
            
        except Exception as e:
            logger.error(f"❌ Risk validation failed: {e}")
            return False, [f"Validation error: {str(e)}"], {}
    
    def add_position(self, entry_price: float, stop_loss: float, take_profits: List[float],
                    direction: str, position_size: float, signal_id: str) -> bool:
        """Add new position to risk tracking"""
        
        try:
            # Validate the trade first
            is_valid, issues, risk_analysis = self.validate_trade_risk(
                entry_price, stop_loss, take_profits, direction, position_size
            )
            
            if not is_valid:
                logger.warning(f"⚠️ Position rejected: {', '.join(issues)}")
                return False
            
            # Calculate risk amount
            sl_distance = abs(entry_price - stop_loss)
            risk_amount = position_size * self.standard_lot_size * sl_distance
            risk_pct = (risk_amount / self.account_balance)
            
            # Create trade risk object
            trade_risk = TradeRisk(
                symbol='XAUUSD',
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profits=take_profits,
                position_size=position_size,
                risk_amount=risk_amount,
                risk_pct=risk_pct,
                trade_id=signal_id,
                timestamp=datetime.now()
            )
            
            # Add to tracking
            self.open_positions.append(trade_risk)
            self.daily_risk_used += risk_pct
            
            # Log the addition
            logger.info(f"📊 Position added: {direction} {position_size:.3f} lots, Risk: ${risk_amount:.2f} ({risk_pct:.1%})")
            logger.info(f"📊 Daily risk used: {self.daily_risk_used:.1%} / {self.max_daily_risk:.1%}")
            
            # Save state
            self._save_risk_state()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add position: {e}")
            return False
    
    def close_position(self, trade_id: str, exit_price: float, exit_type: str = 'manual') -> bool:
        """Close position and update risk tracking"""
        
        try:
            # Find the position
            position_index = None
            for i, pos in enumerate(self.open_positions):
                if pos.trade_id == trade_id:
                    position_index = i
                    break
            
            if position_index is None:
                logger.warning(f"⚠️ Position not found: {trade_id}")
                return False
            
            position = self.open_positions[position_index]
            
            # Calculate P&L
            if position.direction.upper() == 'BUY':
                pnl_pips = (exit_price - position.entry_price) / self.pip_size
            else:
                pnl_pips = (position.entry_price - exit_price) / self.pip_size
            
            pnl_usd = position.position_size * self.standard_lot_size * (pnl_pips * self.pip_size)
            
            # Remove from open positions
            self.open_positions.pop(position_index)
            
            # Update daily risk (risk is freed up when position is closed)
            # Note: We don't reduce daily_risk_used as it represents risk taken today
            
            # Log the closure
            logger.info(f"📊 Position closed: {trade_id}, P&L: {pnl_pips:+.1f} pips (${pnl_usd:+.2f})")
            logger.info(f"📊 Open positions remaining: {len(self.open_positions)}")
            
            # Save state
            self._save_risk_state()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
            return False
    
    def get_portfolio_risk(self) -> PortfolioRisk:
        """Get current portfolio risk metrics"""
        
        try:
            total_exposure = sum(
                pos.position_size * self.standard_lot_size * pos.entry_price
                for pos in self.open_positions
            )
            
            total_risk = sum(pos.risk_amount for pos in self.open_positions)
            
            # Calculate VaR (95% confidence level)
            var_95 = self._calculate_value_at_risk()
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk()
            
            # Margin utilization
            total_margin_required = sum(
                pos.position_size * self.standard_lot_size * pos.entry_price * self.margin_requirement
                for pos in self.open_positions
            )
            margin_utilization = (total_margin_required / self.account_balance) * 100
            
            return PortfolioRisk(
                total_exposure_usd=total_exposure,
                total_risk_usd=total_risk,
                daily_risk_used_pct=self.daily_risk_used * 100,
                open_positions=len(self.open_positions),
                max_drawdown_usd=total_risk,  # Worst case scenario
                value_at_risk_95=var_95,
                correlation_risk=correlation_risk,
                margin_utilization_pct=margin_utilization
            )
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk calculation failed: {e}")
            return PortfolioRisk(0, 0, 0, 0, 0, 0, 0, 0)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        try:
            portfolio_risk = self.get_portfolio_risk()
            
            # Risk utilization
            daily_risk_remaining = max(0, self.max_daily_risk - self.daily_risk_used)
            portfolio_risk_remaining = max(0, self.max_portfolio_risk - (portfolio_risk.total_risk_usd / self.account_balance))
            
            # Position analysis
            position_analysis = []
            for pos in self.open_positions:
                position_analysis.append({
                    'trade_id': pos.trade_id,
                    'direction': pos.direction,
                    'position_size': pos.position_size,
                    'risk_usd': pos.risk_amount,
                    'risk_pct': pos.risk_pct * 100,
                    'entry_price': pos.entry_price,
                    'stop_loss': pos.stop_loss,
                    'hold_time_minutes': int((datetime.now() - pos.timestamp).total_seconds() / 60)
                })
            
            # Risk limits status
            risk_limits = {
                'daily_risk': {
                    'used': self.daily_risk_used * 100,
                    'limit': self.max_daily_risk * 100,
                    'remaining': daily_risk_remaining * 100,
                    'status': 'OK' if daily_risk_remaining > 0.01 else 'LIMIT_REACHED'
                },
                'portfolio_risk': {
                    'used': (portfolio_risk.total_risk_usd / self.account_balance) * 100,
                    'limit': self.max_portfolio_risk * 100,
                    'remaining': portfolio_risk_remaining * 100,
                    'status': 'OK' if portfolio_risk_remaining > 0.01 else 'LIMIT_REACHED'
                },
                'position_count': {
                    'current': len(self.open_positions),
                    'maximum': 5,  # Max 5 concurrent positions
                    'status': 'OK' if len(self.open_positions) < 5 else 'LIMIT_REACHED'
                }
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'account_balance': self.account_balance,
                'portfolio_risk': portfolio_risk,
                'risk_limits': risk_limits,
                'open_positions': position_analysis,
                'risk_score': self._calculate_overall_risk_score(),
                'recommendations': self._generate_risk_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Risk report generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_max_position_size(self, entry_price: float) -> float:
        """Calculate maximum allowed position size"""
        
        # Based on account balance and leverage limits
        max_trade_value = self.account_balance * self.max_leverage
        max_lots_by_leverage = max_trade_value / (self.standard_lot_size * entry_price)
        
        # Based on risk management (max 5% risk per trade with 50 pip SL)
        max_risk_amount = self.account_balance * 0.05
        estimated_sl_distance = entry_price * 0.025  # Assume 2.5% SL
        max_lots_by_risk = max_risk_amount / (self.standard_lot_size * estimated_sl_distance)
        
        # Take the smaller of the two
        max_position = min(max_lots_by_leverage, max_lots_by_risk)
        
        # Hard limit for XAUUSD
        return min(max_position, 5.0)
    
    def _calculate_total_portfolio_risk(self) -> float:
        """Calculate total portfolio risk in USD"""
        return sum(pos.risk_amount for pos in self.open_positions)
    
    def _get_volatility_adjustment(self) -> float:
        """Get volatility adjustment factor"""
        # Simplified volatility calculation
        # In practice, this would use actual market volatility data
        base_volatility = 1.0
        
        # Time-based adjustment (higher volatility during certain hours)
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 16:  # London session
            volatility_multiplier = 1.3
        elif 13 <= current_hour <= 21:  # NY session
            volatility_multiplier = 1.2
        else:  # Asian session
            volatility_multiplier = 1.0
        
        return base_volatility * volatility_multiplier
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk between positions"""
        if len(self.open_positions) <= 1:
            return 0.0
        
        # For XAUUSD, all positions are perfectly correlated
        # Risk increases with number of positions in same direction
        long_positions = sum(1 for pos in self.open_positions if pos.direction.upper() == 'BUY')
        short_positions = len(self.open_positions) - long_positions
        
        # Higher correlation risk if all positions are in same direction
        if long_positions == 0 or short_positions == 0:
            return min(1.0, len(self.open_positions) * 0.2)
        else:
            # Some diversification if positions are in opposite directions
            return min(1.0, abs(long_positions - short_positions) * 0.1)
    
    def _calculate_value_at_risk(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level"""
        if not self.open_positions:
            return 0.0
        
        # Simplified VaR calculation
        total_risk = sum(pos.risk_amount for pos in self.open_positions)
        
        # Assume normal distribution with 2% daily volatility for XAUUSD
        from scipy import stats
        var_multiplier = abs(stats.norm.ppf(1 - confidence))  # ~1.645 for 95%
        daily_volatility = 0.02
        
        var_95 = total_risk * var_multiplier * daily_volatility
        
        return var_95
    
    def _calculate_risk_score(self, risk_analysis: Dict[str, Any], issue_count: int) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        
        base_score = issue_count * 20  # 20 points per issue
        
        # Adjust based on risk metrics
        risk_pct = risk_analysis.get('trade_risk_pct', 0)
        if risk_pct > 3.0:  # Above 3% risk
            base_score += 15
        elif risk_pct > 2.0:
            base_score += 10
        
        # Portfolio risk adjustment
        portfolio_risk = risk_analysis.get('portfolio_risk_pct', 0)
        if portfolio_risk > 12:  # Above 12% portfolio risk
            base_score += 20
        elif portfolio_risk > 8:
            base_score += 10
        
        # R:R ratio adjustment
        rr_ratio = risk_analysis.get('first_tp_rr_ratio', 2.0)
        if rr_ratio < 1.5:
            base_score += 15
        elif rr_ratio < 2.0:
            base_score += 5
        
        return min(100, max(0, base_score))
    
    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall portfolio risk score"""
        
        portfolio_risk = self.get_portfolio_risk()
        
        score = 0
        
        # Daily risk utilization
        daily_risk_pct = portfolio_risk.daily_risk_used_pct
        if daily_risk_pct > 80:
            score += 30
        elif daily_risk_pct > 60:
            score += 20
        elif daily_risk_pct > 40:
            score += 10
        
        # Number of open positions
        if portfolio_risk.open_positions > 4:
            score += 25
        elif portfolio_risk.open_positions > 3:
            score += 15
        elif portfolio_risk.open_positions > 2:
            score += 10
        
        # Correlation risk
        if portfolio_risk.correlation_risk > 0.8:
            score += 20
        elif portfolio_risk.correlation_risk > 0.6:
            score += 15
        elif portfolio_risk.correlation_risk > 0.4:
            score += 10
        
        # Margin utilization
        if portfolio_risk.margin_utilization_pct > 70:
            score += 20
        elif portfolio_risk.margin_utilization_pct > 50:
            score += 10
        
        return min(100, score)
    
    def _generate_risk_recommendations(self) -> List[str]:
        """Generate risk management recommendations"""
        
        recommendations = []
        portfolio_risk = self.get_portfolio_risk()
        
        # Daily risk recommendations
        if portfolio_risk.daily_risk_used_pct > 80:
            recommendations.append("Daily risk limit nearly reached - consider reducing position sizes")
        
        # Position count recommendations
        if portfolio_risk.open_positions > 3:
            recommendations.append("Multiple open positions - monitor correlation risk")
        
        # Correlation recommendations
        if portfolio_risk.correlation_risk > 0.7:
            recommendations.append("High correlation between positions - consider closing some trades")
        
        # Margin recommendations
        if portfolio_risk.margin_utilization_pct > 60:
            recommendations.append("High margin utilization - ensure sufficient account buffer")
        
        # VaR recommendations
        if portfolio_risk.value_at_risk_95 > self.account_balance * 0.1:
            recommendations.append("Value at Risk exceeds 10% of account - consider risk reduction")
        
        return recommendations
    
    def _reset_daily_risk_if_needed(self):
        """Reset daily risk tracking if new day"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_date:
            self.daily_risk_used = 0.0
            self.daily_reset_date = current_date
            logger.info("🔄 Daily risk limits reset")
    
    def _save_risk_state(self):
        """Save current risk state to file"""
        try:
            state_data = {
                'open_positions': [
                    {
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'stop_loss': pos.stop_loss,
                        'take_profits': pos.take_profits,
                        'position_size': pos.position_size,
                        'risk_amount': pos.risk_amount,
                        'risk_pct': pos.risk_pct,
                        'trade_id': pos.trade_id,
                        'timestamp': pos.timestamp.isoformat()
                    }
                    for pos in self.open_positions
                ],
                'daily_risk_used': self.daily_risk_used,
                'daily_reset_date': self.daily_reset_date.isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            risk_state_file = settings.storage.base_dir / 'risk_state.json'
            with open(risk_state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug("💾 Risk state saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save risk state: {e}")
    
    def _load_risk_state(self):
        """Load risk state from file"""
        try:
            risk_state_file = settings.storage.base_dir / 'risk_state.json'
            if not risk_state_file.exists():
                return
            
            with open(risk_state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Load open positions
            self.open_positions = []
            for pos_data in state_data.get('open_positions', []):
                trade_risk = TradeRisk(
                    symbol=pos_data['symbol'],
                    direction=pos_data['direction'],
                    entry_price=pos_data['entry_price'],
                    stop_loss=pos_data['stop_loss'],
                    take_profits=pos_data['take_profits'],
                    position_size=pos_data['position_size'],
                    risk_amount=pos_data['risk_amount'],
                    risk_pct=pos_data['risk_pct'],
                    trade_id=pos_data['trade_id'],
                    timestamp=datetime.fromisoformat(pos_data['timestamp'])
                )
                self.open_positions.append(trade_risk)
            
            # Load daily risk tracking
            self.daily_risk_used = state_data.get('daily_risk_used', 0.0)
            self.daily_reset_date = datetime.fromisoformat(state_data['daily_reset_date']).date()
            
            logger.info(f"📁 Risk state loaded: {len(self.open_positions)} open positions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load risk state: {e}")
    
    def _zero_risk_metrics(self) -> RiskMetrics:
        """Return zero risk metrics"""
        return RiskMetrics(0, 0, 0, [], 0, 0, 0, 0)

# Enhanced helper functions for specific risk calculations
def calculate_pip_value(position_size: float, symbol: str = 'XAUUSD') -> float:
    """Calculate pip value for given position size"""
    if symbol == 'XAUUSD':
        return position_size * 100 * 0.1  # $0.1 per pip per lot
    else:
        return position_size * 100000 * 0.0001  # Standard forex

def calculate_margin_requirement(position_size: float, entry_price: float, 
                               margin_rate: float = 0.01) -> float:
    """Calculate margin requirement for position"""
    trade_value = position_size * 100 * entry_price  # For XAUUSD
    return trade_value * margin_rate

def calculate_position_heat(current_price: float, entry_price: float, 
                          stop_loss: float, direction: str) -> float:
    """Calculate how much of the risk is currently realized (0-1)"""
    try:
        if direction.upper() == 'BUY':
            if current_price <= stop_loss:
                return 1.0  # Full loss
            elif current_price >= entry_price:
                return 0.0  # No loss (in profit)
            else:
                # Calculate partial loss
                total_risk = entry_price - stop_loss
                current_loss = entry_price - current_price
                return current_loss / total_risk
        else:  # SELL
            if current_price >= stop_loss:
                return 1.0  # Full loss
            elif current_price <= entry_price:
                return 0.0  # No loss (in profit)
            else:
                # Calculate partial loss
                total_risk = stop_loss - entry_price
                current_loss = current_price - entry_price
                return current_loss / total_risk
    except:
        return 0.0

# Global risk manager instance
risk_manager = XAUUSDRiskManager()

# Convenience functions
def calculate_trade_risk(entry: float, stop_loss: float, take_profits: List[float], 
                        direction: str = 'BUY') -> Dict[str, Any]:
    """Calculate complete trade risk metrics"""
    
    # Position sizing
    risk_metrics = risk_manager.calculate_position_size(entry, stop_loss)
    
    # TP analysis
    tp_analysis = risk_manager.calculate_take_profit_levels(
        entry, stop_loss, direction, risk_metrics.position_size
    )
    
    # Risk validation
    is_valid, issues, risk_analysis = risk_manager.validate_trade_risk(
        entry, stop_loss, take_profits, direction, risk_metrics.position_size
    )
    
    return {
        'position_size': risk_metrics.position_size,
        'risk_amount': risk_metrics.max_loss_usd,
        'risk_pct': risk_metrics.portfolio_risk_pct,
        'tp_analysis': tp_analysis,
        'margin_required': risk_metrics.margin_required,
        'trade_value': risk_metrics.trade_value,
        'pip_value': risk_metrics.pip_value,
        'is_valid': is_valid,
        'issues': issues,
        'risk_score': risk_analysis.get('overall_risk_score', 0)
    }

def get_portfolio_risk() -> Dict[str, Any]:
    """Get current portfolio risk metrics"""
    portfolio_risk = risk_manager.get_portfolio_risk()
    return {
        'total_exposure_usd': portfolio_risk.total_exposure_usd,
        'total_risk_usd': portfolio_risk.total_risk_usd,
        'daily_risk_used_pct': portfolio_risk.daily_risk_used_pct,
        'open_positions': portfolio_risk.open_positions,
        'max_drawdown_usd': portfolio_risk.max_drawdown_usd,
        'value_at_risk_95': portfolio_risk.value_at_risk_95,
        'correlation_risk': portfolio_risk.correlation_risk,
        'margin_utilization_pct': portfolio_risk.margin_utilization_pct
    }

def validate_trade(entry: float, stop_loss: float, take_profits: List[float], 
                  direction: str, position_size: float = None) -> Tuple[bool, List[str]]:
    """Validate trade against risk management rules"""
    is_valid, issues, _ = risk_manager.validate_trade_risk(
        entry, stop_loss, take_profits, direction, position_size
    )
    return is_valid, issues

def add_position(entry: float, stop_loss: float, take_profits: List[float],
                direction: str, position_size: float, signal_id: str) -> bool:
    """Add new position to risk tracking"""
    return risk_manager.add_position(entry, stop_loss, take_profits, direction, position_size, signal_id)

def close_position(trade_id: str, exit_price: float, exit_type: str = 'manual') -> bool:
    """Close position and update risk tracking"""
    return risk_manager.close_position(trade_id, exit_price, exit_type)

def get_risk_report() -> Dict[str, Any]:
    """Get comprehensive risk report"""
    return risk_manager.get_risk_report()

def get_available_risk() -> Dict[str, float]:
    """Get available risk capacity"""
    risk_manager._reset_daily_risk_if_needed()
    portfolio_risk = risk_manager.get_portfolio_risk()
    
    daily_risk_remaining = max(0, risk_manager.max_daily_risk - risk_manager.daily_risk_used)
    portfolio_risk_remaining = max(0, risk_manager.max_portfolio_risk - (portfolio_risk.total_risk_usd / risk_manager.account_balance))
    
    return {
        'daily_risk_remaining_pct': daily_risk_remaining * 100,
        'portfolio_risk_remaining_pct': portfolio_risk_remaining * 100,
        'daily_risk_remaining_usd': daily_risk_remaining * risk_manager.account_balance,
        'portfolio_risk_remaining_usd': portfolio_risk_remaining * risk_manager.account_balance,
        'can_trade': daily_risk_remaining > 0.005 and portfolio_risk_remaining > 0.005  # At least 0.5% available
    }

# Import required modules for VaR calculation
try:
    from scipy import stats
except ImportError:
    # Fallback if scipy not available
    class MockStats:
        class norm:
            @staticmethod
            def ppf(x):
                # Approximate values for common confidence levels
                if abs(x - 0.95) < 0.01:
                    return 1.645
                elif abs(x - 0.99) < 0.01:
                    return 2.326
                elif abs(x - 0.975) < 0.01:
                    return 1.96
                else:
                    return 1.645  # Default to 95%
    
    stats = MockStats()

import os  # Add missing import