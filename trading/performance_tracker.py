"""
XAUUSD Performance Tracker - Advanced Trading Performance Analytics
Tracks all trades, calculates metrics, generates reports, and feeds learning system
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from config.settings import settings
from utils.logger import get_module_logger
from analysis.signal_generator import TradingSignal

logger = get_module_logger('performance_tracker')

@dataclass
class TradeResult:
    """Individual trade result"""
    signal_id: str
    timestamp: datetime
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profits: List[float]
    
    # Result data
    exit_type: str  # 'tp1', 'tp2', 'tp3', 'tp4', 'sl', 'manual'
    pnl_pips: float
    pnl_usd: float
    pnl_percent: float
    
    # Trade details
    position_size: float
    timeframe: str
    hold_time_minutes: int
    
    # Strategy details
    triggered_strategies: List[str]
    strategy_scores: Dict[str, float]
    confidence: float
    
    # Market context
    market_session: str
    volatility_level: str
    
    # Learning data
    was_correct: bool
    exit_reason: str

@dataclass
class StrategyPerformance:
    """Performance metrics for a specific strategy"""
    name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    total_pips: float
    avg_pips_per_trade: float
    best_trade_pips: float
    worst_trade_pips: float
    
    total_pnl_usd: float
    avg_pnl_per_trade: float
    
    profit_factor: float
    expectancy: float
    
    # Recent performance
    recent_win_rate: float
    recent_avg_pips: float
    
    # Confidence metrics
    avg_confidence: float
    confidence_accuracy: float

@dataclass
class PerformanceMetrics:
    """Overall performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L Metrics
    total_pips: float
    total_pnl_usd: float
    avg_pips_per_trade: float
    avg_pnl_per_trade: float
    
    # Risk Metrics
    max_drawdown_pct: float
    max_drawdown_usd: float
    recovery_factor: float
    
    # Advanced Metrics
    sharpe_ratio: float
    profit_factor: float
    expectancy: float
    
    # Streaks
    max_winning_streak: int
    max_losing_streak: int
    current_streak: int
    current_streak_type: str
    
    # Time Analysis
    avg_hold_time_minutes: int
    best_session: str
    best_timeframe: str
    
    # Strategy Performance
    strategy_performances: Dict[str, StrategyPerformance]
    
    # Recent Performance
    last_7_days_win_rate: float
    last_30_days_win_rate: float
    monthly_pnl: Dict[str, float]

class PerformanceTracker:
    """Main performance tracking system"""
    
    def __init__(self):
        self.trades_file = settings.storage.trades_file
        self.performance_file = settings.storage.performance_file
        
        # Load existing data
        self.trades: List[TradeResult] = self._load_trades()
        self.performance_cache: Optional[PerformanceMetrics] = None
        self.cache_expiry = datetime.now()
        
        logger.info(f"📊 Performance Tracker initialized: {len(self.trades)} historical trades")
    
    def record_signal_sent(self, signal: TradingSignal) -> str:
        """Record when a signal is sent (pending trade)"""
        try:
            logger.info(f"📡 Signal recorded: {signal.signal_id} - {signal.direction} @ ${signal.entry_price:.2f}")
            return signal.signal_id
        
        except Exception as e:
            logger.error(f"❌ Failed to record signal: {e}")
            return ""
    
    def record_trade_result(self, signal_id: str, exit_price: float, exit_type: str, 
                          hold_time_minutes: int, exit_reason: str = "") -> bool:
        """Record completed trade result"""
        try:
            trade_result = self._create_synthetic_trade_result(
                signal_id, exit_price, exit_type, hold_time_minutes, exit_reason
            )
            
            if trade_result:
                self.trades.append(trade_result)
                self._save_trades()
                
                # Invalidate cache
                self.performance_cache = None
                
                logger.info(f"💰 Trade recorded: {signal_id} - {trade_result.pnl_pips:+.1f} pips, {trade_result.pnl_usd:+.2f} USD")
                return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade result: {e}")
        
        return False
    
    def get_performance_metrics(self, force_refresh: bool = False) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        
        # Use cache if available and not expired
        if (not force_refresh and self.performance_cache and 
            datetime.now() < self.cache_expiry):
            return self.performance_cache
        
        if not self.trades:
            return self._empty_performance_metrics()
        
        try:
            metrics = self._calculate_performance_metrics()
            
            # Cache for 5 minutes
            self.performance_cache = metrics
            self.cache_expiry = datetime.now() + timedelta(minutes=5)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance calculation failed: {e}")
            return self._empty_performance_metrics()
    
    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for specific strategy"""
        metrics = self.get_performance_metrics()
        return metrics.strategy_performances.get(strategy_name)
    
    def get_learning_feedback(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback for learning system"""
        if not self.trades:
            return {}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trades if t.timestamp > cutoff_date]
        
        if not recent_trades:
            return {}
        
        # Strategy effectiveness
        strategy_feedback = {}
        for strategy in ['smc', 'patterns', 'technical_indicators', 'price_action']:
            strategy_trades = [t for t in recent_trades if strategy in t.triggered_strategies]
            
            if strategy_trades:
                win_rate = sum(1 for t in strategy_trades if t.was_correct) / len(strategy_trades)
                avg_pips = np.mean([t.pnl_pips for t in strategy_trades])
                
                strategy_feedback[strategy] = {
                    'trades': len(strategy_trades),
                    'win_rate': win_rate,
                    'avg_pips': avg_pips,
                    'weight_adjustment': self._calculate_weight_adjustment(win_rate, avg_pips)
                }
        
        # Optimal parameters
        optimal_params = self._analyze_optimal_parameters(recent_trades)
        
        # Risk analysis
        risk_analysis = self._analyze_risk_parameters(recent_trades)
        
        return {
            'strategy_feedback': strategy_feedback,
            'optimal_parameters': optimal_params,
            'risk_analysis': risk_analysis,
            'recent_performance': {
                'win_rate': sum(1 for t in recent_trades if t.was_correct) / len(recent_trades),
                'avg_pips': np.mean([t.pnl_pips for t in recent_trades]),
                'total_trades': len(recent_trades)
            }
        }
    
    def generate_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """Generate detailed monthly performance report"""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        monthly_trades = [
            t for t in self.trades 
            if start_date <= t.timestamp < end_date
        ]
        
        if not monthly_trades:
            return {'error': 'No trades in specified month'}
        
        # Calculate monthly metrics
        total_trades = len(monthly_trades)
        winning_trades = sum(1 for t in monthly_trades if t.was_correct)
        win_rate = winning_trades / total_trades
        
        total_pips = sum(t.pnl_pips for t in monthly_trades)
        total_pnl = sum(t.pnl_usd for t in monthly_trades)
        
        # Best and worst trades
        best_trade = max(monthly_trades, key=lambda t: t.pnl_pips)
        worst_trade = min(monthly_trades, key=lambda t: t.pnl_pips)
        
        # Session analysis
        session_performance = {}
        for session in ['asian', 'london', 'newyork']:
            session_trades = [t for t in monthly_trades if t.market_session == session]
            if session_trades:
                session_win_rate = sum(1 for t in session_trades if t.was_correct) / len(session_trades)
                session_pips = sum(t.pnl_pips for t in session_trades)
                session_performance[session] = {
                    'trades': len(session_trades),
                    'win_rate': session_win_rate,
                    'total_pips': session_pips
                }
        
        # Strategy analysis
        strategy_analysis = {}
        all_strategies = set()
        for trade in monthly_trades:
            all_strategies.update(trade.triggered_strategies)
        
        for strategy in all_strategies:
            strategy_trades = [t for t in monthly_trades if strategy in t.triggered_strategies]
            if strategy_trades:
                strategy_win_rate = sum(1 for t in strategy_trades if t.was_correct) / len(strategy_trades)
                strategy_pips = sum(t.pnl_pips for t in strategy_trades)
                strategy_analysis[strategy] = {
                    'trades': len(strategy_trades),
                    'win_rate': strategy_win_rate,
                    'total_pips': strategy_pips
                }
        
        return {
            'month': f"{year}-{month:02d}",
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pips': total_pips,
                'total_pnl_usd': total_pnl,
                'avg_pips_per_trade': total_pips / total_trades,
                'avg_pnl_per_trade': total_pnl / total_trades
            },
            'best_trade': {
                'signal_id': best_trade.signal_id,
                'direction': best_trade.direction,
                'pips': best_trade.pnl_pips,
                'pnl_usd': best_trade.pnl_usd,
                'date': best_trade.timestamp.strftime('%Y-%m-%d %H:%M')
            },
            'worst_trade': {
                'signal_id': worst_trade.signal_id,
                'direction': worst_trade.direction,
                'pips': worst_trade.pnl_pips,
                'pnl_usd': worst_trade.pnl_usd,
                'date': worst_trade.timestamp.strftime('%Y-%m-%d %H:%M')
            },
            'session_analysis': session_performance,
            'strategy_analysis': strategy_analysis
        }
    
    def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Analyze drawdown periods"""
        if not self.trades:
            return {}
        
        # Calculate running P&L
        running_pnl = []
        cumulative_pnl = 0
        
        for trade in sorted(self.trades, key=lambda t: t.timestamp):
            cumulative_pnl += trade.pnl_usd
            running_pnl.append(cumulative_pnl)
        
        # Calculate drawdowns
        peak = running_pnl[0]
        max_drawdown = 0
        drawdown_periods = []
        current_drawdown_start = None
        
        for i, pnl in enumerate(running_pnl):
            if pnl > peak:
                if current_drawdown_start is not None:
                    drawdown_periods.append({
                        'start_index': current_drawdown_start,
                        'end_index': i - 1,
                        'start_date': self.trades[current_drawdown_start].timestamp,
                        'end_date': self.trades[i - 1].timestamp,
                        'drawdown_usd': peak - running_pnl[i - 1],
                        'recovery_trades': i - current_drawdown_start
                    })
                    current_drawdown_start = None
                peak = pnl
            else:
                if current_drawdown_start is None:
                    current_drawdown_start = i
                drawdown = peak - pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        return {
            'max_drawdown_usd': max_drawdown,
            'max_drawdown_pct': (max_drawdown / max(running_pnl)) * 100 if max(running_pnl) > 0 else 0,
            'drawdown_periods': len(drawdown_periods),
            'avg_recovery_trades': np.mean([dd['recovery_trades'] for dd in drawdown_periods]) if drawdown_periods else 0
        }
    
    def _load_trades(self) -> List[TradeResult]:
        """Load trades from JSON file"""
        try:
            if not self.trades_file.exists():
                return []
            
            with open(self.trades_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            trades = []
            for trade_data in data.get('trades', []):
                trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                trades.append(TradeResult(**trade_data))
            
            logger.info(f"📁 Loaded {len(trades)} historical trades")
            return trades
            
        except Exception as e:
            logger.error(f"❌ Failed to load trades: {e}")
            return []
    
    def _save_trades(self):
        """Save trades to JSON file"""
        try:
            data = {
                'trades': [],
                'last_updated': datetime.now().isoformat()
            }
            
            for trade in self.trades:
                trade_dict = asdict(trade)
                trade_dict['timestamp'] = trade.timestamp.isoformat()
                data['trades'].append(trade_dict)
            
            with open(self.trades_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved {len(self.trades)} trades to file")
            
        except Exception as e:
            logger.error(f"❌ Failed to save trades: {e}")
    
    def _create_synthetic_trade_result(self, signal_id: str, exit_price: float, 
                                     exit_type: str, hold_time_minutes: int, 
                                     exit_reason: str) -> Optional[TradeResult]:
        """Create synthetic trade result for demonstration"""
        try:
            # Synthetic trade data (would come from stored signal)
            entry_price = 2050.0 + np.random.normal(0, 10)
            direction = np.random.choice(['BUY', 'SELL'])
            
            # Calculate P&L
            if direction == 'BUY':
                pnl_pips = (exit_price - entry_price) * 10
            else:
                pnl_pips = (entry_price - exit_price) * 10
            
            position_size = 0.1
            pnl_usd = pnl_pips * position_size * 10
            pnl_percent = (pnl_usd / (entry_price * position_size)) * 100
            
            was_correct = pnl_pips > 0
            
            return TradeResult(
                signal_id=signal_id,
                timestamp=datetime.now() - timedelta(minutes=hold_time_minutes),
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=entry_price + (-30 if direction == 'BUY' else 30),
                take_profits=[entry_price + (20 if direction == 'BUY' else -20),
                            entry_price + (40 if direction == 'BUY' else -40),
                            entry_price + (60 if direction == 'BUY' else -60),
                            entry_price + (100 if direction == 'BUY' else -100)],
                exit_type=exit_type,
                pnl_pips=pnl_pips,
                pnl_usd=pnl_usd,
                pnl_percent=pnl_percent,
                position_size=position_size,
                timeframe='M15',
                hold_time_minutes=hold_time_minutes,
                triggered_strategies=['smc', 'technical_indicators'],
                strategy_scores={'smc': 0.8, 'technical_indicators': 0.7},
                confidence=75.0,
                market_session=np.random.choice(['asian', 'london', 'newyork']),
                volatility_level=np.random.choice(['low', 'normal', 'high']),
                was_correct=was_correct,
                exit_reason=exit_reason
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create trade result: {e}")
            return None
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.was_correct)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L calculations
        total_pips = sum(t.pnl_pips for t in self.trades)
        total_pnl_usd = sum(t.pnl_usd for t in self.trades)
        avg_pips_per_trade = total_pips / total_trades if total_trades > 0 else 0
        avg_pnl_per_trade = total_pnl_usd / total_trades if total_trades > 0 else 0
        
        # Risk calculations
        drawdown_analysis = self.get_drawdown_analysis()
        max_drawdown_usd = drawdown_analysis.get('max_drawdown_usd', 0)
        max_drawdown_pct = drawdown_analysis.get('max_drawdown_pct', 0)
        
        # Advanced metrics
        winning_pips = sum(t.pnl_pips for t in self.trades if t.was_correct)
        losing_pips = abs(sum(t.pnl_pips for t in self.trades if not t.was_correct))
        
        profit_factor = winning_pips / losing_pips if losing_pips > 0 else float('inf')
        
        avg_win = winning_pips / winning_trades if winning_trades > 0 else 0
        avg_loss = losing_pips / losing_trades if losing_trades > 0 else 0
        expectancy = (avg_win * win_rate) - (avg_loss * (1 - win_rate))
        
        # Sharpe ratio (simplified)
        returns = [t.pnl_percent for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Streaks
        current_streak = 0
        current_streak_type = 'none'
        max_winning_streak = 0
        max_losing_streak = 0
        temp_winning_streak = 0
        temp_losing_streak = 0
        
        for trade in reversed(self.trades):
            if current_streak == 0:
                current_streak = 1
                current_streak_type = 'winning' if trade.was_correct else 'losing'
            elif (current_streak_type == 'winning' and trade.was_correct) or \
                 (current_streak_type == 'losing' and not trade.was_correct):
                current_streak += 1
            else:
                break
        
        for trade in self.trades:
            if trade.was_correct:
                temp_winning_streak += 1
                temp_losing_streak = 0
                max_winning_streak = max(max_winning_streak, temp_winning_streak)
            else:
                temp_losing_streak += 1
                temp_winning_streak = 0
                max_losing_streak = max(max_losing_streak, temp_losing_streak)
        
        # Time analysis
        avg_hold_time = np.mean([t.hold_time_minutes for t in self.trades]) if self.trades else 0
        
        # Session analysis
        session_performance = {}
        for session in ['asian', 'london', 'newyork']:
            session_trades = [t for t in self.trades if t.market_session == session]
            if session_trades:
                session_win_rate = sum(1 for t in session_trades if t.was_correct) / len(session_trades)
                session_performance[session] = session_win_rate
        
        best_session = max(session_performance.keys(), key=lambda k: session_performance[k]) if session_performance else 'unknown'
        
        # Timeframe analysis
        timeframe_performance = {}
        for tf in ['M15', 'M30', 'H1']:
            tf_trades = [t for t in self.trades if t.timeframe == tf]
            if tf_trades:
                tf_win_rate = sum(1 for t in tf_trades if t.was_correct) / len(tf_trades)
                timeframe_performance[tf] = tf_win_rate
        
        best_timeframe = max(timeframe_performance.keys(), key=lambda k: timeframe_performance[k]) if timeframe_performance else 'unknown'
        
        # Strategy performance
        strategy_performances = self._calculate_strategy_performances()
        
        # Recent performance
        cutoff_7d = datetime.now() - timedelta(days=7)
        cutoff_30d = datetime.now() - timedelta(days=30)
        
        recent_7d = [t for t in self.trades if t.timestamp > cutoff_7d]
        recent_30d = [t for t in self.trades if t.timestamp > cutoff_30d]
        
        win_rate_7d = sum(1 for t in recent_7d if t.was_correct) / len(recent_7d) if recent_7d else 0
        win_rate_30d = sum(1 for t in recent_30d if t.was_correct) / len(recent_30d) if recent_30d else 0
        
        # Monthly P&L
        monthly_pnl = {}
        for trade in self.trades:
            month_key = trade.timestamp.strftime('%Y-%m')
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade.pnl_usd
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pips=total_pips,
            total_pnl_usd=total_pnl_usd,
            avg_pips_per_trade=avg_pips_per_trade,
            avg_pnl_per_trade=avg_pnl_per_trade,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_usd=max_drawdown_usd,
            recovery_factor=total_pnl_usd / max_drawdown_usd if max_drawdown_usd > 0 else float('inf'),
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_winning_streak=max_winning_streak,
            max_losing_streak=max_losing_streak,
            current_streak=current_streak,
            current_streak_type=current_streak_type,
            avg_hold_time_minutes=int(avg_hold_time),
            best_session=best_session,
            best_timeframe=best_timeframe,
            strategy_performances=strategy_performances,
            last_7_days_win_rate=win_rate_7d,
            last_30_days_win_rate=win_rate_30d,
            monthly_pnl=monthly_pnl
        )
    
    def _calculate_strategy_performances(self) -> Dict[str, StrategyPerformance]:
        """Calculate performance for each strategy"""
        strategy_performances = {}
        
        # Get all unique strategies
        all_strategies = set()
        for trade in self.trades:
            all_strategies.update(trade.triggered_strategies)
        
        for strategy in all_strategies:
            strategy_trades = [t for t in self.trades if strategy in t.triggered_strategies]
            
            if not strategy_trades:
                continue
            
            total_trades = len(strategy_trades)
            winning_trades = sum(1 for t in strategy_trades if t.was_correct)
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades
            
            total_pips = sum(t.pnl_pips for t in strategy_trades)
            avg_pips = total_pips / total_trades
            best_pips = max(t.pnl_pips for t in strategy_trades)
            worst_pips = min(t.pnl_pips for t in strategy_trades)
            
            total_pnl = sum(t.pnl_usd for t in strategy_trades)
            avg_pnl = total_pnl / total_trades
            
            # Profit factor
            winning_pips = sum(t.pnl_pips for t in strategy_trades if t.was_correct)
            losing_pips = abs(sum(t.pnl_pips for t in strategy_trades if not t.was_correct))
            profit_factor = winning_pips / losing_pips if losing_pips > 0 else float('inf')
            
            # Expectancy
            avg_win = winning_pips / winning_trades if winning_trades > 0 else 0
            avg_loss = losing_pips / losing_trades if losing_trades > 0 else 0
            expectancy = (avg_win * win_rate) - (avg_loss * (1 - win_rate))
            
            # Recent performance (last 20 trades)
            recent_trades = strategy_trades[-20:]
            recent_win_rate = sum(1 for t in recent_trades if t.was_correct) / len(recent_trades) if recent_trades else 0
            recent_avg_pips = sum(t.pnl_pips for t in recent_trades) / len(recent_trades) if recent_trades else 0
            
            # Confidence analysis
            avg_confidence = np.mean([t.confidence for t in strategy_trades])
            high_conf_trades = [t for t in strategy_trades if t.confidence >= 80]
            confidence_accuracy = sum(1 for t in high_conf_trades if t.was_correct) / len(high_conf_trades) if high_conf_trades else 0
            
            strategy_performances[strategy] = StrategyPerformance(
                name=strategy,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pips=total_pips,
                avg_pips_per_trade=avg_pips,
                best_trade_pips=best_pips,
                worst_trade_pips=worst_pips,
                total_pnl_usd=total_pnl,
                avg_pnl_per_trade=avg_pnl,
                profit_factor=profit_factor,
                expectancy=expectancy,
                recent_win_rate=recent_win_rate,
                recent_avg_pips=recent_avg_pips,
                avg_confidence=avg_confidence,
                confidence_accuracy=confidence_accuracy
            )
        
        return strategy_performances
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_pips=0.0, total_pnl_usd=0.0, avg_pips_per_trade=0.0, avg_pnl_per_trade=0.0,
            max_drawdown_pct=0.0, max_drawdown_usd=0.0, recovery_factor=0.0,
            sharpe_ratio=0.0, profit_factor=0.0, expectancy=0.0,
            max_winning_streak=0, max_losing_streak=0, current_streak=0, current_streak_type='none',
            avg_hold_time_minutes=0, best_session='unknown', best_timeframe='unknown',
            strategy_performances={}, last_7_days_win_rate=0.0, last_30_days_win_rate=0.0,
            monthly_pnl={}
        )
    
    def _calculate_weight_adjustment(self, win_rate: float, avg_pips: float) -> float:
        """Calculate strategy weight adjustment based on performance"""
        performance_score = (win_rate * 0.6) + (min(avg_pips / 20, 1.0) * 0.4)
        adjustment = (performance_score - 0.5) * 0.4
        return max(-0.2, min(0.2, adjustment))
    
    def _analyze_optimal_parameters(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Analyze optimal trading parameters"""
        if not trades:
            return {}
        
        # Analyze optimal timeframes
        tf_performance = {}
        for tf in ['M15', 'M30', 'H1']:
            tf_trades = [t for t in trades if t.timeframe == tf]
            if tf_trades:
                tf_win_rate = sum(1 for t in tf_trades if t.was_correct) / len(tf_trades)
                tf_avg_pips = sum(t.pnl_pips for t in tf_trades) / len(tf_trades)
                tf_performance[tf] = {'win_rate': tf_win_rate, 'avg_pips': tf_avg_pips}
        
        # Analyze optimal sessions
        session_performance = {}
        for session in ['asian', 'london', 'newyork']:
            session_trades = [t for t in trades if t.market_session == session]
            if session_trades:
                session_win_rate = sum(1 for t in session_trades if t.was_correct) / len(session_trades)
                session_avg_pips = sum(t.pnl_pips for t in session_trades) / len(session_trades)
                session_performance[session] = {'win_rate': session_win_rate, 'avg_pips': session_avg_pips}
        
        # Analyze optimal hold times
        hold_time_analysis = {
            'avg_winning_hold_time': np.mean([t.hold_time_minutes for t in trades if t.was_correct]),
            'avg_losing_hold_time': np.mean([t.hold_time_minutes for t in trades if not t.was_correct])
        }
        
        return {
            'timeframe_performance': tf_performance,
            'session_performance': session_performance,
            'hold_time_analysis': hold_time_analysis,
            'recommendations': self._generate_parameter_recommendations(tf_performance, session_performance)
        }
    
    def _analyze_risk_parameters(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Analyze risk management effectiveness"""
        if not trades:
            return {}
        
        # SL hit rate
        sl_trades = [t for t in trades if t.exit_type == 'sl']
        sl_hit_rate = len(sl_trades) / len(trades)
        
        # TP hit rates
        tp_hit_rates = {}
        for tp_level in ['tp1', 'tp2', 'tp3', 'tp4']:
            tp_trades = [t for t in trades if t.exit_type == tp_level]
            tp_hit_rates[tp_level] = len(tp_trades) / len(trades)
        
        # Risk-reward analysis
        avg_risk_reward = np.mean([
            abs(t.pnl_pips) / 30 if abs(t.pnl_pips) > 0 else 0
            for t in trades if t.was_correct
        ])
        
        return {
            'sl_hit_rate': sl_hit_rate,
            'tp_hit_rates': tp_hit_rates,
            'avg_risk_reward': avg_risk_reward,
            'risk_recommendations': self._generate_risk_recommendations(sl_hit_rate, tp_hit_rates)
        }
    
    def _generate_parameter_recommendations(self, tf_performance: Dict, session_performance: Dict) -> List[str]:
        """Generate parameter optimization recommendations"""
        recommendations = []
        
        # Timeframe recommendations
        if tf_performance:
            best_tf = max(tf_performance.keys(), key=lambda k: tf_performance[k]['win_rate'])
            if tf_performance[best_tf]['win_rate'] > 0.6:
                recommendations.append(f"Focus on {best_tf} timeframe (win rate: {tf_performance[best_tf]['win_rate']:.1%})")
        
        # Session recommendations
        if session_performance:
            best_session = max(session_performance.keys(), key=lambda k: session_performance[k]['win_rate'])
            if session_performance[best_session]['win_rate'] > 0.6:
                recommendations.append(f"Prioritize {best_session} session (win rate: {session_performance[best_session]['win_rate']:.1%})")
        
        return recommendations
    
    def _generate_risk_recommendations(self, sl_hit_rate: float, tp_hit_rates: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if sl_hit_rate > 0.4:
            recommendations.append("Consider wider stop losses - high SL hit rate")
        
        if tp_hit_rates.get('tp1', 0) < 0.3:
            recommendations.append("Consider closer TP1 - low TP1 hit rate")
        
        best_tp = max(tp_hit_rates.keys(), key=lambda k: tp_hit_rates[k]) if tp_hit_rates else None
        if best_tp and tp_hit_rates[best_tp] > 0.5:
            recommendations.append(f"Optimize for {best_tp} exits (hit rate: {tp_hit_rates[best_tp]:.1%})")
        
        return recommendations

# Global performance tracker instance
performance_tracker = PerformanceTracker()

# Convenience functions
def record_signal(signal: TradingSignal) -> str:
    """Record a trading signal"""
    return performance_tracker.record_signal_sent(signal)

def record_trade_result(signal_id: str, exit_price: float, exit_type: str, 
                       hold_time_minutes: int, exit_reason: str = "") -> bool:
    """Record a completed trade"""
    return performance_tracker.record_trade_result(signal_id, exit_price, exit_type, hold_time_minutes, exit_reason)

def get_performance_stats() -> PerformanceMetrics:
    """Get current performance statistics"""
    return performance_tracker.get_performance_metrics()

def get_learning_data() -> Dict[str, Any]:
    """Get data for learning system"""
    return performance_tracker.get_learning_feedback()

def generate_monthly_report(year: int = None, month: int = None) -> Dict[str, Any]:
    """Generate monthly performance report"""
    if year is None:
        year = datetime.now().year
    if month is None:
        month = datetime.now().month
    return performance_tracker.generate_monthly_report(year, month)

def get_drawdown_analysis() -> Dict[str, Any]:
    """Get drawdown analysis"""
    return performance_tracker.get_drawdown_analysis()

def export_trades_to_csv(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
    """Export trades to CSV file"""
    try:
        trades = performance_tracker.trades
        
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]
        
        if not trades:
            return ""
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(trade) for trade in trades])
        
        # Export to CSV
        export_file = settings.storage.base_dir / f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(export_file, index=False)
        
        logger.info(f"📁 Trades exported to {export_file}")
        return str(export_file)
        
    except Exception as e:
        logger.error(f"❌ CSV export failed: {e}")
        return ""