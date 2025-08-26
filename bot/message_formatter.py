"""
XAUUSD Telegram Message Formatter - Professional Message Templates
Creates beautifully formatted messages for all bot communications
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

from config.settings import settings
from utils.logger import get_module_logger
from analysis.signal_generator import TradingSignal
from trading.performance_tracker import PerformanceMetrics, TradeResult
from learning.strategy_optimizer import OptimizationResult

logger = get_module_logger('message_formatter')

@dataclass
class MessageTemplate:
    """Message template configuration"""
    use_emojis: bool = True
    use_html: bool = True
    max_length: int = 4096  # Telegram limit
    include_timestamp: bool = True
    include_branding: bool = True

class MessageFormatter:
    """Professional message formatter for Telegram"""
    
    def __init__(self):
        self.template = MessageTemplate()
        self.brand_name = "XAUUSD Trading Bot"
        self.brand_emoji = "🤖"
        
    def format_trading_signal(self, signal: TradingSignal, chart_path: Optional[str] = None) -> Dict[str, Any]:
        """Format trading signal message with professional layout"""
        
        try:
            direction_emoji = "🟢" if signal.direction == 'BUY' else "🔴"
            arrow_emoji = "📈" if signal.direction == 'BUY' else "📉"
            
            rr_ratios = " | ".join([f"1:{rr:.1f}" for rr in signal.risk_reward_ratios[:3]])
            
            top_strategies = signal.triggered_strategies[:3]
            strategy_text = " + ".join([s.replace('_', ' ').title() for s in top_strategies])
            
            session_emoji = self._get_session_emoji(signal.market_session)
            trend_emoji = self._get_trend_emoji(signal.trend_context)
            volatility_emoji = self._get_volatility_emoji(signal.volatility_level)
            
            message = f"""{direction_emoji} <b>XAUUSD {signal.direction} SIGNAL</b> {direction_emoji}
{arrow_emoji} <b>Confidence: {signal.confidence:.1f}%</b> | {session_emoji} <b>{signal.market_session.title()} Session</b>

💰 <b>TRADE SETUP:</b>
🔵 <b>Entry:</b> <code>${signal.entry_price:.2f}</code>
🛑 <b>Stop Loss:</b> <code>${signal.stop_loss:.2f}</code>
💎 <b>Position Size:</b> <code>{signal.position_size:.3f} lots</code>
💸 <b>Max Risk:</b> <code>${signal.max_risk_usd:.2f}</code>

🎯 <b>TAKE PROFIT LEVELS:</b>"""

            for i, (tp, rr) in enumerate(zip(signal.take_profits, signal.risk_reward_ratios), 1):
                message += f"\n<b>TP{i}:</b> <code>${tp:.2f}</code> (1:{rr:.1f})"
            
            message += f"""

🧠 <b>ANALYSIS SUMMARY:</b>
📊 <b>Strategies:</b> {strategy_text}
{trend_emoji} <b>Trend Context:</b> {signal.trend_context.title()}
{volatility_emoji} <b>Volatility:</b> {signal.volatility_level.title()}

📋 <b>KEY REASONS:</b>"""

            for i, reason in enumerate(signal.reasoning[:3], 1):
                message += f"\n{i}. {reason}"
            
            message += f"""

⏰ <b>TIMING:</b>
🕐 <b>Signal Time:</b> {signal.timestamp.strftime('%H:%M:%S UTC')}
📅 <b>Date:</b> {signal.timestamp.strftime('%Y-%m-%d')}
⚡ <b>Timeframe:</b> {signal.timeframe}

🔬 <b>Signal ID:</b> <code>{signal.signal_id}</code>
{self.brand_emoji} <i>{self.brand_name} • Automated Analysis</i>"""

            return {
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True,
                'photo_path': chart_path,
                'caption': f"{direction_emoji} {signal.direction} XAUUSD @ ${signal.entry_price:.2f} | {signal.confidence:.1f}% confidence"
            }
            
        except Exception as e:
            logger.error(f"❌ Signal message formatting failed: {e}")
            return {'text': f"Signal formatting error: {str(e)}", 'parse_mode': 'HTML'}
    
    def format_performance_report(self, metrics: PerformanceMetrics, chart_path: Optional[str] = None) -> Dict[str, Any]:
        """Format comprehensive performance report"""
        
        try:
            win_rate_emoji = "🟢" if metrics.win_rate >= 0.6 else ("🟡" if metrics.win_rate >= 0.5 else "🔴")
            pnl_emoji = "💚" if metrics.total_pnl_usd > 0 else ("💛" if metrics.total_pnl_usd == 0 else "❤️")
            trend_emoji = "📈" if metrics.total_pips > 0 else ("📊" if metrics.total_pips == 0 else "📉")
            
            message = f"""📊 <b>PERFORMANCE REPORT</b> 📊
📅 <b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

{win_rate_emoji} <b>OVERALL PERFORMANCE:</b>
📈 <b>Total Trades:</b> <code>{metrics.total_trades}</code>
🎯 <b>Win Rate:</b> <code>{metrics.win_rate:.1%}</code> ({metrics.winning_trades}W / {metrics.losing_trades}L)
{trend_emoji} <b>Total Pips:</b> <code>{metrics.total_pips:+.1f}</code>
{pnl_emoji} <b>Total P&L:</b> <code>${metrics.total_pnl_usd:+.2f}</code>

💰 <b>AVERAGES:</b>
📏 <b>Avg Pips/Trade:</b> <code>{metrics.avg_pips_per_trade:+.1f}</code>
💵 <b>Avg P&L/Trade:</b> <code>${metrics.avg_pnl_per_trade:+.2f}</code>

⚠️ <b>RISK METRICS:</b>
📉 <b>Max Drawdown:</b> <code>{metrics.max_drawdown_pct:.1f}%</code>
🏆 <b>Recovery Factor:</b> <code>{metrics.recovery_factor:.2f}</code>
📊 <b>Profit Factor:</b> <code>{metrics.profit_factor:.2f}</code>

🔥 <b>STREAKS:</b>
🟢 <b>Best Winning:</b> <code>{metrics.max_winning_streak}</code>
🔴 <b>Worst Losing:</b> <code>{metrics.max_losing_streak}</code>

⏰ <b>TIME ANALYSIS:</b>
🕐 <b>Avg Hold Time:</b> <code>{metrics.avg_hold_time_minutes} min</code>
🌏 <b>Best Session:</b> <code>{metrics.best_session.title()}</code>
📊 <b>Best Timeframe:</b> <code>{metrics.best_timeframe}</code>

📈 <b>RECENT PERFORMANCE:</b>
📅 <b>Last 7 Days:</b> <code>{metrics.last_7_days_win_rate:.1%}</code>
📅 <b>Last 30 Days:</b> <code>{metrics.last_30_days_win_rate:.1%}</code>"""

            if metrics.strategy_performances:
                message += "\n\n🧠 <b>TOP STRATEGIES:</b>"
                
                sorted_strategies = sorted(
                    metrics.strategy_performances.items(),
                    key=lambda x: x[1].win_rate,
                    reverse=True
                )
                
                for strategy_name, perf in sorted_strategies[:3]:
                    strategy_emoji = "🟢" if perf.win_rate >= 0.6 else ("🟡" if perf.win_rate >= 0.5 else "🔴")
                    clean_name = strategy_name.replace('_', ' ').title()
                    message += f"\n{strategy_emoji} <b>{clean_name}:</b> {perf.win_rate:.1%}"
            
            message += f"\n\n{self.brand_emoji} <i>{self.brand_name} • Performance Analytics</i>"
            
            return {
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True,
                'photo_path': chart_path,
                'caption': f"📊 Performance Dashboard | {metrics.win_rate:.1%} Win Rate"
            }
            
        except Exception as e:
            logger.error(f"❌ Performance report formatting failed: {e}")
            return {'text': f"Performance report error: {str(e)}", 'parse_mode': 'HTML'}
    
    def format_bot_status(self, status_type: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format bot status messages"""
        
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            
            if status_type == 'startup':
                message = f"""🚀 <b>XAUUSD BOT STARTED</b> 🚀
⚡ <b>System Online</b> | {current_time}

🔧 <b>CONFIGURATION:</b>
📊 <b>Analysis Interval:</b> <code>{settings.trading.analysis_interval//60} minutes</code>
🎯 <b>Min Signal Score:</b> <code>{settings.trading.min_signal_score}</code>
💰 <b>Risk per Trade:</b> <code>{settings.trading.risk_percentage}%</code>
📈 <b>Timeframes:</b> <code>{', '.join(settings.trading.timeframes)}</code>

📡 <b>DATA SOURCES:</b>
📊 <b>TradingView:</b> {'✅ Connected' if settings.tradingview.enabled else '❌ Disabled'}
📈 <b>Yahoo Finance:</b> {'✅ Fallback Ready' if settings.yahoo_finance.enabled else '❌ Disabled'}

🎯 <b>READY TO ANALYZE XAUUSD MARKETS</b>"""

            elif status_type == 'shutdown':
                message = f"""⏹️ <b>XAUUSD BOT STOPPED</b> ⏹️
🛑 <b>System Shutdown</b> | {current_time}

📊 <b>SESSION SUMMARY:</b>
🎯 <b>Signals Sent:</b> <code>{details.get('signals_sent', 0)}</code>
💰 <b>Session P&L:</b> <code>${details.get('session_pnl', 0):+.2f}</code>

👋 <b>Bot offline until manual restart</b>"""

            elif status_type == 'error':
                message = f"""❌ <b>SYSTEM ERROR</b> ❌
🚨 <b>Error Detected</b> | {current_time}

🔍 <b>ERROR DETAILS:</b>
<code>{details.get('error_message', 'Unknown error')}</code>

🔄 <b>Bot attempting automatic recovery...</b>"""

            elif status_type == 'daily_report':
                message = f"""📋 <b>DAILY REPORT</b> 📋
📅 {current_time}

📈 <b>TODAY'S ACTIVITY:</b>
🎯 <b>Signals Generated:</b> <code>{details.get('signals_today', 0)}</code>
💰 <b>Daily P&L:</b> <code>${details.get('daily_pnl', 0):+.2f}</code>
📊 <b>Win Rate:</b> <code>{details.get('daily_win_rate', 0):.1%}</code>

🤖 <b>Bot Status:</b> Online and monitoring"""

            else:
                message = f"{self.brand_emoji} <b>Bot Status Update</b>\n{current_time}\n\n{details.get('message', 'Status update')}"
            
            message += f"\n\n{self.brand_emoji} <i>{self.brand_name}</i>"
            
            return {
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
        except Exception as e:
            logger.error(f"❌ Status message formatting failed: {e}")
            return {'text': f"Status formatting error: {str(e)}", 'parse_mode': 'HTML'}
    
    def _get_session_emoji(self, session: str) -> str:
        """Get emoji for trading session"""
        session_emojis = {
            'asian': '🌏',
            'london': '🇬🇧', 
            'newyork': '🇺🇸',
            'unknown': '🌍'
        }
        return session_emojis.get(session.lower(), '🌍')
    
    def _get_trend_emoji(self, trend: str) -> str:
        """Get emoji for trend context"""
        trend_emojis = {
            'uptrend': '📈',
            'downtrend': '📉',
            'sideways': '📊',
            'neutral': '📊'
        }
        return trend_emojis.get(trend.lower(), '📊')
    
    def _get_volatility_emoji(self, volatility: str) -> str:
        """Get emoji for volatility level"""
        volatility_emojis = {
            'low': '😴',
            'normal': '🌊',
            'high': '⚡',
            'extreme': '🔥'
        }
        return volatility_emojis.get(volatility.lower(), '🌊')

# Global message formatter instance
message_formatter = MessageFormatter()

# Convenience functions
def format_trading_signal(signal: TradingSignal, chart_path: Optional[str] = None) -> Dict[str, Any]:
    """Format trading signal message"""
    return message_formatter.format_trading_signal(signal, chart_path)

def format_performance_report(metrics: PerformanceMetrics, chart_path: Optional[str] = None) -> Dict[str, Any]:
    """Format performance report message"""
    return message_formatter.format_performance_report(metrics, chart_path)

def format_learning_update(optimization: OptimizationResult, learning_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format learning update message"""
    try:
        improvement_emoji = "🚀" if optimization.improvement_score > 0.1 else "📈"
        confidence_emoji = "🟢" if optimization.confidence >= 0.8 else ("🟡" if optimization.confidence >= 0.6 else "🔴")
        
        message = f"""🧠 <b>LEARNING UPDATE</b> 🧠
⚡ <b>Optimization Complete</b> | {optimization.optimization_date.strftime('%Y-%m-%d %H:%M UTC')}

{improvement_emoji} <b>RESULTS:</b>
📊 <b>Improvement:</b> <code>{optimization.improvement_score:+.1%}</code>
{confidence_emoji} <b>Confidence:</b> <code>{optimization.confidence:.0%}</code>
📈 <b>Trades Analyzed:</b> <code>{optimization.trades_analyzed}</code>

🔧 <b>WEIGHT CHANGES:</b>"""

        for strategy, (old_weight, new_weight) in zip(
            optimization.old_weights.keys(),
            zip(optimization.old_weights.values(), optimization.new_weights.values())
        ):
            change_pct = ((new_weight - old_weight) / old_weight * 100) if old_weight > 0 else 0
            
            if abs(change_pct) >= 5:
                change_emoji = "🔺" if change_pct > 0 else "🔻"
                clean_strategy = strategy.replace('_', ' ').title()
                message += f"\n{change_emoji} <b>{clean_strategy}:</b> {change_pct:+.1f}%"
        
        if optimization.reasoning:
            message += "\n\n💡 <b>KEY INSIGHTS:</b>"
            for i, reason in enumerate(optimization.reasoning[:2], 1):
                message += f"\n{i}. {reason}"
        
        message += f"\n\n🤖 <i>{message_formatter.brand_name} • Self-Learning AI</i>"
        
        return {
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
    except Exception as e:
        logger.error(f"❌ Learning update formatting failed: {e}")
        return {'text': f"Learning update error: {str(e)}", 'parse_mode': 'HTML'}

def format_bot_status(status_type: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format bot status message"""
    return message_formatter.format_bot_status(status_type, details)