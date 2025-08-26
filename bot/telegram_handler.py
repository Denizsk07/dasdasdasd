"""XAUUSD Telegram Bot Handler - Sends signals to your group"""
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from telegram import Bot
from telegram.error import TelegramError

from config.settings import settings
from utils.logger import get_module_logger
from analysis.signal_generator import TradingSignal

logger = get_module_logger('telegram_handler')

class TelegramSignalBot:
    def __init__(self):
        self.bot_token = settings.telegram.bot_token
        self.group_id = settings.telegram.group_id
        self.bot = Bot(token=self.bot_token)
        
    async def send_signal(self, signal: TradingSignal) -> bool:
        """Send trading signal to Telegram group"""
        try:
            message = self._format_signal_message(signal)
            
            await self.bot.send_message(
                chat_id=self.group_id,
                text=message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            
            logger.info(f"📤 Signal sent: {signal.direction} @ ${signal.entry_price:.2f}")
            return True
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Signal send failed: {e}")
            return False
    
    async def send_status_update(self, message: str) -> bool:
        """Send status update to group"""
        try:
            await self.bot.send_message(
                chat_id=self.group_id,
                text=f"🤖 <b>XAUUSD Bot Status</b>\n\n{message}",
                parse_mode='HTML'
            )
            return True
        except Exception as e:
            logger.error(f"❌ Status send failed: {e}")
            return False
    
    def _format_signal_message(self, signal: TradingSignal) -> str:
        """Format signal for Telegram"""
        direction_emoji = "🟢" if signal.direction == 'BUY' else "🔴"
        arrow_emoji = "📈" if signal.direction == 'BUY' else "📉"
        
        # R:R ratios
        rr_text = " | ".join([f"1:{rr:.1f}" for rr in signal.risk_reward_ratios])
        
        # Top 3 reasons
        reasons = "\n".join([f"• {reason}" for reason in signal.reasoning[:3]])
        
        message = f"""{direction_emoji} <b>{signal.direction} XAUUSD SIGNAL</b> {direction_emoji}
{arrow_emoji} <b>Confidence: {signal.confidence:.1f}%</b>

💰 <b>TRADE SETUP:</b>
🔵 Entry: <b>${signal.entry_price:.2f}</b>
🛑 Stop Loss: <b>${signal.stop_loss:.2f}</b>

🎯 <b>TAKE PROFITS:</b>
TP1: ${signal.take_profits[0]:.2f}
TP2: ${signal.take_profits[1]:.2f} 
TP3: ${signal.take_profits[2]:.2f}
TP4: ${signal.take_profits[3]:.2f}

📊 <b>RISK METRICS:</b>
💎 Position Size: {signal.position_size:.3f} lots
📏 Risk:Reward: {rr_text}
💰 Max Risk: ${signal.max_risk_usd:.2f}

🧠 <b>ANALYSIS:</b>
{reasons}

⏰ <b>Session:</b> {signal.market_session.title()}
📈 <b>Trend:</b> {signal.trend_context.title()}
🌊 <b>Volatility:</b> {signal.volatility_level.title()}

🔬 <b>Signal ID:</b> {signal.signal_id}
🕐 <b>Time:</b> {signal.timestamp.strftime('%H:%M:%S UTC')}

🤖 <i>XAUUSD Trading Bot • Auto-Analysis</i>"""

        return message

# Global bot instance
telegram_bot = TelegramSignalBot()

async def send_trading_signal(signal: TradingSignal) -> bool:
    """Send trading signal"""
    return await telegram_bot.send_signal(signal)

async def send_bot_status(message: str) -> bool:
    """Send status update"""
    return await telegram_bot.send_status_update(message)