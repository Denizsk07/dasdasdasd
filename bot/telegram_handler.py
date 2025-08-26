"""XAUUSD Telegram Bot Handler - Sends signals to your group"""
import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from telegram import Bot
    from telegram.error import TelegramError, NetworkError, RetryAfter, TimedOut
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ python-telegram-bot not installed. Install with: pip install python-telegram-bot")

from config.settings import settings
from utils.logger import get_module_logger
from analysis.signal_generator import TradingSignal
from bot.message_formatter import format_trading_signal, format_bot_status, format_performance_report
from trading.performance_tracker import PerformanceMetrics

logger = get_module_logger('telegram_handler')

class TelegramSignalBot:
    """Main Telegram bot for sending XAUUSD trading signals"""
    
    def __init__(self):
        self.bot_token = settings.telegram.bot_token
        self.group_id = settings.telegram.group_id
        self.bot = None
        self.is_initialized = False
        self.last_error_time = None
        self.error_count = 0
        self.max_retries = 3
        
        if not TELEGRAM_AVAILABLE:
            logger.error("❌ Telegram bot unavailable - python-telegram-bot not installed")
            return
        
        if not self.bot_token or not self.group_id:
            logger.error("❌ Telegram credentials missing in .env file")
            return
        
        self._initialize_bot()
    
    def _initialize_bot(self):
        """Initialize Telegram bot"""
        try:
            if not TELEGRAM_AVAILABLE:
                return False
            
            self.bot = Bot(token=self.bot_token)
            self.is_initialized = True
            logger.info("✅ Telegram bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram bot initialization failed: {e}")
            self.is_initialized = False
            return False
    
    async def send_signal(self, signal: TradingSignal, chart_path: Optional[str] = None) -> bool:
        """Send trading signal to Telegram group"""
        if not self.is_initialized:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            # Format the signal message
            message_data = format_trading_signal(signal, chart_path)
            
            # Send with photo if chart available
            if chart_path and Path(chart_path).exists():
                success = await self._send_photo_with_caption(
                    photo_path=chart_path,
                    caption=message_data['text'],
                    parse_mode=message_data['parse_mode']
                )
            else:
                # Send as text message only
                success = await self._send_text_message(
                    text=message_data['text'],
                    parse_mode=message_data['parse_mode'],
                    disable_web_page_preview=message_data.get('disable_web_page_preview', True)
                )
            
            if success:
                logger.info(f"📤 Signal sent: {signal.direction} @ ${signal.entry_price:.2f} ({signal.confidence:.1f}%)")
                self._reset_error_count()
                return True
            else:
                logger.error(f"❌ Failed to send signal: {signal.signal_id}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Signal send error: {e}")
            self._handle_error(e)
            return False
    
    async def send_status_update(self, status_type: str, details: Dict[str, Any] = None) -> bool:
        """Send status update to group"""
        if not self.is_initialized:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            message_data = format_bot_status(status_type, details)
            
            success = await self._send_text_message(
                text=message_data['text'],
                parse_mode=message_data['parse_mode'],
                disable_web_page_preview=message_data.get('disable_web_page_preview', True)
            )
            
            if success:
                logger.info(f"📤 Status update sent: {status_type}")
                self._reset_error_count()
                return True
            else:
                logger.error(f"❌ Failed to send status update: {status_type}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Status update error: {e}")
            self._handle_error(e)
            return False
    
    async def send_performance_report(self, metrics: PerformanceMetrics, 
                                    chart_path: Optional[str] = None) -> bool:
        """Send performance report to group"""
        if not self.is_initialized:
            logger.error("❌ Telegram bot not initialized")
            return False
        
        try:
            message_data = format_performance_report(metrics, chart_path)
            
            # Send with chart if available
            if chart_path and Path(chart_path).exists():
                success = await self._send_photo_with_caption(
                    photo_path=chart_path,
                    caption=message_data['text'],
                    parse_mode=message_data['parse_mode']
                )
            else:
                success = await self._send_text_message(
                    text=message_data['text'],
                    parse_mode=message_data['parse_mode'],
                    disable_web_page_preview=message_data.get('disable_web_page_preview', True)
                )
            
            if success:
                logger.info("📤 Performance report sent")
                self._reset_error_count()
                return True
            else:
                logger.error("❌ Failed to send performance report")
                return False
            
        except Exception as e:
            logger.error(f"❌ Performance report error: {e}")
            self._handle_error(e)
            return False
    
    async def _send_text_message(self, text: str, parse_mode: str = 'HTML', 
                               disable_web_page_preview: bool = True) -> bool:
        """Send text message with retry logic"""
        if not self.bot:
            return False
        
        for attempt in range(self.max_retries):
            try:
                await self.bot.send_message(
                    chat_id=self.group_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_web_page_preview
                )
                return True
                
            except RetryAfter as e:
                retry_after = int(e.retry_after)
                logger.warning(f"⏳ Rate limited, waiting {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                
            except TimedOut as e:
                logger.warning(f"⏰ Request timed out on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except NetworkError as e:
                logger.warning(f"🌐 Network error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                
            except TelegramError as e:
                logger.error(f"❌ Telegram API error: {e}")
                if "chat not found" in str(e).lower() or "forbidden" in str(e).lower():
                    logger.error("❌ Bot may not be added to the group or lacks permissions")
                    return False
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
        
        logger.error(f"❌ Failed to send message after {self.max_retries} attempts")
        return False
    
    async def _send_photo_with_caption(self, photo_path: str, caption: str, 
                                     parse_mode: str = 'HTML') -> bool:
        """Send photo with caption and retry logic"""
        if not self.bot:
            return False
        
        try:
            photo_file = Path(photo_path)
            if not photo_file.exists():
                logger.error(f"❌ Chart file not found: {photo_path}")
                return False
            
            # Check file size (Telegram limit is 50MB for photos)
            file_size = photo_file.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB
                logger.error(f"❌ Chart file too large: {file_size / 1024 / 1024:.1f}MB")
                return await self._send_text_message(caption, parse_mode)
            
            for attempt in range(self.max_retries):
                try:
                    with open(photo_file, 'rb') as photo:
                        await self.bot.send_photo(
                            chat_id=self.group_id,
                            photo=photo,
                            caption=caption,
                            parse_mode=parse_mode
                        )
                    return True
                    
                except RetryAfter as e:
                    retry_after = int(e.retry_after)
                    logger.warning(f"⏳ Rate limited, waiting {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                    
                except TimedOut as e:
                    logger.warning(f"⏰ Photo upload timed out on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(3 ** attempt)
                    
                except NetworkError as e:
                    logger.warning(f"🌐 Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    
                except TelegramError as e:
                    logger.error(f"❌ Telegram photo error: {e}")
                    if "too large" in str(e).lower():
                        logger.warning("📷 Photo too large, sending as text only")
                        return await self._send_text_message(caption, parse_mode)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        
                except Exception as e:
                    logger.error(f"❌ Photo send error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            
            logger.error(f"❌ Failed to send photo after {self.max_retries} attempts")
            # Fallback: send as text message
            return await self._send_text_message(caption, parse_mode)
            
        except Exception as e:
            logger.error(f"❌ Photo preparation error: {e}")
            return await self._send_text_message(caption, parse_mode)
    
    async def test_connection(self) -> bool:
        """Test bot connection and permissions"""
        if not self.is_initialized:
            return False
        
        try:
            # Get bot info
            bot_info = await self.bot.get_me()
            logger.info(f"🤖 Bot info: @{bot_info.username} ({bot_info.first_name})")
            
            # Test sending a simple message
            test_message = f"🧪 <b>Connection Test</b>\n⚡ Bot online: {datetime.now().strftime('%H:%M:%S UTC')}"
            
            success = await self._send_text_message(test_message)
            
            if success:
                logger.info("✅ Telegram connection test successful")
                return True
            else:
                logger.error("❌ Telegram connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test error: {e}")
            return False
    
    async def send_startup_message(self) -> bool:
        """Send bot startup notification"""
        return await self.send_status_update('startup', {
            'signals_sent': 0,
            'analysis_interval': settings.trading.analysis_interval // 60,
            'min_signal_score': settings.trading.min_signal_score,
            'risk_percentage': settings.trading.risk_percentage
        })
    
    async def send_shutdown_message(self, session_stats: Dict[str, Any] = None) -> bool:
        """Send bot shutdown notification"""
        return await self.send_status_update('shutdown', session_stats or {
            'signals_sent': 0,
            'session_pnl': 0.0,
            'uptime_hours': 0
        })
    
    async def send_error_notification(self, error_message: str, context: str = "") -> bool:
        """Send error notification"""
        return await self.send_status_update('error', {
            'error_message': error_message[:200],  # Truncate long messages
            'context': context,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        })
    
    async def send_daily_report(self, daily_stats: Dict[str, Any]) -> bool:
        """Send daily performance report"""
        return await self.send_status_update('daily_report', daily_stats)
    
    def _handle_error(self, error: Exception):
        """Handle and log errors with rate limiting"""
        current_time = datetime.now()
        
        # Reset error count if last error was > 1 hour ago
        if self.last_error_time and (current_time - self.last_error_time).total_seconds() > 3600:
            self.error_count = 0
        
        self.error_count += 1
        self.last_error_time = current_time
        
        # Log error with context
        logger.error(f"❌ Telegram error #{self.error_count}: {error}")
        
        # If too many errors, try to reinitialize
        if self.error_count >= 5:
            logger.warning("⚠️ Too many Telegram errors, attempting to reinitialize...")
            self._initialize_bot()
    
    def _reset_error_count(self):
        """Reset error counter after successful operation"""
        if self.error_count > 0:
            logger.info(f"✅ Telegram connection recovered after {self.error_count} errors")
            self.error_count = 0
            self.last_error_time = None
    
    def get_bot_status(self) -> Dict[str, Any]:
        """Get current bot status information"""
        return {
            'is_initialized': self.is_initialized,
            'telegram_available': TELEGRAM_AVAILABLE,
            'bot_token_set': bool(self.bot_token),
            'group_id_set': bool(self.group_id),
            'error_count': self.error_count,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
        }

class TelegramBotManager:
    """Manager class for multiple bot operations"""
    
    def __init__(self):
        self.main_bot = TelegramSignalBot()
        self.is_running = False
    
    async def start(self) -> bool:
        """Start the Telegram bot manager"""
        if not self.main_bot.is_initialized:
            logger.error("❌ Cannot start - Telegram bot not initialized")
            return False
        
        try:
            # Test connection
            connection_ok = await self.main_bot.test_connection()
            if not connection_ok:
                logger.error("❌ Telegram connection test failed")
                return False
            
            # Send startup message
            await self.main_bot.send_startup_message()
            
            self.is_running = True
            logger.info("✅ Telegram bot manager started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot manager start failed: {e}")
            return False
    
    async def stop(self, session_stats: Dict[str, Any] = None):
        """Stop the Telegram bot manager"""
        if self.is_running:
            try:
                await self.main_bot.send_shutdown_message(session_stats)
                self.is_running = False
                logger.info("✅ Telegram bot manager stopped")
            except Exception as e:
                logger.error(f"❌ Bot manager stop error: {e}")
    
    async def send_signal(self, signal: TradingSignal, chart_path: Optional[str] = None) -> bool:
        """Send trading signal"""
        if not self.is_running:
            logger.warning("⚠️ Bot manager not running")
            return False
        
        return await self.main_bot.send_signal(signal, chart_path)
    
    async def send_status(self, status_type: str, details: Dict[str, Any] = None) -> bool:
        """Send status update"""
        if not self.is_running:
            return False
        
        return await self.main_bot.send_status_update(status_type, details)
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        bot_status = self.main_bot.get_bot_status()
        bot_status['manager_running'] = self.is_running
        return bot_status

# Global bot instances
telegram_bot = TelegramSignalBot()
bot_manager = TelegramBotManager()

# Convenience functions
async def send_trading_signal(signal: TradingSignal, chart_path: Optional[str] = None) -> bool:
    """Send trading signal to Telegram"""
    return await telegram_bot.send_signal(signal, chart_path)

async def send_bot_status(status_type: str, details: Dict[str, Any] = None) -> bool:
    """Send bot status update"""
    return await telegram_bot.send_status_update(status_type, details)

async def send_performance_report(metrics: PerformanceMetrics, chart_path: Optional[str] = None) -> bool:
    """Send performance report"""
    return await telegram_bot.send_performance_report(metrics, chart_path)

async def send_error_notification(error_message: str, context: str = "") -> bool:
    """Send error notification"""
    return await telegram_bot.send_error_notification(error_message, context)

async def test_telegram_connection() -> bool:
    """Test Telegram bot connection"""
    return await telegram_bot.test_connection()

def get_telegram_status() -> Dict[str, Any]:
    """Get Telegram bot status"""
    return telegram_bot.get_bot_status()