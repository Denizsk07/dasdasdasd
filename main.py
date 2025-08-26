 #!/usr/bin/env python3
"""XAUUSD Trading Bot - Main Entry Point"""
import asyncio
import sys
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import settings  
from utils.logger import setup_logger, log_startup
from analysis.signal_generator import generate_trading_signal, validate_trading_signal, signal_validator
from bot.telegram_handler import send_trading_signal, send_bot_status
from data import is_market_open

logger = setup_logger()

class XAUUSDTradingBot:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    async def start(self):
        """Start the trading bot"""
        log_startup()
        
        try:
            # Send startup message
            await send_bot_status(
                f"🚀 XAUUSD Trading Bot Started\n"
                f"📊 Analysis Interval: {settings.trading.analysis_interval//60} minutes\n"
                f"🎯 Min Signal Score: {settings.trading.min_signal_score}\n"
                f"💰 Risk per Trade: {settings.trading.risk_percentage}%\n"
                f"📡 TradingView: {'✅ Enabled' if settings.tradingview.enabled else '❌ Disabled'}"
            )
            
            # Schedule signal generation
            self.scheduler.add_job(
                self.analyze_and_signal,
                'interval',
                seconds=settings.trading.analysis_interval,
                id='signal_generation',
                max_instances=1
            )
            
            # Schedule daily report  
            self.scheduler.add_job(
                self.send_daily_report,
                'cron',
                hour=22,
                minute=0,
                id='daily_report'
            )
            
            self.scheduler.start()
            self.is_running = True
            
            logger.info("✅ XAUUSD Trading Bot started successfully")
            
            # Keep running
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("👋 Bot stopped by user")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
            await send_bot_status(f"❌ Bot Error: {str(e)[:100]}")
        finally:
            await self.stop()
    
    async def analyze_and_signal(self):
        """Main analysis and signal generation"""
        try:
            if not is_market_open():
                logger.debug("📴 Market closed - skipping analysis")
                return
            
            logger.info("🔍 Starting market analysis...")
            
            # Generate signal
            signal = await generate_trading_signal()
            
            if signal:
                # Validate signal
                is_valid, reason = validate_trading_signal(signal)
                
                if is_valid:
                    # Send to Telegram
                    success = await send_trading_signal(signal)
                    
                    if success:
                        signal_validator.record_signal_sent()
                        logger.info(f"✅ {signal.direction} signal sent: ${signal.entry_price:.2f} ({signal.confidence:.1f}%)")
                    else:
                        logger.error("❌ Failed to send signal to Telegram")
                else:
                    logger.info(f"🚫 Signal rejected: {reason}")
            else:
                logger.debug("📊 No signal generated")
                
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
    
    async def send_daily_report(self):
        """Send daily performance report"""
        try:
            from trading.risk_manager import risk_manager
            
            stats = risk_manager.get_portfolio_risk()
            
            report = f"""📊 <b>Daily Report</b>
📅 {datetime.now().strftime('%Y-%m-%d')}

📈 <b>Signals Today:</b> {signal_validator.daily_signal_count}
🎯 <b>Daily Limit:</b> {signal_validator.max_daily_signals}
💰 <b>Risk Used:</b> {stats['daily_risk_used']:.1f}%
📊 <b>Open Positions:</b> {stats['open_positions']}

🤖 Bot running normally"""
            
            await send_bot_status(report)
            
        except Exception as e:
            logger.error(f"❌ Daily report error: {e}")
    
    async def stop(self):
        """Stop the bot gracefully"""
        self.is_running = False
        if self.scheduler.running:
            self.scheduler.shutdown()
        
        await send_bot_status("⏹️ XAUUSD Trading Bot stopped")
        logger.info("👋 Bot stopped gracefully")

async def main():
    """Main entry point"""
    print("🚀 XAUUSD Trading Bot - Starting...")
    
    # Validate configuration
    if not settings.telegram.bot_token or not settings.telegram.group_id:
        print("❌ Missing Telegram configuration in .env")
        return
    
    # Start bot
    bot = XAUUSDTradingBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())