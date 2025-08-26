"""
XAUUSD Trading Bot - Advanced Logging System
Provides structured logging for all bot components with file rotation
"""
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Add emoji based on level
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': '📊',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        record.emoji = emoji_map.get(record.levelname.strip('\033[0m\033[32m\033[33m\033[31m\033[35m\033[36m'), '📊')
        
        return super().format(record)

class TradingFormatter(logging.Formatter):
    """Specialized formatter for trading events"""
    
    def format(self, record):
        # Add timestamp
        record.timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Categorize log messages
        msg = record.getMessage().lower()
        if 'signal' in msg:
            record.category = '📡'
        elif 'buy' in msg or 'sell' in msg:
            record.category = '💰'
        elif 'error' in msg or 'fail' in msg:
            record.category = '❌'
        elif 'price' in msg:
            record.category = '📊'
        elif 'analysis' in msg:
            record.category = '🔍'
        else:
            record.category = '🤖'
            
        return super().format(record)

class JSONFileHandler(logging.Handler):
    """Custom handler to write structured logs as JSON"""
    
    def __init__(self, filename):
        super().__init__()
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'module': record.name,
                'message': record.getMessage(),
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'signal_data'):
                log_entry['signal_data'] = record.signal_data
            if hasattr(record, 'price_data'):
                log_entry['price_data'] = record.price_data
                
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception:
            self.handleError(record)

class TradingLogger:
    """Main logging manager for the trading bot"""
    
    def __init__(self, name: str = "xauusd_bot", level: str = "INFO", log_to_file: bool = True):
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_to_file = log_to_file
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        if log_to_file:
            self._setup_file_handlers()
    
    def _setup_console_handler(self):
        """Setup colorized console output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        
        # Use colored formatter for console
        console_format = '%(emoji)s %(asctime)s | %(levelname)s | %(name)s | %(message)s'
        console_formatter = ColoredFormatter(
            console_format,
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Setup file handlers with rotation"""
        
        # 1. Main log file with rotation
        main_log_file = self.logs_dir / f'{self.name}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        
        file_format = '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s'
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 2. Trading events log (INFO and above)
        trading_log_file = self.logs_dir / 'trading_events.log'
        trading_handler = logging.handlers.RotatingFileHandler(
            trading_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        trading_handler.setLevel(logging.INFO)
        
        trading_format = '%(category)s %(timestamp)s | %(levelname)-8s | %(message)s'
        trading_formatter = TradingFormatter(trading_format)
        trading_handler.setFormatter(trading_formatter)
        self.logger.addHandler(trading_handler)
        
        # 3. Error log (ERROR and above only)
        error_log_file = self.logs_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # 4. JSON structured log for analysis
        json_log_file = self.logs_dir / f'{self.name}_structured.jsonl'
        json_handler = JSONFileHandler(json_log_file)
        json_handler.setLevel(logging.INFO)
        self.logger.addHandler(json_handler)
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get logger instance for specific module"""
        if module_name:
            return logging.getLogger(f'{self.name}.{module_name}')
        return self.logger
    
    def log_signal(self, signal_data: dict, message: str = "Signal generated"):
        """Log trading signal with structured data"""
        extra = {'signal_data': signal_data}
        self.logger.info(message, extra=extra)
    
    def log_price_update(self, price_data: dict, message: str = "Price updated"):
        """Log price update with structured data"""
        extra = {'price_data': price_data}
        self.logger.debug(message, extra=extra)
    
    def log_performance(self, performance_data: dict, message: str = "Performance update"):
        """Log performance metrics"""
        extra = {'performance_data': performance_data}
        self.logger.info(message, extra=extra)

# Global logger setup
def setup_logger(name: str = "xauusd_bot", level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Setup and return main logger instance"""
    trading_logger = TradingLogger(name, level, log_to_file)
    return trading_logger.get_logger()

def get_module_logger(module_name: str) -> logging.Logger:
    """Get logger for specific module"""
    return logging.getLogger(f"xauusd_bot.{module_name}")

# Create main logger instance
main_logger = setup_logger()

# Export convenience functions
def log_startup():
    """Log bot startup"""
    main_logger.info("🚀 XAUUSD Trading Bot Starting Up")
    main_logger.info("📊 Real-time market analysis enabled")
    main_logger.info("🧠 Self-learning system activated")

def log_signal_sent(direction: str, entry_price: float, score: float):
    """Log when signal is sent"""
    main_logger.info(f"📡 {direction} signal sent: Entry ${entry_price:.2f}, Score {score:.1f}")

def log_data_source(source: str, success: bool, price: Optional[float] = None):
    """Log data source status"""
    if success and price:
        main_logger.info(f"📊 {source}: ✅ ${price:.2f}")
    else:
        main_logger.warning(f"📊 {source}: ❌ Failed")

def log_learning_update(strategy: str, old_weight: float, new_weight: float):
    """Log strategy weight updates"""
    change = ((new_weight - old_weight) / old_weight * 100) if old_weight > 0 else 0
    main_logger.info(f"🧠 {strategy}: {old_weight:.3f} → {new_weight:.3f} ({change:+.1f}%)")

def log_error_with_context(error: Exception, context: str):
    """Log error with additional context"""
    main_logger.error(f"❌ {context}: {str(error)}", exc_info=True)