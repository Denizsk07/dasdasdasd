"""
XAUUSD Trading Bot - Utility Helper Functions
Common utility functions used across the trading bot
"""
import os
import time
import hashlib
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import requests
from functools import wraps, lru_cache
import pickle

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('helpers')

# =============================================================================
# TIME AND DATE UTILITIES
# =============================================================================

def get_current_utc() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)

def get_market_time() -> datetime:
    """Get current market time (UTC)"""
    return get_current_utc()

def format_timestamp(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Format datetime to string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime(format_str)

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime"""
    try:
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S UTC",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        logger.warning(f"⚠️ Could not parse timestamp: {timestamp_str}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Timestamp parsing error: {e}")
        return None

def get_trading_session(dt: datetime = None) -> str:
    """Determine current trading session"""
    if dt is None:
        dt = get_current_utc()
    
    hour = dt.hour
    
    # XAUUSD trading sessions (UTC)
    if 22 <= hour or hour < 8:
        return 'asian'
    elif 8 <= hour < 16:
        return 'london' 
    elif 16 <= hour < 22:
        return 'newyork'
    else:
        return 'unknown'

def is_weekend() -> bool:
    """Check if current time is weekend"""
    now = get_current_utc()
    # Weekend: Friday 21:00 UTC to Sunday 22:00 UTC
    weekday = now.weekday()
    hour = now.hour
    
    if weekday == 4 and hour >= 21:  # Friday after 21:00
        return True
    elif weekday == 5:  # Saturday
        return True
    elif weekday == 6 and hour < 22:  # Sunday before 22:00
        return True
    
    return False

def time_until_market_open() -> Optional[timedelta]:
    """Calculate time until market opens"""
    if not is_weekend():
        return timedelta(0)  # Market is open
    
    now = get_current_utc()
    
    # Find next Sunday 22:00 UTC
    days_until_sunday = (6 - now.weekday()) % 7
    if days_until_sunday == 0 and now.hour >= 22:  # Already past Sunday 22:00
        days_until_sunday = 7
    
    next_open = now.replace(hour=22, minute=0, second=0, microsecond=0) + timedelta(days=days_until_sunday)
    
    return next_open - now

# =============================================================================
# DATA VALIDATION AND CLEANING
# =============================================================================

def validate_ohlc_data(df: pd.DataFrame) -> bool:
    """Validate OHLC data integrity"""
    try:
        if df.empty:
            return False
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check OHLC relationships
        invalid_bars = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['high'] < df['low']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['high'])
        )
        
        invalid_count = invalid_bars.sum()
        invalid_ratio = invalid_count / len(df)
        
        # Allow up to 1% invalid bars (data errors)
        return invalid_ratio <= 0.01
        
    except Exception as e:
        logger.error(f"❌ OHLC validation error: {e}")
        return False

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate price data"""
    try:
        cleaned_df = df.copy()
        
        # Remove NaN values
        cleaned_df = cleaned_df.dropna()
        
        # Remove zero or negative prices
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[col] > 0]
        
        # Remove extreme price changes (> 10% per bar)
        if len(cleaned_df) > 1:
            price_changes = cleaned_df['close'].pct_change().abs()
            cleaned_df = cleaned_df[price_changes <= 0.1]
        
        # Validate OHLC relationships
        if not validate_ohlc_data(cleaned_df):
            logger.warning("⚠️ OHLC validation failed after cleaning")
        
        logger.debug(f"📊 Cleaned data: {len(df)} → {len(cleaned_df)} bars")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"❌ Data cleaning error: {e}")
        return df

def detect_price_outliers(prices: Union[pd.Series, List[float]], threshold: float = 3.0) -> List[int]:
    """Detect price outliers using Z-score method"""
    try:
        if isinstance(prices, list):
            prices = pd.Series(prices)
        
        if len(prices) < 10:
            return []
        
        z_scores = np.abs((prices - prices.mean()) / prices.std())
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        
        return outlier_indices
        
    except Exception as e:
        logger.error(f"❌ Outlier detection error: {e}")
        return []

# =============================================================================
# PRICE CALCULATIONS
# =============================================================================

def calculate_pips(price1: float, price2: float, instrument: str = 'XAUUSD') -> float:
    """Calculate pip difference between two prices"""
    try:
        if instrument.upper() == 'XAUUSD':
            # For XAUUSD, 1 pip = 0.1 (10 cents)
            pip_value = 0.1
        else:
            # Default forex pip value
            pip_value = 0.0001
        
        return (price2 - price1) / pip_value
        
    except Exception as e:
        logger.error(f"❌ Pip calculation error: {e}")
        return 0.0

def calculate_position_value(price: float, lots: float, instrument: str = 'XAUUSD') -> float:
    """Calculate position value in USD"""
    try:
        if instrument.upper() == 'XAUUSD':
            # 1 lot = 100 oz of gold
            return price * lots * 100
        else:
            # Standard forex lot size
            return lots * 100000
            
    except Exception as e:
        logger.error(f"❌ Position value calculation error: {e}")
        return 0.0

def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calculate risk:reward ratio"""
    try:
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
        
    except Exception as e:
        logger.error(f"❌ R:R calculation error: {e}")
        return 0.0

def normalize_price(price: float, decimals: int = 2) -> float:
    """Normalize price to specified decimal places"""
    try:
        return round(price, decimals)
    except Exception as e:
        logger.error(f"❌ Price normalization error: {e}")
        return price

# =============================================================================
# FILE AND DATA UTILITIES
# =============================================================================

def safe_file_read(file_path: Union[str, Path], default=None) -> Any:
    """Safely read file with error handling"""
    try:
        path = Path(file_path)
        
        if not path.exists():
            logger.debug(f"📁 File not found: {path}")
            return default
        
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.suffix.lower() == '.pkl':
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        logger.error(f"❌ File read error {file_path}: {e}")
        return default

def safe_file_write(file_path: Union[str, Path], data: Any, backup: bool = True) -> bool:
    """Safely write file with backup"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        if backup and path.exists():
            backup_path = path.with_suffix(f'{path.suffix}.backup')
            try:
                path.rename(backup_path)
            except:
                pass  # Backup failed, continue anyway
        
        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif path.suffix.lower() == '.pkl':
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File write error {file_path}: {e}")
        return False

def get_file_age(file_path: Union[str, Path]) -> Optional[timedelta]:
    """Get file age as timedelta"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        file_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return get_current_utc() - file_time
        
    except Exception as e:
        logger.error(f"❌ File age calculation error: {e}")
        return None

def cleanup_old_files(directory: Union[str, Path], max_age_days: int = 30, pattern: str = "*") -> int:
    """Clean up old files in directory"""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        cutoff_time = get_current_utc() - timedelta(days=max_age_days)
        deleted_count = 0
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Could not delete file {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"🗑️ Cleaned up {deleted_count} old files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ File cleanup error: {e}")
        return 0

# =============================================================================
# PERFORMANCE AND CACHING
# =============================================================================

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⏱️ {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def async_timing_decorator(func):
    """Async version of timing decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⏱️ {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"❌ {func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator

async def async_retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Async version of retry decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"❌ {func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# HASH AND SECURITY UTILITIES
# =============================================================================

def generate_hash(data: str, algorithm: str = 'sha256') -> str:
    """Generate hash of data"""
    try:
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode('utf-8'))
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"❌ Hash generation error: {e}")
        return ""

def generate_signal_id(timestamp: datetime, entry_price: float, direction: str) -> str:
    """Generate unique signal ID"""
    try:
        data = f"{timestamp.isoformat()}{entry_price}{direction}"
        hash_val = generate_hash(data, 'md5')
        return f"{direction}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash_val[:8]}"
    except Exception as e:
        logger.error(f"❌ Signal ID generation error: {e}")
        return f"SIGNAL_{int(time.time())}"

def validate_env_vars() -> Dict[str, bool]:
    """Validate required environment variables"""
    required_vars = {
        'TELEGRAM_BOT_TOKEN': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
        'TELEGRAM_GROUP_ID': bool(os.getenv('TELEGRAM_GROUP_ID')),
        'TV_USERNAME': bool(os.getenv('TV_USERNAME')),
        'TV_PASSWORD': bool(os.getenv('TV_PASSWORD'))
    }
    
    missing_vars = [var for var, present in required_vars.items() if not present]
    
    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
    
    return required_vars

# =============================================================================
# NETWORK AND API UTILITIES
# =============================================================================

def check_internet_connection(url: str = "https://www.google.com", timeout: int = 5) -> bool:
    """Check if internet connection is available"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def safe_api_request(url: str, method: str = 'GET', timeout: int = 30, **kwargs) -> Optional[requests.Response]:
    """Make safe API request with error handling"""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        logger.error(f"❌ API request failed {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error in API request: {e}")
        return None

# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero handling"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    try:
        if old_value == 0:
            return 0.0 if new_value == 0 else 100.0
        return ((new_value - old_value) / abs(old_value)) * 100
    except Exception:
        return 0.0

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def round_to_significant_digits(value: float, digits: int = 3) -> float:
    """Round to significant digits"""
    try:
        if value == 0:
            return 0
        return round(value, -int(np.floor(np.log10(abs(value)))) + (digits - 1))
    except:
        return value

# =============================================================================
# STRING AND FORMAT UTILITIES
# =============================================================================

def format_currency(amount: float, currency: str = 'USD', decimals: int = 2) -> str:
    """Format amount as currency"""
    try:
        symbol = ' if currency == 'USD' else currency
        return f"{symbol}{amount:,.{decimals}f}"
    except:
        return f"{amount:.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage"""
    try:
        return f"{value:.{decimals}f}%"
    except:
        return f"{value}%"

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_string(text: str) -> str:
    """Clean string for safe usage"""
    try:
        # Remove control characters and excessive whitespace
        cleaned = ''.join(char for char in text if ord(char) >= 32)
        cleaned = ' '.join(cleaned.split())
        return cleaned
    except:
        return str(text)

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config_with_defaults(config_file: Union[str, Path], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load configuration with default fallbacks"""
    try:
        config = safe_file_read(config_file, {})
        if not isinstance(config, dict):
            config = {}
        
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        return config
        
    except Exception as e:
        logger.error(f"❌ Config loading error: {e}")
        return defaults

def update_config(config_file: Union[str, Path], updates: Dict[str, Any]) -> bool:
    """Update configuration file"""
    try:
        config = safe_file_read(config_file, {})
        if not isinstance(config, dict):
            config = {}
        
        config.update(updates)
        return safe_file_write(config_file, config)
        
    except Exception as e:
        logger.error(f"❌ Config update error: {e}")
        return False

# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'uptime': time.time() - psutil.boot_time()
        }
    except Exception as e:
        logger.error(f"❌ System info error: {e}")
        return {'error': str(e)}

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    dependencies = {
        'pandas': False,
        'numpy': False,
        'requests': False,
        'matplotlib': False,
        'mplfinance': False,
        'pandas_ta': False,
        'yfinance': False,
        'python-telegram-bot': False,
        'APScheduler': False,
        'python-dotenv': False
    }
    
    for dep in dependencies:
        try:
            if dep == 'python-telegram-bot':
                import telegram
            elif dep == 'mplfinance':
                import mplfinance
            elif dep == 'pandas_ta':
                import pandas_ta
            elif dep == 'yfinance':
                import yfinance
            elif dep == 'APScheduler':
                import apscheduler
            elif dep == 'python-dotenv':
                import dotenv
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def setup_directories():
    """Setup required directories"""
    directories = [
        settings.storage.base_dir,
        settings.storage.charts_dir,
        settings.storage.logs_dir
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"❌ Directory setup error {directory}: {e}")

def is_trading_hours() -> bool:
    """Check if within trading hours"""
    return not is_weekend()

def get_next_analysis_time(interval_minutes: int = None) -> datetime:
    """Get next analysis time"""
    if interval_minutes is None:
        interval_minutes = settings.trading.analysis_interval // 60
    
    now = get_current_utc()
    next_time = now + timedelta(minutes=interval_minutes)
    return next_time

# Export commonly used functions
__all__ = [
    'get_current_utc', 'get_market_time', 'format_timestamp', 'parse_timestamp',
    'get_trading_session', 'is_weekend', 'time_until_market_open',
    'validate_ohlc_data', 'clean_price_data', 'detect_price_outliers',
    'calculate_pips', 'calculate_position_value', 'calculate_risk_reward_ratio',
    'normalize_price', 'safe_file_read', 'safe_file_write', 'get_file_age',
    'cleanup_old_files', 'timing_decorator', 'async_timing_decorator',
    'retry_on_failure', 'async_retry_on_failure', 'generate_hash',
    'generate_signal_id', 'validate_env_vars', 'check_internet_connection',
    'safe_api_request', 'safe_divide', 'calculate_percentage_change',
    'clamp', 'round_to_significant_digits', 'format_currency',
    'format_percentage', 'truncate_string', 'clean_string',
    'load_config_with_defaults', 'update_config', 'get_system_info',
    'check_dependencies', 'setup_directories', 'is_trading_hours',
    'get_next_analysis_time'
]