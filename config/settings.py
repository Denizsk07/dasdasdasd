 
"""
XAUUSD Trading Bot - Central Configuration
Loads settings from .env and provides validated configuration objects
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# Load environment variables
load_dotenv()

@dataclass
class TelegramConfig:
    """Telegram Bot Configuration"""
    bot_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    group_id: str = os.getenv('TELEGRAM_GROUP_ID', '')
    
    def __post_init__(self):
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required in .env")
        if not self.group_id:
            raise ValueError("TELEGRAM_GROUP_ID is required in .env")

@dataclass
class TradingViewConfig:
    """TradingView Data Source Configuration"""
    username: str = os.getenv('TV_USERNAME', '')
    password: str = os.getenv('TV_PASSWORD', '')
    enabled: bool = True
    
    def __post_init__(self):
        if not self.username or not self.password:
            print("⚠️ TradingView credentials missing - using Yahoo Finance only")
            self.enabled = False

@dataclass
class YahooFinanceConfig:
    """Yahoo Finance Fallback Configuration"""
    symbol: str = os.getenv('YF_SYMBOL', 'XAUUSD=X')
    enabled: bool = True

@dataclass
class TradingConfig:
    """Core Trading Configuration"""
    # Symbol Settings
    primary_symbol: str = os.getenv('PRIMARY_SYMBOL', 'XAUUSD')
    timeframes: List[str] = field(default_factory=lambda: os.getenv('TIMEFRAMES', '15,30,60').split(','))
    
    # Risk Management
    risk_percentage: float = float(os.getenv('RISK_PERCENTAGE', '2'))
    min_signal_score: float = float(os.getenv('MIN_SIGNAL_SCORE', '75'))
    stop_loss_pips: float = float(os.getenv('STOP_LOSS_PIPS', '30'))
    tp_levels: List[float] = field(default_factory=lambda: [float(x) for x in os.getenv('TP_LEVELS', '20,40,60,100').split(',')])
    
    # Bot Behavior
    analysis_interval: int = int(os.getenv('ANALYSIS_INTERVAL', '300'))  # seconds
    max_signals_per_day: int = int(os.getenv('MAX_SIGNALS_PER_DAY', '10'))
    min_hours_between_signals: int = int(os.getenv('MIN_HOURS_BETWEEN_SIGNALS', '2'))
    
    def __post_init__(self):
        # Validate risk percentage
        if self.risk_percentage > 5.0:
            raise ValueError("Risk percentage cannot exceed 5%")
        
        # Validate timeframes
        valid_timeframes = ['1', '5', '15', '30', '60', '240', '1D']
        for tf in self.timeframes:
            if tf not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {tf}")

@dataclass
class StrategyWeights:
    """Strategy Weight Configuration - Will be updated by learning system"""
    bollinger: float = 0.15
    volume: float = 0.10
    price_action: float = 0.15
    smc: float = 0.20  # Smart Money Concepts
    patterns: float = 0.12
    candlesticks: float = 0.08
    fvg: float = 0.10   # Fair Value Gaps
    support_resistance: float = 0.10
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'bollinger': self.bollinger,
            'volume': self.volume,
            'price_action': self.price_action,
            'smc': self.smc,
            'patterns': self.patterns,
            'candlesticks': self.candlesticks,
            'fvg': self.fvg,
            'support_resistance': self.support_resistance
        }
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = sum(self.to_dict().values())
        if total > 0:
            factor = 1.0 / total
            self.bollinger *= factor
            self.volume *= factor
            self.price_action *= factor
            self.smc *= factor
            self.patterns *= factor
            self.candlesticks *= factor
            self.fvg *= factor
            self.support_resistance *= factor

@dataclass
class StorageConfig:
    """File Storage Configuration"""
    base_dir: Path = field(default_factory=lambda: Path('storage'))
    charts_dir: Path = field(default_factory=lambda: Path('storage/charts'))
    logs_dir: Path = field(default_factory=lambda: Path('logs'))
    
    trades_file: Path = field(default_factory=lambda: Path('storage/trades.json'))
    performance_file: Path = field(default_factory=lambda: Path('storage/performance.json'))
    weights_file: Path = field(default_factory=lambda: Path('storage/strategy_weights.json'))
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.base_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

@dataclass
class LoggingConfig:
    """Logging Configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_to_file: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    max_log_files: int = 10
    max_log_size_mb: int = 50

@dataclass
class XAUUSDMarketConfig:
    """XAUUSD Specific Market Configuration"""
    # Market characteristics for XAUUSD
    typical_spread: float = 0.3  # pips
    pip_value: float = 0.01      # $0.01 per pip for standard lot
    lot_size: float = 100        # 100 oz per standard lot
    
    # Trading hours (UTC) - Gold trades almost 24/5
    market_open_sunday: str = "22:00"
    market_close_friday: str = "21:00"
    
    # Price validation ranges
    min_realistic_price: float = 1500.0
    max_realistic_price: float = 4000.0
    max_price_change_percent: float = 5.0  # Max 5% price change per hour

class Settings:
    """Main Settings Container"""
    
    def __init__(self):
        # Load all configuration sections
        self.telegram = TelegramConfig()
        self.tradingview = TradingViewConfig()
        self.yahoo_finance = YahooFinanceConfig()
        self.trading = TradingConfig()
        self.strategy_weights = StrategyWeights()
        self.storage = StorageConfig()
        self.logging = LoggingConfig()
        self.xauusd_market = XAUUSDMarketConfig()
        
        # Normalize strategy weights
        self.strategy_weights.normalize()
        
        # Print startup configuration
        self._print_startup_info()
    
    def _print_startup_info(self):
        """Print key configuration on startup"""
        print("🔧 XAUUSD Trading Bot Configuration Loaded")
        print(f"📊 Symbol: {self.trading.primary_symbol}")
        print(f"⏱️ Timeframes: {', '.join(self.trading.timeframes)}")
        print(f"💰 Risk per trade: {self.trading.risk_percentage}%")
        print(f"🎯 Min signal score: {self.trading.min_signal_score}")
        print(f"📈 TP levels: {self.trading.tp_levels}")
        print(f"🛑 Stop loss: {self.trading.stop_loss_pips} pips")
        print(f"📡 TradingView: {'✅ Enabled' if self.tradingview.enabled else '❌ Disabled (check credentials)'}")
        print(f"📊 Yahoo Finance: {'✅ Enabled' if self.yahoo_finance.enabled else '❌ Disabled'}")
        print(f"📱 Telegram Group: {self.telegram.group_id}")

# Global settings instance
settings = Settings()

# Export commonly used configs
TELEGRAM_CONFIG = settings.telegram
TRADING_CONFIG = settings.trading
STORAGE_CONFIG = settings.storage