 """
XAUUSD Market Data Provider - Real-time data from TradingView and Yahoo Finance
Provides authenticated access to live XAUUSD prices and historical data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import asyncio
import time
from dataclasses import dataclass

try:
    from tvdatafeed import TvDatafeed, Interval
    TVDATAFEED_AVAILABLE = True
except ImportError:
    TVDATAFEED_AVAILABLE = False
    print("⚠️ tvdatafeed not available - using Yahoo Finance only")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("❌ yfinance not available - no fallback data source")

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('market_provider')

@dataclass
class PriceData:
    """Current price data structure"""
    symbol: str
    price: float
    timestamp: datetime
    source: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None

@dataclass
class MarketStatus:
    """Market status information"""
    is_open: bool
    next_open: Optional[datetime]
    next_close: Optional[datetime]
    session: str  # "asian", "london", "newyork", "closed"

class TradingViewProvider:
    """TradingView data provider using tvdatafeed"""
    
    def __init__(self):
        self.tv = None
        self.authenticated = False
        self.last_auth_attempt = None
        self.auth_cooldown = 300  # 5 minutes between auth attempts
        
        if TVDATAFEED_AVAILABLE:
            self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with TradingView"""
        try:
            # Check cooldown
            if (self.last_auth_attempt and 
                time.time() - self.last_auth_attempt < self.auth_cooldown):
                return False
            
            self.last_auth_attempt = time.time()
            
            if settings.tradingview.username and settings.tradingview.password:
                logger.info("🔐 Authenticating with TradingView...")
                
                self.tv = TvDatafeed(
                    username=settings.tradingview.username,
                    password=settings.tradingview.password
                )
                
                # Test connection with a simple request
                test_data = self.tv.get_hist(
                    symbol='XAUUSD',
                    exchange='FOREXCOM',
                    interval=Interval.in_1_minute,
                    n_bars=5
                )
                
                if test_data is not None and not test_data.empty:
                    self.authenticated = True
                    logger.info("✅ TradingView authentication successful")
                    return True
                else:
                    logger.warning("❌ TradingView test request failed")
                    
            else:
                logger.warning("⚠️ TradingView credentials missing in .env")
                
        except Exception as e:
            logger.error(f"❌ TradingView authentication failed: {e}")
        
        self.authenticated = False
        return False
    
    def get_current_price(self) -> Optional[PriceData]:
        """Get current XAUUSD price from TradingView"""
        try:
            if not self.authenticated:
                if not self._authenticate():
                    return None
            
            # Get latest 1-minute bar
            data = self.tv.get_hist(
                symbol='XAUUSD',
                exchange='FOREXCOM',
                interval=Interval.in_1_minute,
                n_bars=1
            )
            
            if data is not None and not data.empty:
                latest = data.iloc[-1]
                return PriceData(
                    symbol="XAUUSD",
                    price=float(latest['close']),
                    timestamp=datetime.now(),
                    source="TradingView",
                    volume=float(latest.get('volume', 0))
                )
                
        except Exception as e:
            logger.error(f"❌ TradingView price fetch failed: {e}")
            self.authenticated = False
        
        return None
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from TradingView"""
        try:
            if not self.authenticated:
                if not self._authenticate():
                    return None
            
            # Map timeframe to TradingView interval
            interval_map = {
                '1': Interval.in_1_minute,
                '5': Interval.in_5_minute,
                '15': Interval.in_15_minute,
                '30': Interval.in_30_minute,
                '60': Interval.in_1_hour,
                '240': Interval.in_4_hour,
                '1D': Interval.in_daily
            }
            
            interval = interval_map.get(timeframe, Interval.in_15_minute)
            
            # Get data from TradingView
            data = self.tv.get_hist(
                symbol='XAUUSD',
                exchange='FOREXCOM',
                interval=interval,
                n_bars=bars
            )
            
            if data is not None and not data.empty:
                # Ensure proper column names and data types
                data = data.rename(columns={col: col.lower() for col in data.columns})
                
                # Convert to standard format
                df = pd.DataFrame({
                    'open': data['open'].astype(float),
                    'high': data['high'].astype(float),
                    'low': data['low'].astype(float),
                    'close': data['close'].astype(float),
                    'volume': data['volume'].astype(float) if 'volume' in data.columns else 0
                })
                
                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                logger.info(f"📊 TradingView: {len(df)} bars of {timeframe}min data loaded")
                return df
                
        except Exception as e:
            logger.error(f"❌ TradingView historical data failed: {e}")
            self.authenticated = False
        
        return None

class YahooFinanceProvider:
    """Yahoo Finance fallback data provider"""
    
    def __init__(self):
        self.symbol = settings.yahoo_finance.symbol
        self.session_cache = {}
        self.cache_expiry = 60  # 1 minute cache
    
    def get_current_price(self) -> Optional[PriceData]:
        """Get current XAUUSD price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            
            # Try multiple methods to get current price
            current_price = None
            
            # Method 1: info (most reliable for current price)
            try:
                info = ticker.info
                if 'regularMarketPrice' in info:
                    current_price = float(info['regularMarketPrice'])
                elif 'currentPrice' in info:
                    current_price = float(info['currentPrice'])
                elif 'previousClose' in info:
                    current_price = float(info['previousClose'])
            except:
                pass
            
            # Method 2: recent history if info fails
            if current_price is None:
                try:
                    hist = ticker.history(period="1d", interval="1m")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                except:
                    pass
            
            if current_price and current_price > 0:
                return PriceData(
                    symbol="XAUUSD",
                    price=current_price,
                    timestamp=datetime.now(),
                    source="Yahoo Finance"
                )
                
        except Exception as e:
            logger.error(f"❌ Yahoo Finance price fetch failed: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data from Yahoo Finance"""
        try:
            # Map timeframe to Yahoo Finance interval
            interval_map = {
                '1': '1m',
                '5': '5m', 
                '15': '15m',
                '30': '30m',
                '60': '1h',
                '240': '4h',
                '1D': '1d'
            }
            
            interval = interval_map.get(timeframe, '15m')
            
            # Determine period based on timeframe and bars needed
            if timeframe in ['1', '5']:
                period = '7d'  # 1-5min data limited to 7 days
            elif timeframe in ['15', '30']:
                period = '60d'  # 15-30min data up to 60 days
            else:
                period = '2y'  # Hourly+ data up to 2 years
            
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Take last N bars
                data = data.tail(bars)
                
                # Convert to standard format
                df = pd.DataFrame({
                    'open': data['Open'].astype(float),
                    'high': data['High'].astype(float),
                    'low': data['Low'].astype(float),
                    'close': data['Close'].astype(float),
                    'volume': data['Volume'].astype(float) if 'Volume' in data.columns else 0
                })
                
                logger.info(f"📊 Yahoo Finance: {len(df)} bars of {timeframe}min data loaded")
                return df
                
        except Exception as e:
            logger.error(f"❌ Yahoo Finance historical data failed: {e}")
        
        return None

class MarketDataManager:
    """Main market data manager with failover logic"""
    
    def __init__(self):
        self.providers = {}
        
        # Initialize providers
        if TVDATAFEED_AVAILABLE and settings.tradingview.enabled:
            self.providers['tradingview'] = TradingViewProvider()
            logger.info("📊 TradingView provider initialized")
        
        if YFINANCE_AVAILABLE and settings.yahoo_finance.enabled:
            self.providers['yahoo'] = YahooFinanceProvider()
            logger.info("📊 Yahoo Finance provider initialized")
        
        if not self.providers:
            raise RuntimeError("❌ No data providers available! Check dependencies and configuration.")
        
        # Provider priority (TradingView first, then Yahoo)
        self.provider_priority = ['tradingview', 'yahoo']
        self.current_provider = None
        self.last_price_check = None
        self.price_cache = None
        self.cache_duration = 5  # 5 seconds price cache
    
    def get_current_price(self) -> Optional[PriceData]:
        """Get current price with failover logic"""
        
        # Check cache first
        now = time.time()
        if (self.price_cache and self.last_price_check and 
            now - self.last_price_check < self.cache_duration):
            return self.price_cache
        
        # Try providers in order of priority
        for provider_name in self.provider_priority:
            if provider_name not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_name]
                price_data = provider.get_current_price()
                
                if price_data and self._validate_price(price_data.price):
                    self.current_provider = provider_name
                    self.price_cache = price_data
                    self.last_price_check = now
                    
                    logger.debug(f"📊 Price from {provider_name}: ${price_data.price:.2f}")
                    return price_data
                    
            except Exception as e:
                logger.warning(f"⚠️ {provider_name} failed: {e}")
                continue
        
        logger.error("❌ All price providers failed!")
        return None
    
    def get_historical_data(self, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data with failover logic"""
        
        # Try providers in order of priority
        for provider_name in self.provider_priority:
            if provider_name not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_name]
                data = provider.get_historical_data('XAUUSD', timeframe, bars)
                
                if data is not None and len(data) >= 50:  # Minimum 50 bars required
                    logger.info(f"📊 Historical data from {provider_name}: {len(data)} bars")
                    return self._validate_historical_data(data)
                    
            except Exception as e:
                logger.warning(f"⚠️ {provider_name} historical data failed: {e}")
                continue
        
        logger.error("❌ All historical data providers failed!")
        return None
    
    def _validate_price(self, price: float) -> bool:
        """Validate if price is realistic for XAUUSD"""
        market_config = settings.xauusd_market
        return (market_config.min_realistic_price <= price <= market_config.max_realistic_price)
    
    def _validate_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate historical data"""
        try:
            # Remove any NaN values
            df = df.dropna()
            
            # Ensure OHLC relationships are correct
            df = df[df['high'] >= df['low']]
            df = df[df['high'] >= df['open']]
            df = df[df['high'] >= df['close']]
            df = df[df['low'] <= df['open']]
            df = df[df['low'] <= df['close']]
            
            # Remove extreme outliers (more than 10% price change in one bar)
            df['price_change'] = df['close'].pct_change().abs()
            df = df[df['price_change'] <= 0.1]  # 10% max change
            df = df.drop('price_change', axis=1)
            
            # Validate price ranges
            market_config = settings.xauusd_market
            df = df[
                (df['close'] >= market_config.min_realistic_price) & 
                (df['close'] <= market_config.max_realistic_price)
            ]
            
            logger.debug(f"📊 Data validation: {len(df)} clean bars remaining")
            return df
            
        except Exception as e:
            logger.error(f"❌ Data validation failed: {e}")
            return df
    
    def get_market_status(self) -> MarketStatus:
        """Get current market status for XAUUSD"""
        now = datetime.now()
        
        # XAUUSD trades almost 24/5 (closed Fri 21:00 UTC to Sun 22:00 UTC)
        weekday = now.weekday()
        hour = now.hour
        
        if weekday == 4 and hour >= 21:  # Friday after 21:00
            is_open = False
            session = "closed"
        elif weekday == 5:  # Saturday
            is_open = False
            session = "closed"
        elif weekday == 6 and hour < 22:  # Sunday before 22:00
            is_open = False
            session = "closed"
        else:
            is_open = True
            # Determine session
            if 22 <= hour or hour < 8:
                session = "asian"
            elif 8 <= hour < 16:
                session = "london"
            else:
                session = "newyork"
        
        return MarketStatus(
            is_open=is_open,
            next_open=None,  # Calculate if needed
            next_close=None,
            session=session
        )
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all data providers"""
        status = {}
        
        for name, provider in self.providers.items():
            try:
                if name == 'tradingview':
                    status[name] = {
                        'available': True,
                        'authenticated': provider.authenticated,
                        'last_auth_attempt': provider.last_auth_attempt
                    }
                elif name == 'yahoo':
                    # Test Yahoo with a quick request
                    test_price = provider.get_current_price()
                    status[name] = {
                        'available': True,
                        'working': test_price is not None
                    }
            except Exception as e:
                status[name] = {
                    'available': False,
                    'error': str(e)
                }
        
        status['current_provider'] = self.current_provider
        return status

# Global market data manager instance
market_data = MarketDataManager()

# Convenience functions for easy access
async def get_current_price() -> Optional[float]:
    """Get current XAUUSD price (async wrapper)"""
    price_data = market_data.get_current_price()
    return price_data.price if price_data else None

async def get_historical_data(timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
    """Get historical XAUUSD data (async wrapper)"""
    return market_data.get_historical_data(timeframe, bars)

def is_market_open() -> bool:
    """Check if XAUUSD market is currently open"""
    return market_data.get_market_status().is_open
