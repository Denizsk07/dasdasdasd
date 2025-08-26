 """
XAUUSD Technical Indicators - Comprehensive TA Library
Calculates all technical indicators needed for signal generation
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('technical_indicators')

@dataclass
class IndicatorSignal:
    """Signal from a technical indicator"""
    name: str
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0-1
    value: float
    description: str

class TechnicalIndicators:
    """Main technical indicators calculator"""
    
    def __init__(self):
        self.indicators = {}
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the dataset"""
        if df.empty or len(df) < 50:
            logger.warning("⚠️ Insufficient data for technical indicators")
            return df
        
        logger.debug(f"📊 Calculating indicators for {len(df)} bars")
        
        try:
            # Make a copy to avoid modifying original
            df_indicators = df.copy()
            
            # Moving Averages
            df_indicators = self._add_moving_averages(df_indicators)
            
            # Oscillators
            df_indicators = self._add_oscillators(df_indicators)
            
            # Volatility Indicators
            df_indicators = self._add_volatility_indicators(df_indicators)
            
            # Volume Indicators
            df_indicators = self._add_volume_indicators(df_indicators)
            
            # Support/Resistance Levels
            df_indicators = self._add_support_resistance(df_indicators)
            
            # Custom XAUUSD Indicators
            df_indicators = self._add_xauusd_specific(df_indicators)
            
            logger.debug(f"✅ Added {len(df_indicators.columns) - len(df.columns)} indicators")
            return df_indicators
            
        except Exception as e:
            logger.error(f"❌ Indicator calculation failed: {e}")
            return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        try:
            # Simple Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            
            # Exponential Moving Averages
            df['ema_8'] = ta.ema(df['close'], length=8)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_100'] = ta.ema(df['close'], length=100)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # Weighted Moving Average
            df['wma_20'] = ta.wma(df['close'], length=20)
            
            # Hull Moving Average
            df['hma_20'] = ta.hma(df['close'], length=20)
            
            # VWAP (if volume available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # MA Signals
            df['ma_trend'] = self._calculate_ma_trend(df)
            df['ma_cross_signal'] = self._calculate_ma_crossover_signals(df)
            
        except Exception as e:
            logger.warning(f"⚠️ Moving averages calculation failed: {e}")
        
        return df
    
    def _add_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add oscillator indicators"""
        try:
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_signal'] = self._classify_rsi_signal(df['rsi'] if 'rsi' in df.columns else pd.Series())
            
            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                df['macd'] = macd_result.iloc[:, 0]
                df['macd_signal'] = macd_result.iloc[:, 1] 
                df['macd_histogram'] = macd_result.iloc[:, 2]
                df['macd_trend'] = self._classify_macd_signal(df)
            
            # Stochastic
            stoch_result = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch_result is not None and not stoch_result.empty:
                df['stoch_k'] = stoch_result.iloc[:, 0]
                df['stoch_d'] = stoch_result.iloc[:, 1]
                df['stoch_signal'] = self._classify_stoch_signal(df)
            
            # Williams %R
            df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Commodity Channel Index
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Awesome Oscillator
            df['ao'] = ta.ao(df['high'], df['low'])
            
        except Exception as e:
            logger.warning(f"⚠️ Oscillators calculation failed: {e}")
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        try:
            # Bollinger Bands
            bb_result = ta.bbands(df['close'], length=20, std=2)
            if bb_result is not None and not bb_result.empty:
                df['bb_upper'] = bb_result.iloc[:, 0]
                df['bb_middle'] = bb_result.iloc[:, 1]
                df['bb_lower'] = bb_result.iloc[:, 2]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                df['bb_signal'] = self._classify_bb_signal(df)
            
            # Average True Range
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            # Keltner Channels
            kc_result = ta.kc(df['high'], df['low'], df['close'], length=20)
            if kc_result is not None and not kc_result.empty:
                df['kc_upper'] = kc_result.iloc[:, 0]
                df['kc_middle'] = kc_result.iloc[:, 1]
                df['kc_lower'] = kc_result.iloc[:, 2]
            
            # Donchian Channels
            dc_result = ta.donchian(df['high'], df['low'], length=20)
            if dc_result is not None and not dc_result.empty:
                df['dc_upper'] = dc_result.iloc[:, 0]
                df['dc_middle'] = dc_result.iloc[:, 1] 
                df['dc_lower'] = dc_result.iloc[:, 2]
            
        except Exception as e:
            logger.warning(f"⚠️ Volatility indicators calculation failed: {e}")
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                # Create synthetic volume for XAUUSD if not available
                df['volume'] = self._generate_synthetic_volume(df)
            
            # On Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Volume Weighted Average Price (daily)
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Money Flow Index
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
            
            # Accumulation/Distribution Line
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
            
            # Volume Profile (simplified)
            df['volume_profile'] = self._calculate_volume_profile(df)
            
        except Exception as e:
            logger.warning(f"⚠️ Volume indicators calculation failed: {e}")
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        try:
            # Pivot Points
            pivots = self._calculate_pivot_points(df)
            df = pd.concat([df, pivots], axis=1)
            
            # Dynamic Support/Resistance
            sr_levels = self._calculate_dynamic_sr_levels(df)
            df = pd.concat([df, sr_levels], axis=1)
            
            # Fibonacci Retracements (for recent swing)
            fib_levels = self._calculate_fibonacci_levels(df)
            for level_name, level_value in fib_levels.items():
                df[f'fib_{level_name}'] = level_value
            
        except Exception as e:
            logger.warning(f"⚠️ Support/Resistance calculation failed: {e}")
        
        return df
    
    def _add_xauusd_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add XAUUSD-specific indicators"""
        try:
            # Gold-specific volatility measure
            df['gold_volatility'] = df['atr'].rolling(20).mean() / df['close'].rolling(20).mean() * 100
            
            # Dollar strength indicator (simplified - based on price momentum)
            df['dollar_strength'] = -df['close'].pct_change(20) * 100
            
            # Market session indicator
            df['market_session'] = self._determine_market_sessions(df)
            
            # XAUUSD trend strength
            df['trend_strength'] = self._calculate_trend_strength(df)
            
            # Round number levels (psychological levels)
            df['round_number_distance'] = self._calculate_round_number_distance(df)
            
        except Exception as e:
            logger.warning(f"⚠️ XAUUSD-specific indicators failed: {e}")
        
        return df
    
    def _calculate_ma_trend(self, df: pd.DataFrame) -> pd.Series:
        """Calculate moving average trend"""
        trend = pd.Series(index=df.index, dtype='object')
        
        if all(col in df.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            conditions = [
                (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50']),
                (df['ema_8'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
            ]
            choices = ['BULLISH', 'BEARISH']
            trend = pd.Series(np.select(conditions, choices, 'NEUTRAL'), index=df.index)
        
        return trend
    
    def _calculate_ma_crossover_signals(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MA crossover signals"""
        signals = pd.Series(0, index=df.index)
        
        if all(col in df.columns for col in ['ema_8', 'ema_21']):
            # Golden Cross / Death Cross
            ema8_above_ema21 = df['ema_8'] > df['ema_21']
            cross_up = ema8_above_ema21 & ~ema8_above_ema21.shift(1)
            cross_down = ~ema8_above_ema21 & ema8_above_ema21.shift(1)
            
            signals[cross_up] = 1  # Buy signal
            signals[cross_down] = -1  # Sell signal
        
        return signals
    
    def _classify_rsi_signal(self, rsi: pd.Series) -> pd.Series:
        """Classify RSI signals"""
        signals = pd.Series('NEUTRAL', index=rsi.index)
        
        if not rsi.empty:
            signals[rsi <= 30] = 'OVERSOLD'
            signals[rsi >= 70] = 'OVERBOUGHT'
            signals[(rsi > 50) & (rsi < 70)] = 'BULLISH'
            signals[(rsi < 50) & (rsi > 30)] = 'BEARISH'
        
        return signals
    
    def _classify_macd_signal(self, df: pd.DataFrame) -> pd.Series:
        """Classify MACD signals"""
        signals = pd.Series('NEUTRAL', index=df.index)
        
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'] > 0)
            macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'] < 0)
            
            signals[macd_bullish] = 'BULLISH'
            signals[macd_bearish] = 'BEARISH'
        
        return signals
    
    def _classify_stoch_signal(self, df: pd.DataFrame) -> pd.Series:
        """Classify Stochastic signals"""
        signals = pd.Series('NEUTRAL', index=df.index)
        
        if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
            oversold = (df['stoch_k'] <= 20) & (df['stoch_d'] <= 20)
            overbought = (df['stoch_k'] >= 80) & (df['stoch_d'] >= 80)
            bullish_cross = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
            bearish_cross = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
            
            signals[oversold & bullish_cross] = 'BUY'
            signals[overbought & bearish_cross] = 'SELL'
            signals[oversold] = 'OVERSOLD'
            signals[overbought] = 'OVERBOUGHT'
        
        return signals
    
    def _classify_bb_signal(self, df: pd.DataFrame) -> pd.Series:
        """Classify Bollinger Bands signals"""
        signals = pd.Series('NEUTRAL', index=df.index)
        
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
            # BB position-based signals
            squeeze = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.2)  # Bottom 20% width
            expansion = df['bb_width'] > df['bb_width'].rolling(20).quantile(0.8)  # Top 20% width
            
            at_upper = df['close'] >= df['bb_upper'] * 0.999
            at_lower = df['close'] <= df['bb_lower'] * 1.001
            
            signals[squeeze & (df['bb_position'] > 0.8)] = 'SQUEEZE_HIGH'
            signals[squeeze & (df['bb_position'] < 0.2)] = 'SQUEEZE_LOW'
            signals[at_upper & expansion] = 'OVERBOUGHT'
            signals[at_lower & expansion] = 'OVERSOLD'
        
        return signals
    
    def _generate_synthetic_volume(self, df: pd.DataFrame) -> pd.Series:
        """Generate synthetic volume based on price movement and volatility"""
        base_volume = 1000
        
        # Volume based on price range
        price_range = (df['high'] - df['low']) / df['close']
        range_volume = price_range * 2000
        
        # Volume based on price change
        price_change = abs(df['close'].pct_change())
        change_volume = price_change * 1500
        
        # Add some randomness
        random_factor = np.random.normal(1, 0.1, len(df))
        
        synthetic_volume = (base_volume + range_volume + change_volume) * random_factor
        return pd.Series(synthetic_volume.clip(100, 10000), index=df.index)
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pivot points"""
        pivot_data = pd.DataFrame(index=df.index)
        
        # Use previous day's HLC for pivot calculation
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1) 
        prev_close = df['close'].shift(1)
        
        # Standard Pivot Points
        pivot_data['pivot'] = (prev_high + prev_low + prev_close) / 3
        pivot_data['r1'] = 2 * pivot_data['pivot'] - prev_low
        pivot_data['s1'] = 2 * pivot_data['pivot'] - prev_high
        pivot_data['r2'] = pivot_data['pivot'] + (prev_high - prev_low)
        pivot_data['s2'] = pivot_data['pivot'] - (prev_high - prev_low)
        pivot_data['r3'] = prev_high + 2 * (pivot_data['pivot'] - prev_low)
        pivot_data['s3'] = prev_low - 2 * (prev_high - pivot_data['pivot'])
        
        return pivot_data
    
    def _calculate_dynamic_sr_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic support/resistance levels"""
        sr_data = pd.DataFrame(index=df.index)
        
        # Rolling support/resistance
        period = 20
        sr_data['dynamic_resistance'] = df['high'].rolling(period).max()
        sr_data['dynamic_support'] = df['low'].rolling(period).min()
        
        # Exponential support/resistance
        alpha = 0.1
        sr_data['ema_resistance'] = df['high'].ewm(alpha=alpha).max()
        sr_data['ema_support'] = df['low'].ewm(alpha=alpha).min()
        
        return sr_data
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        if len(df) < lookback:
            return {}
        
        # Find recent swing high and low
        recent_data = df.tail(lookback)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        diff = swing_high - swing_low
        
        return {
            '0': swing_high,
            '236': swing_high - 0.236 * diff,
            '382': swing_high - 0.382 * diff,
            '500': swing_high - 0.500 * diff,
            '618': swing_high - 0.618 * diff,
            '786': swing_high - 0.786 * diff,
            '1000': swing_low
        }
    
    def _determine_market_sessions(self, df: pd.DataFrame) -> pd.Series:
        """Determine market session for each bar"""
        sessions = pd.Series('UNKNOWN', index=df.index)
        
        if isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour
            
            # XAUUSD sessions (UTC times)
            asian = (hours >= 22) | (hours < 8)
            london = (hours >= 8) & (hours < 16)
            newyork = (hours >= 13) & (hours < 22)
            
            sessions[asian] = 'ASIAN'
            sessions[london] = 'LONDON'
            sessions[newyork] = 'NEWYORK'
            sessions[london & newyork] = 'LONDON_NEWYORK'  # Overlap
        
        return sessions
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength (0-100)"""
        trend_strength = pd.Series(50, index=df.index)  # Neutral = 50
        
        if len(df) >= 20:
            # ADX-like calculation
            price_changes = df['close'].diff()
            pos_changes = price_changes.where(price_changes > 0, 0)
            neg_changes = abs(price_changes.where(price_changes < 0, 0))
            
            pos_dm = pos_changes.rolling(14).mean()
            neg_dm = neg_changes.rolling(14).mean()
            
            if 'atr' in df.columns:
                dx = 100 * abs(pos_dm - neg_dm) / (pos_dm + neg_dm)
                trend_strength = dx.rolling(14).mean().fillna(50)
        
        return trend_strength.clip(0, 100)
    
    def _calculate_round_number_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to nearest round number (psychological levels)"""
        round_numbers = [1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300]
        
        distances = pd.Series(index=df.index, dtype=float)
        
        for i, price in df['close'].items():
            nearest_round = min(round_numbers, key=lambda x: abs(x - price))
            distances[i] = abs(price - nearest_round)
        
        return distances

    def get_indicator_signals(self, df: pd.DataFrame) -> List[IndicatorSignal]:
        """Get all current indicator signals"""
        signals = []
        
        if df.empty:
            return signals
        
        latest = df.iloc[-1]
        
        try:
            # RSI Signal
            if 'rsi' in df.columns and not pd.isna(latest['rsi']):
                rsi_val = latest['rsi']
                if rsi_val <= 30:
                    signals.append(IndicatorSignal('RSI', 'BUY', 0.8, rsi_val, f'Oversold at {rsi_val:.1f}'))
                elif rsi_val >= 70:
                    signals.append(IndicatorSignal('RSI', 'SELL', 0.8, rsi_val, f'Overbought at {rsi_val:.1f}'))
            
            # MACD Signal
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                macd_val = latest['macd']
                macd_signal = latest['macd_signal']
                if macd_val > macd_signal and macd_val > 0:
                    signals.append(IndicatorSignal('MACD', 'BUY', 0.7, macd_val, 'Bullish MACD'))
                elif macd_val < macd_signal and macd_val < 0:
                    signals.append(IndicatorSignal('MACD', 'SELL', 0.7, macd_val, 'Bearish MACD'))
            
            # Bollinger Bands Signal
            if 'bb_position' in df.columns and not pd.isna(latest['bb_position']):
                bb_pos = latest['bb_position']
                if bb_pos <= 0.1:
                    signals.append(IndicatorSignal('BB', 'BUY', 0.6, bb_pos, 'At lower Bollinger Band'))
                elif bb_pos >= 0.9:
                    signals.append(IndicatorSignal('BB', 'SELL', 0.6, bb_pos, 'At upper Bollinger Band'))
            
            # Moving Average Trend
            if 'ma_trend' in df.columns and latest['ma_trend'] in ['BULLISH', 'BEARISH']:
                trend = latest['ma_trend']
                direction = 'BUY' if trend == 'BULLISH' else 'SELL'
                signals.append(IndicatorSignal('MA_TREND', direction, 0.5, 0, f'{trend} MA alignment'))
            
        except Exception as e:
            logger.warning(f"⚠️ Error getting indicator signals: {e}")
        
        return signals

# Global indicators calculator
indicators = TechnicalIndicators()

# Convenience function
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe"""
    return indicators.calculate_all_indicators(df)

def get_current_signals(df: pd.DataFrame) -> List[IndicatorSignal]:
    """Get current indicator signals"""
    return indicators.get_indicator_signals(df)
