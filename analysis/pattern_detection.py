"""
XAUUSD Pattern Detection - Chart Patterns & Candlestick Analysis
Detects classic chart patterns and Japanese candlestick formations
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('pattern_detection')

@dataclass
class ChartPattern:
    """Chart pattern detection result"""
    name: str
    type: str           # 'reversal', 'continuation', 'breakout'
    direction: str      # 'bullish', 'bearish', 'neutral'
    confidence: float   # 0-1
    start_index: int
    end_index: int
    key_levels: List[float]
    description: str
    target: Optional[float] = None
    stop_loss: Optional[float] = None

@dataclass 
class CandlestickPattern:
    """Candlestick pattern result"""
    name: str
    type: str          # 'reversal', 'continuation', 'doji', 'hammer'
    direction: str     # 'bullish', 'bearish', 'neutral'
    strength: float    # 0-1
    index: int
    description: str
    reliability: str   # 'high', 'medium', 'low'

class ChartPatternDetector:
    """Detects classic chart patterns"""
    
    def __init__(self):
        self.min_pattern_bars = 10  # Minimum bars for pattern
        self.max_pattern_bars = 50  # Maximum bars for pattern
        self.tolerance = 0.02       # 2% price tolerance for pattern matching
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect all chart patterns"""
        if len(df) < self.min_pattern_bars:
            return []
        
        patterns = []
        
        try:
            # Head and Shoulders patterns
            patterns.extend(self._detect_head_shoulders(df))
            patterns.extend(self._detect_inverse_head_shoulders(df))
            
            # Triangle patterns
            patterns.extend(self._detect_ascending_triangles(df))
            patterns.extend(self._detect_descending_triangles(df))
            patterns.extend(self._detect_symmetrical_triangles(df))
            
            # Double patterns
            patterns.extend(self._detect_double_tops(df))
            patterns.extend(self._detect_double_bottoms(df))
            
            # Flag and Pennant patterns
            patterns.extend(self._detect_bull_flags(df))
            patterns.extend(self._detect_bear_flags(df))
            
            # Channel patterns
            patterns.extend(self._detect_channels(df))
            
            # Wedge patterns
            patterns.extend(self._detect_wedges(df))
            
            logger.debug(f"🔍 Detected {len(patterns)} chart patterns")
            
        except Exception as e:
            logger.warning(f"⚠️ Pattern detection error: {e}")
        
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Head and Shoulders reversal pattern"""
        patterns = []
        
        # Find peaks (potential shoulders and head)
        peaks = self._find_peaks(df['high'])
        
        if len(peaks) < 3:
            return patterns
        
        # Look for H&S pattern in recent data
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Check if middle peak is highest (head)
            if (head['price'] > left_shoulder['price'] and 
                head['price'] > right_shoulder['price']):
                
                # Check if shoulders are approximately equal
                shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
                avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
                
                if shoulder_diff / avg_shoulder_price <= self.tolerance:
                    # Find neckline (lows between shoulders and head)
                    neckline_low = self._find_neckline_low(df, left_shoulder['index'], right_shoulder['index'])
                    
                    if neckline_low:
                        # Calculate pattern metrics
                        pattern_height = head['price'] - neckline_low
                        target = neckline_low - pattern_height
                        
                        patterns.append(ChartPattern(
                            name="Head and Shoulders",
                            type="reversal",
                            direction="bearish",
                            confidence=self._calculate_hs_confidence(df, left_shoulder, head, right_shoulder, neckline_low),
                            start_index=left_shoulder['index'],
                            end_index=right_shoulder['index'],
                            key_levels=[left_shoulder['price'], head['price'], right_shoulder['price'], neckline_low],
                            description=f"H&S: Head ${head['price']:.2f}, Neckline ${neckline_low:.2f}",
                            target=target,
                            stop_loss=head['price']
                        ))
        
        return patterns
    
    def _detect_inverse_head_shoulders(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Inverse Head and Shoulders pattern"""
        patterns = []
        
        # Find valleys (potential shoulders and head)
        valleys = self._find_valleys(df['low'])
        
        if len(valleys) < 3:
            return patterns
        
        for i in range(len(valleys) - 2):
            left_shoulder = valleys[i]
            head = valleys[i + 1]
            right_shoulder = valleys[i + 2]
            
            # Check if middle valley is lowest (head)
            if (head['price'] < left_shoulder['price'] and 
                head['price'] < right_shoulder['price']):
                
                # Check if shoulders are approximately equal
                shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
                avg_shoulder_price = (left_shoulder['price'] + right_shoulder['price']) / 2
                
                if shoulder_diff / avg_shoulder_price <= self.tolerance:
                    # Find neckline (highs between shoulders and head)
                    neckline_high = self._find_neckline_high(df, left_shoulder['index'], right_shoulder['index'])
                    
                    if neckline_high:
                        pattern_height = neckline_high - head['price']
                        target = neckline_high + pattern_height
                        
                        patterns.append(ChartPattern(
                            name="Inverse Head and Shoulders",
                            type="reversal",
                            direction="bullish",
                            confidence=self._calculate_hs_confidence(df, left_shoulder, head, right_shoulder, neckline_high),
                            start_index=left_shoulder['index'],
                            end_index=right_shoulder['index'],
                            key_levels=[left_shoulder['price'], head['price'], right_shoulder['price'], neckline_high],
                            description=f"Inv H&S: Head ${head['price']:.2f}, Neckline ${neckline_high:.2f}",
                            target=target,
                            stop_loss=head['price']
                        ))
        
        return patterns
    
    def _detect_double_tops(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Double Top reversal pattern"""
        patterns = []
        
        peaks = self._find_peaks(df['high'])
        if len(peaks) < 2:
            return patterns
        
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Check if peaks are approximately equal
            price_diff = abs(peak1['price'] - peak2['price'])
            avg_price = (peak1['price'] + peak2['price']) / 2
            
            if price_diff / avg_price <= self.tolerance:
                # Find the valley between peaks
                valley_start = peak1['index']
                valley_end = peak2['index']
                valley_section = df.iloc[valley_start:valley_end + 1]
                valley_low = valley_section['low'].min()
                valley_idx = valley_section['low'].idxmin()
                
                # Ensure valley is significantly lower than peaks
                valley_depth = (avg_price - valley_low) / avg_price
                
                if valley_depth >= 0.03:  # At least 3% depth
                    target = valley_low - (avg_price - valley_low)
                    
                    patterns.append(ChartPattern(
                        name="Double Top",
                        type="reversal",
                        direction="bearish",
                        confidence=min(0.9, 0.5 + valley_depth),
                        start_index=peak1['index'],
                        end_index=peak2['index'],
                        key_levels=[peak1['price'], peak2['price'], valley_low],
                        description=f"Double Top: ${avg_price:.2f}, Support ${valley_low:.2f}",
                        target=target,
                        stop_loss=avg_price + 5  # $5 above pattern high
                    ))
        
        return patterns
    
    def _detect_double_bottoms(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Double Bottom reversal pattern"""
        patterns = []
        
        valleys = self._find_valleys(df['low'])
        if len(valleys) < 2:
            return patterns
        
        for i in range(len(valleys) - 1):
            valley1 = valleys[i]
            valley2 = valleys[i + 1]
            
            # Check if valleys are approximately equal
            price_diff = abs(valley1['price'] - valley2['price'])
            avg_price = (valley1['price'] + valley2['price']) / 2
            
            if price_diff / avg_price <= self.tolerance:
                # Find the peak between valleys
                peak_start = valley1['index']
                peak_end = valley2['index']
                peak_section = df.iloc[peak_start:peak_end + 1]
                peak_high = peak_section['high'].max()
                
                # Ensure peak is significantly higher than valleys
                peak_height = (peak_high - avg_price) / avg_price
                
                if peak_height >= 0.03:  # At least 3% height
                    target = peak_high + (peak_high - avg_price)
                    
                    patterns.append(ChartPattern(
                        name="Double Bottom",
                        type="reversal",
                        direction="bullish",
                        confidence=min(0.9, 0.5 + peak_height),
                        start_index=valley1['index'],
                        end_index=valley2['index'],
                        key_levels=[valley1['price'], valley2['price'], peak_high],
                        description=f"Double Bottom: ${avg_price:.2f}, Resistance ${peak_high:.2f}",
                        target=target,
                        stop_loss=avg_price - 5  # $5 below pattern low
                    ))
        
        return patterns
    
    def _detect_ascending_triangles(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Ascending Triangle continuation pattern"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Look for horizontal resistance and rising support
        for window_size in [20, 30, 40]:
            if len(df) < window_size:
                continue
                
            for start in range(len(df) - window_size):
                end = start + window_size
                window_data = df.iloc[start:end]
                
                # Find peaks and valleys in window
                peaks = self._find_peaks_in_window(window_data['high'])
                valleys = self._find_valleys_in_window(window_data['low'])
                
                if len(peaks) >= 2 and len(valleys) >= 2:
                    # Check for horizontal resistance (flat top)
                    peak_prices = [p['price'] for p in peaks]
                    resistance_level = np.mean(peak_prices)
                    peak_variance = np.var(peak_prices) / resistance_level
                    
                    # Check for rising support
                    valley_indices = [v['index'] for v in valleys]
                    valley_prices = [v['price'] for v in valleys]
                    
                    if len(valley_indices) >= 2:
                        support_slope = np.polyfit(valley_indices, valley_prices, 1)[0]
                        
                        # Pattern criteria
                        if (peak_variance < 0.001 and  # Flat resistance
                            support_slope > 0 and      # Rising support
                            resistance_level > valley_prices[-1]):  # Resistance above support
                            
                            patterns.append(ChartPattern(
                                name="Ascending Triangle",
                                type="continuation",
                                direction="bullish",
                                confidence=0.7,
                                start_index=start,
                                end_index=end - 1,
                                key_levels=[resistance_level, valley_prices[-1]],
                                description=f"Ascending Triangle: Resistance ${resistance_level:.2f}",
                                target=resistance_level + (resistance_level - valley_prices[0]),
                                stop_loss=valley_prices[-1] - 5
                            ))
        
        return patterns
    
    def _detect_bull_flags(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect Bull Flag continuation pattern"""
        patterns = []
        
        if len(df) < 15:
            return patterns
        
        # Look for strong upward move followed by consolidation
        for i in range(10, len(df) - 5):
            # Check for strong bullish move (flagpole)
            lookback = min(10, i)
            flagpole_start = i - lookback
            flagpole_data = df.iloc[flagpole_start:i]
            
            price_change = (df.iloc[i-1]['close'] - flagpole_data['close'].iloc[0]) / flagpole_data['close'].iloc[0]
            
            if price_change > 0.02:  # At least 2% move up
                # Check for consolidation (flag)
                flag_end = min(i + 8, len(df) - 1)
                flag_data = df.iloc[i:flag_end]
                
                if len(flag_data) >= 5:
                    # Flag should be sideways/slightly down
                    flag_slope = self._calculate_trend_slope(flag_data['close'])
                    flag_volatility = flag_data['close'].std() / flag_data['close'].mean()
                    
                    if flag_slope >= -0.1 and flag_volatility < 0.02:  # Sideways consolidation
                        patterns.append(ChartPattern(
                            name="Bull Flag",
                            type="continuation", 
                            direction="bullish",
                            confidence=0.6 + min(0.3, price_change * 5),  # Stronger move = higher confidence
                            start_index=flagpole_start,
                            end_index=flag_end - 1,
                            key_levels=[flagpole_data['close'].iloc[0], df.iloc[i-1]['close'], flag_data['close'].iloc[-1]],
                            description=f"Bull Flag: Flagpole +{price_change:.1%}",
                            target=flag_data['close'].iloc[-1] + (df.iloc[i-1]['close'] - flagpole_data['close'].iloc[0]),
                            stop_loss=flag_data['low'].min() - 3
                        ))
        
        return patterns
    
    def _detect_channels(self, df: pd.DataFrame) -> List[ChartPattern]:
        """Detect price channels"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Look for parallel trend lines
        for window_size in [30, 40, 50]:
            if len(df) < window_size:
                continue
                
            for start in range(len(df) - window_size):
                end = start + window_size
                window_data = df.iloc[start:end]
                
                # Calculate trend lines for highs and lows
                indices = np.arange(len(window_data))
                
                try:
                    # Upper trend line (resistance)
                    upper_slope, upper_intercept = np.polyfit(indices, window_data['high'], 1)
                    upper_line = upper_slope * indices + upper_intercept
                    
                    # Lower trend line (support)
                    lower_slope, lower_intercept = np.polyfit(indices, window_data['low'], 1)
                    lower_line = lower_slope * indices + lower_intercept
                    
                    # Check if lines are roughly parallel
                    slope_diff = abs(upper_slope - lower_slope)
                    
                    if slope_diff < 0.5:  # Parallel enough
                        # Calculate how well price respects the channel
                        upper_touches = sum(1 for i, price in enumerate(window_data['high']) 
                                          if abs(price - upper_line[i]) < 2)  # Within $2
                        lower_touches = sum(1 for i, price in enumerate(window_data['low']) 
                                          if abs(price - lower_line[i]) < 2)
                        
                        total_touches = upper_touches + lower_touches
                        
                        if total_touches >= 4:  # At least 4 touches
                            direction = "bullish" if upper_slope > 0.1 else ("bearish" if upper_slope < -0.1 else "neutral")
                            channel_type = "ascending" if upper_slope > 0.1 else ("descending" if upper_slope < -0.1 else "horizontal")
                            
                            patterns.append(ChartPattern(
                                name=f"{channel_type.title()} Channel",
                                type="continuation",
                                direction=direction,
                                confidence=min(0.8, 0.4 + total_touches * 0.1),
                                start_index=start,
                                end_index=end - 1,
                                key_levels=[upper_line[-1], lower_line[-1]],
                                description=f"{channel_type.title()} channel: {total_touches} touches",
                                target=None,  # Depends on channel direction
                                stop_loss=None
                            ))
                
                except (np.linalg.LinAlgError, ValueError):
                    continue
        
        return patterns
    
    def _find_peaks(self, price_series: pd.Series, distance: int = 5) -> List[Dict]:
        """Find peaks in price series"""
        peaks = []
        
        try:
            peak_indices = argrelextrema(price_series.values, np.greater, order=distance)[0]
            
            for idx in peak_indices:
                if 0 <= idx < len(price_series):
                    peaks.append({
                        'index': idx,
                        'price': price_series.iloc[idx]
                    })
        except:
            pass
        
        return peaks
    
    def _find_valleys(self, price_series: pd.Series, distance: int = 5) -> List[Dict]:
        """Find valleys in price series"""
        valleys = []
        
        try:
            valley_indices = argrelextrema(price_series.values, np.less, order=distance)[0]
            
            for idx in valley_indices:
                if 0 <= idx < len(price_series):
                    valleys.append({
                        'index': idx,
                        'price': price_series.iloc[idx]
                    })
        except:
            pass
        
        return valleys
    
    def _find_neckline_low(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[float]:
        """Find neckline low between two points"""
        if start_idx >= end_idx or end_idx >= len(df):
            return None
        
        section = df.iloc[start_idx:end_idx + 1]
        return section['low'].min()
    
    def _find_neckline_high(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[float]:
        """Find neckline high between two points"""
        if start_idx >= end_idx or end_idx >= len(df):
            return None
        
        section = df.iloc[start_idx:end_idx + 1]
        return section['high'].max()
    
    def _calculate_hs_confidence(self, df: pd.DataFrame, left: Dict, head: Dict, right: Dict, neckline: float) -> float:
        """Calculate Head & Shoulders pattern confidence"""
        base_confidence = 0.6
        
        # Symmetry bonus
        shoulder_symmetry = 1 - abs(left['price'] - right['price']) / ((left['price'] + right['price']) / 2)
        symmetry_bonus = shoulder_symmetry * 0.2
        
        # Volume confirmation (if available)
        volume_bonus = 0
        if 'volume' in df.columns:
            head_volume = df.iloc[head['index']]['volume']
            avg_volume = df['volume'].rolling(20).mean().iloc[head['index']]
            if head_volume > avg_volume * 1.2:  # Higher volume at head
                volume_bonus = 0.1
        
        # Pattern completeness
        pattern_height = abs(head['price'] - neckline)
        avg_price = (left['price'] + head['price'] + right['price']) / 3
        height_ratio = pattern_height / avg_price
        
        if height_ratio > 0.03:  # At least 3% pattern height
            height_bonus = min(0.1, height_ratio * 2)
        else:
            height_bonus = 0
        
        return min(0.95, base_confidence + symmetry_bonus + volume_bonus + height_bonus)

class CandlestickPatternDetector:
    """Detects Japanese candlestick patterns"""
    
    def __init__(self):
        self.body_threshold = 0.1  # Minimum body size as % of range
        self.doji_threshold = 0.05  # Maximum body size for doji
        self.long_shadow_ratio = 2.0  # Shadow to body ratio for long shadows
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect all candlestick patterns"""
        if len(df) < 3:
            return []
        
        patterns = []
        
        try:
            # Single candle patterns
            patterns.extend(self._detect_doji(df))
            patterns.extend(self._detect_hammer(df))
            patterns.extend(self._detect_shooting_star(df))
            patterns.extend(self._detect_spinning_tops(df))
            
            # Two candle patterns
            patterns.extend(self._detect_engulfing(df))
            patterns.extend(self._detect_harami(df))
            patterns.extend(self._detect_piercing_line(df))
            patterns.extend(self._detect_dark_cloud_cover(df))
            
            # Three candle patterns
            patterns.extend(self._detect_morning_star(df))
            patterns.extend(self._detect_evening_star(df))
            patterns.extend(self._detect_three_white_soldiers(df))
            patterns.extend(self._detect_three_black_crows(df))
            
            logger.debug(f"🕯️ Detected {len(patterns)} candlestick patterns")
            
        except Exception as e:
            logger.warning(f"⚠️ Candlestick pattern detection error: {e}")
        
        return patterns
    
    def _get_candle_properties(self, row) -> Dict[str, float]:
        """Get candle properties"""
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        body_size = abs(c - o)
        range_size = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        
        return {
            'body_size': body_size,
            'range_size': range_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'body_ratio': body_size / range_size if range_size > 0 else 0,
            'is_bullish': c > o,
            'is_bearish': c < o
        }
    
    def _detect_doji(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Doji patterns"""
        patterns = []
        
        for i in range(len(df)):
            props = self._get_candle_properties(df.iloc[i])
            
            if props['body_ratio'] <= self.doji_threshold:  # Very small body
                # Classify doji type
                if props['upper_shadow'] > props['lower_shadow'] * 2:
                    doji_type = "Dragonfly Doji"
                    direction = "bullish"
                elif props['lower_shadow'] > props['upper_shadow'] * 2:
                    doji_type = "Gravestone Doji"
                    direction = "bearish"
                else:
                    doji_type = "Classic Doji"
                    direction = "neutral"
                
                # Context matters for doji significance
                strength = 0.5
                if i > 5:  # Check trend context
                    recent_trend = self._determine_trend_context(df, i, 5)
                    if recent_trend != 'sideways':
                        strength = 0.7  # Stronger in trending market
                
                patterns.append(CandlestickPattern(
                    name=doji_type,
                    type="reversal",
                    direction=direction,
                    strength=strength,
                    index=i,
                    description=f"{doji_type} at ${df.iloc[i]['close']:.2f}",
                    reliability="medium"
                ))
        
        return patterns
    
    def _detect_hammer(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Hammer and Hanging Man patterns"""
        patterns = []
        
        for i in range(len(df)):
            props = self._get_candle_properties(df.iloc[i])
            
            # Hammer/Hanging man criteria
            if (props['lower_shadow'] > props['body_size'] * 2 and  # Long lower shadow
                props['upper_shadow'] < props['body_size'] * 0.3 and  # Short upper shadow
                props['body_ratio'] > 0.1):  # Reasonable body size
                
                # Determine context
                if i > 3:
                    trend_context = self._determine_trend_context(df, i, 5)
                    
                    if trend_context == 'downtrend':
                        pattern_name = "Hammer"
                        direction = "bullish"
                        reliability = "high"
                        strength = 0.8
                    elif trend_context == 'uptrend':
                        pattern_name = "Hanging Man"
                        direction = "bearish"
                        reliability = "medium"
                        strength = 0.6
                    else:
                        pattern_name = "Hammer/Hanging Man"
                        direction = "neutral"
                        reliability = "low"
                        strength = 0.4
                    
                    patterns.append(CandlestickPattern(
                        name=pattern_name,
                        type="reversal",
                        direction=direction,
                        strength=strength,
                        index=i,
                        description=f"{pattern_name} after {trend_context}",
                        reliability=reliability
                    ))
        
        return patterns
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Shooting Star pattern"""
        patterns = []
        
        for i in range(len(df)):
            props = self._get_candle_properties(df.iloc[i])
            
            # Shooting star criteria
            if (props['upper_shadow'] > props['body_size'] * 2 and  # Long upper shadow
                props['lower_shadow'] < props['body_size'] * 0.3 and  # Short lower shadow
                props['body_ratio'] > 0.1):  # Reasonable body size
                
                if i > 3:
                    trend_context = self._determine_trend_context(df, i, 5)
                    
                    if trend_context == 'uptrend':
                        strength = 0.8
                        reliability = "high"
                    else:
                        strength = 0.5
                        reliability = "medium"
                    
                    patterns.append(CandlestickPattern(
                        name="Shooting Star",
                        type="reversal",
                        direction="bearish",
                        strength=strength,
                        index=i,
                        description=f"Shooting Star after {trend_context}",
                        reliability=reliability
                    ))
        
        return patterns
    
    def _detect_engulfing(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Bullish/Bearish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            prev_candle = self._get_candle_properties(df.iloc[i-1])
            curr_candle = self._get_candle_properties(df.iloc[i])
            
            prev_o, prev_c = df.iloc[i-1]['open'], df.iloc[i-1]['close']
            curr_o, curr_c = df.iloc[i]['open'], df.iloc[i]['close']
            
            # Bullish Engulfing
            if (prev_candle['is_bearish'] and curr_candle['is_bullish'] and
                curr_o < prev_c and curr_c > prev_o):  # Current engulfs previous
                
                strength = min(0.9, 0.6 + (curr_candle['body_size'] / prev_candle['body_size'] - 1) * 0.3)
                
                patterns.append(CandlestickPattern(
                    name="Bearish Engulfing",
                    type="reversal", 
                    direction="bearish",
                    strength=strength,
                    index=i,
                    description="Bearish Engulfing pattern",
                    reliability="high"
                ))
        
        return patterns
    
    def _detect_morning_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Morning Star pattern"""
        patterns = []
        
        for i in range(2, len(df)):
            candle1 = self._get_candle_properties(df.iloc[i-2])  # First candle
            candle2 = self._get_candle_properties(df.iloc[i-1])  # Star candle
            candle3 = self._get_candle_properties(df.iloc[i])    # Third candle
            
            c1_close, c2_close, c3_close = df.iloc[i-2]['close'], df.iloc[i-1]['close'], df.iloc[i]['close']
            
            # Morning Star criteria
            if (candle1['is_bearish'] and candle3['is_bullish'] and  # First bearish, third bullish
                candle2['body_ratio'] < 0.3 and  # Small star body
                c2_close < c1_close and c3_close > (c1_close + c2_close) / 2):  # Gap and recovery
                
                patterns.append(CandlestickPattern(
                    name="Morning Star",
                    type="reversal",
                    direction="bullish",
                    strength=0.8,
                    index=i,
                    description="Morning Star reversal pattern",
                    reliability="high"
                ))
        
        return patterns
    
    def _detect_evening_star(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Evening Star pattern"""
        patterns = []
        
        for i in range(2, len(df)):
            candle1 = self._get_candle_properties(df.iloc[i-2])
            candle2 = self._get_candle_properties(df.iloc[i-1])
            candle3 = self._get_candle_properties(df.iloc[i])
            
            c1_close, c2_close, c3_close = df.iloc[i-2]['close'], df.iloc[i-1]['close'], df.iloc[i]['close']
            
            # Evening Star criteria
            if (candle1['is_bullish'] and candle3['is_bearish'] and
                candle2['body_ratio'] < 0.3 and
                c2_close > c1_close and c3_close < (c1_close + c2_close) / 2):
                
                patterns.append(CandlestickPattern(
                    name="Evening Star",
                    type="reversal",
                    direction="bearish",
                    strength=0.8,
                    index=i,
                    description="Evening Star reversal pattern",
                    reliability="high"
                ))
        
        return patterns
    
    def _detect_three_white_soldiers(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Three White Soldiers pattern"""
        patterns = []
        
        for i in range(2, len(df)):
            candle1 = self._get_candle_properties(df.iloc[i-2])
            candle2 = self._get_candle_properties(df.iloc[i-1])
            candle3 = self._get_candle_properties(df.iloc[i])
            
            # All three candles must be bullish with decent bodies
            if (candle1['is_bullish'] and candle2['is_bullish'] and candle3['is_bullish'] and
                all(candle['body_ratio'] > 0.4 for candle in [candle1, candle2, candle3])):
                
                c1_close = df.iloc[i-2]['close']
                c2_close = df.iloc[i-1]['close'] 
                c3_close = df.iloc[i]['close']
                
                # Each close should be higher than previous
                if c2_close > c1_close and c3_close > c2_close:
                    patterns.append(CandlestickPattern(
                        name="Three White Soldiers",
                        type="continuation",
                        direction="bullish", 
                        strength=0.7,
                        index=i,
                        description="Three White Soldiers bullish continuation",
                        reliability="high"
                    ))
        
        return patterns
    
    def _detect_three_black_crows(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Three Black Crows pattern"""
        patterns = []
        
        for i in range(2, len(df)):
            candle1 = self._get_candle_properties(df.iloc[i-2])
            candle2 = self._get_candle_properties(df.iloc[i-1])
            candle3 = self._get_candle_properties(df.iloc[i])
            
            # All three candles must be bearish with decent bodies
            if (candle1['is_bearish'] and candle2['is_bearish'] and candle3['is_bearish'] and
                all(candle['body_ratio'] > 0.4 for candle in [candle1, candle2, candle3])):
                
                c1_close = df.iloc[i-2]['close']
                c2_close = df.iloc[i-1]['close']
                c3_close = df.iloc[i]['close']
                
                # Each close should be lower than previous
                if c2_close < c1_close and c3_close < c2_close:
                    patterns.append(CandlestickPattern(
                        name="Three Black Crows",
                        type="continuation",
                        direction="bearish",
                        strength=0.7,
                        index=i,
                        description="Three Black Crows bearish continuation",
                        reliability="high"
                    ))
        
        return patterns
    
    def _detect_harami(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Harami patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            prev_candle = self._get_candle_properties(df.iloc[i-1])
            curr_candle = self._get_candle_properties(df.iloc[i])
            
            prev_o, prev_c = df.iloc[i-1]['open'], df.iloc[i-1]['close']
            curr_o, curr_c = df.iloc[i]['open'], df.iloc[i]['close']
            
            # Current candle body inside previous candle body
            if (min(prev_o, prev_c) < min(curr_o, curr_c) and
                max(prev_o, prev_c) > max(curr_o, curr_c) and
                prev_candle['body_size'] > curr_candle['body_size'] * 2):  # Previous much larger
                
                # Bullish Harami
                if prev_candle['is_bearish'] and curr_candle['is_bullish']:
                    patterns.append(CandlestickPattern(
                        name="Bullish Harami",
                        type="reversal",
                        direction="bullish",
                        strength=0.6,
                        index=i,
                        description="Bullish Harami reversal pattern",
                        reliability="medium"
                    ))
                
                # Bearish Harami
                elif prev_candle['is_bullish'] and curr_candle['is_bearish']:
                    patterns.append(CandlestickPattern(
                        name="Bearish Harami",
                        type="reversal",
                        direction="bearish",
                        strength=0.6,
                        index=i,
                        description="Bearish Harami reversal pattern",
                        reliability="medium"
                    ))
        
        return patterns
    
    def _detect_piercing_line(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Piercing Line pattern"""
        patterns = []
        
        for i in range(1, len(df)):
            prev_candle = self._get_candle_properties(df.iloc[i-1])
            curr_candle = self._get_candle_properties(df.iloc[i])
            
            prev_o, prev_c = df.iloc[i-1]['open'], df.iloc[i-1]['close']
            curr_o, curr_c = df.iloc[i]['open'], df.iloc[i]['close']
            
            # Piercing Line criteria
            if (prev_candle['is_bearish'] and curr_candle['is_bullish'] and
                curr_o < prev_c and  # Gap down
                curr_c > (prev_o + prev_c) / 2 and  # Close above midpoint
                curr_c < prev_o):  # But below previous open
                
                penetration = (curr_c - prev_c) / (prev_o - prev_c)
                strength = 0.5 + min(0.3, penetration * 0.6)  # Deeper penetration = stronger
                
                patterns.append(CandlestickPattern(
                    name="Piercing Line",
                    type="reversal",
                    direction="bullish",
                    strength=strength,
                    index=i,
                    description=f"Piercing Line ({penetration:.0%} penetration)",
                    reliability="medium"
                ))
        
        return patterns
    
    def _detect_dark_cloud_cover(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Dark Cloud Cover pattern"""
        patterns = []
        
        for i in range(1, len(df)):
            prev_candle = self._get_candle_properties(df.iloc[i-1])
            curr_candle = self._get_candle_properties(df.iloc[i])
            
            prev_o, prev_c = df.iloc[i-1]['open'], df.iloc[i-1]['close']
            curr_o, curr_c = df.iloc[i]['open'], df.iloc[i]['close']
            
            # Dark Cloud Cover criteria
            if (prev_candle['is_bullish'] and curr_candle['is_bearish'] and
                curr_o > prev_c and  # Gap up
                curr_c < (prev_o + prev_c) / 2 and  # Close below midpoint
                curr_c > prev_o):  # But above previous open
                
                penetration = (prev_c - curr_c) / (prev_c - prev_o)
                strength = 0.5 + min(0.3, penetration * 0.6)
                
                patterns.append(CandlestickPattern(
                    name="Dark Cloud Cover",
                    type="reversal", 
                    direction="bearish",
                    strength=strength,
                    index=i,
                    description=f"Dark Cloud Cover ({penetration:.0%} penetration)",
                    reliability="medium"
                ))
        
        return patterns
    
    def _detect_spinning_tops(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """Detect Spinning Top patterns"""
        patterns = []
        
        for i in range(len(df)):
            props = self._get_candle_properties(df.iloc[i])
            
            # Spinning top: small body, long shadows on both sides
            if (0.05 < props['body_ratio'] < 0.3 and  # Small but not doji
                props['upper_shadow'] > props['body_size'] and
                props['lower_shadow'] > props['body_size']):
                
                patterns.append(CandlestickPattern(
                    name="Spinning Top",
                    type="reversal",
                    direction="neutral",
                    strength=0.4,
                    index=i,
                    description="Spinning Top - market indecision",
                    reliability="low"
                ))
        
        return patterns
    
    def _determine_trend_context(self, df: pd.DataFrame, index: int, lookback: int) -> str:
        """Determine trend context for pattern"""
        if index < lookback:
            return 'unknown'
        
        start_price = df.iloc[index - lookback]['close']
        end_price = df.iloc[index - 1]['close']  # Previous candle
        
        price_change = (end_price - start_price) / start_price
        
        if price_change > 0.02:  # 2% up
            return 'uptrend'
        elif price_change < -0.02:  # 2% down
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_trend_slope(self, price_series: pd.Series) -> float:
        """Calculate trend slope"""
        if len(price_series) < 2:
            return 0
        
        x = np.arange(len(price_series))
        try:
            slope = np.polyfit(x, price_series.values, 1)[0]
            return slope / price_series.mean()  # Normalize by price level
        except:
            return 0
    
    def _find_peaks_in_window(self, price_series: pd.Series) -> List[Dict]:
        """Find peaks within a window"""
        peaks = []
        for i in range(2, len(price_series) - 2):
            if (price_series.iloc[i] > price_series.iloc[i-1] and
                price_series.iloc[i] > price_series.iloc[i+1] and
                price_series.iloc[i] > price_series.iloc[i-2] and
                price_series.iloc[i] > price_series.iloc[i+2]):
                
                peaks.append({
                    'index': i,
                    'price': price_series.iloc[i]
                })
        
        return peaks
    
    def _find_valleys_in_window(self, price_series: pd.Series) -> List[Dict]:
        """Find valleys within a window"""
        valleys = []
        for i in range(2, len(price_series) - 2):
            if (price_series.iloc[i] < price_series.iloc[i-1] and
                price_series.iloc[i] < price_series.iloc[i+1] and
                price_series.iloc[i] < price_series.iloc[i-2] and
                price_series.iloc[i] < price_series.iloc[i+2]):
                
                valleys.append({
                    'index': i,
                    'price': price_series.iloc[i]
                })
        
        return valleys

class PatternSignalGenerator:
    """Generate trading signals from pattern analysis"""
    
    def __init__(self):
        self.chart_detector = ChartPatternDetector()
        self.candlestick_detector = CandlestickPatternDetector()
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete pattern analysis"""
        result = {
            'chart_patterns': [],
            'candlestick_patterns': [],
            'signals': [],
            'pattern_score': 0
        }
        
        try:
            # Detect chart patterns
            chart_patterns = self.chart_detector.detect_all_patterns(df)
            result['chart_patterns'] = chart_patterns
            
            # Detect candlestick patterns
            candlestick_patterns = self.candlestick_detector.detect_all_patterns(df)
            result['candlestick_patterns'] = candlestick_patterns
            
            # Generate combined signals
            result['signals'] = self._generate_pattern_signals(chart_patterns, candlestick_patterns, df)
            
            # Calculate overall pattern score
            result['pattern_score'] = self._calculate_pattern_score(chart_patterns, candlestick_patterns)
            
        except Exception as e:
            logger.error(f"❌ Pattern analysis failed: {e}")
        
        return result
    
    def _generate_pattern_signals(self, chart_patterns: List[ChartPattern], 
                                 candlestick_patterns: List[CandlestickPattern],
                                 df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals from patterns"""
        signals = []
        
        # Chart pattern signals
        for pattern in chart_patterns[-3:]:  # Recent patterns only
            if pattern.confidence > 0.6:
                signals.append({
                    'type': 'CHART_PATTERN',
                    'direction': pattern.direction.upper(),
                    'strength': pattern.confidence,
                    'description': pattern.description,
                    'pattern_name': pattern.name,
                    'target': pattern.target,
                    'stop_loss': pattern.stop_loss
                })
        
        # Candlestick pattern signals
        recent_candles = candlestick_patterns[-5:]  # Last 5 patterns
        for pattern in recent_candles:
            if pattern.strength > 0.6 and pattern.reliability in ['high', 'medium']:
                signals.append({
                    'type': 'CANDLESTICK_PATTERN',
                    'direction': pattern.direction.upper(),
                    'strength': pattern.strength,
                    'description': pattern.description,
                    'pattern_name': pattern.name,
                    'reliability': pattern.reliability
                })
        
        return signals
    
    def _calculate_pattern_score(self, chart_patterns: List[ChartPattern], 
                               candlestick_patterns: List[CandlestickPattern]) -> float:
        """Calculate overall pattern score"""
        score = 0
        
        # Chart patterns contribution
        if chart_patterns:
            chart_score = max(p.confidence for p in chart_patterns[-3:]) if chart_patterns[-3:] else 0
            score += chart_score * 0.6
        
        # Candlestick patterns contribution
        if candlestick_patterns:
            recent_candles = candlestick_patterns[-3:]
            if recent_candles:
                candle_score = max(p.strength for p in recent_candles)
                score += candle_score * 0.4
        
        return min(1.0, score)

# Global pattern analyzers
pattern_signal_generator = PatternSignalGenerator()

# Convenience functions
def analyze_chart_patterns(df: pd.DataFrame) -> List[ChartPattern]:
    """Analyze chart patterns"""
    return pattern_signal_generator.chart_detector.detect_all_patterns(df)

def analyze_candlestick_patterns(df: pd.DataFrame) -> List[CandlestickPattern]:
    """Analyze candlestick patterns"""
    return pattern_signal_generator.candlestick_detector.detect_all_patterns(df)

def get_pattern_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Get all pattern-based signals"""
    return pattern_signal_generator.analyze_patterns(df)body_size'] - 1) * 0.3)
                
                patterns.append(CandlestickPattern(
                    name="Bullish Engulfing",
                    type="reversal",
                    direction="bullish",
                    strength=strength,
                    index=i,
                    description="Bullish Engulfing pattern",
                    reliability="high"
                ))
            
            # Bearish Engulfing
            elif (prev_candle['is_bullish'] and curr_candle['is_bearish'] and
                  curr_o > prev_c and curr_c < prev_o):  # Current engulfs previous
                
                strength = min(0.9, 0.6 + (curr_candle['body_size'] / prev_candle['