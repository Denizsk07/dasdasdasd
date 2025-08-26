 """
XAUUSD Smart Money Concepts (SMC) Analysis
Advanced institutional trading concepts: BOS, CHOCH, Order Blocks, Liquidity, FVG
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('smc_analysis')

@dataclass
class SwingPoint:
    """Swing high or low point"""
    index: int
    price: float
    type: str  # 'high' or 'low'
    timestamp: Optional[datetime] = None
    strength: int = 0  # Number of bars on each side

@dataclass
class StructureBreak:
    """Market structure break"""
    break_index: int
    break_price: float
    break_type: str  # 'BOS' (Break of Structure) or 'CHOCH' (Change of Character)
    direction: str   # 'bullish' or 'bearish'
    previous_swing: SwingPoint
    strength: float  # 0-1 confidence
    description: str

@dataclass
class OrderBlock:
    """Institutional order block"""
    start_index: int
    end_index: int
    high: float
    low: float
    type: str        # 'bullish' or 'bearish'
    origin_type: str # 'bos', 'choch', 'liquidity_grab'
    strength: float  # 0-1 confidence
    tested: bool = False
    broken: bool = False

@dataclass
class LiquidityPool:
    """Liquidity concentration area"""
    price: float
    type: str       # 'equal_highs', 'equal_lows', 'triple_top', 'triple_bottom'
    strength: float # Amount of liquidity (0-1)
    swept: bool = False
    index: int = 0

@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance)"""
    start_index: int
    end_index: int
    top: float
    bottom: float
    type: str       # 'bullish' or 'bearish'
    size: float     # Gap size in price
    filled: bool = False
    partially_filled: float = 0.0  # Percentage filled

class SMCAnalyzer:
    """Smart Money Concepts Analyzer"""
    
    def __init__(self):
        self.swing_detection_period = 5  # Bars on each side for swing detection
        self.min_structure_break_pips = 2.0  # Minimum break size for XAUUSD
        self.order_block_lookback = 10  # Bars to look back for order blocks
        
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Complete SMC analysis of market structure"""
        if len(df) < 50:
            logger.warning("⚠️ Insufficient data for SMC analysis")
            return self._empty_smc_result()
        
        logger.debug(f"🧠 Running SMC analysis on {len(df)} bars")
        
        try:
            # 1. Identify swing points
            swing_points = self._find_swing_points(df)
            
            # 2. Detect structure breaks
            structure_breaks = self._detect_structure_breaks(df, swing_points)
            
            # 3. Find order blocks
            order_blocks = self._find_order_blocks(df, structure_breaks)
            
            # 4. Identify liquidity pools
            liquidity_pools = self._find_liquidity_pools(df, swing_points)
            
            # 5. Detect Fair Value Gaps
            fair_value_gaps = self._find_fair_value_gaps(df)
            
            # 6. Current market bias
            market_bias = self._determine_market_bias(df, structure_breaks, order_blocks)
            
            result = {
                'swing_points': swing_points,
                'structure_breaks': structure_breaks,
                'order_blocks': order_blocks,
                'liquidity_pools': liquidity_pools,
                'fair_value_gaps': fair_value_gaps,
                'market_bias': market_bias,
                'signals': self._generate_smc_signals(df, structure_breaks, order_blocks, liquidity_pools, fair_value_gaps)
            }
            
            logger.debug(f"✅ SMC analysis complete: {len(structure_breaks)} breaks, {len(order_blocks)} OBs, {len(liquidity_pools)} liquidity")
            return result
            
        except Exception as e:
            logger.error(f"❌ SMC analysis failed: {e}")
            return self._empty_smc_result()
    
    def _find_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing highs and lows"""
        swing_points = []
        period = self.swing_detection_period
        
        for i in range(period, len(df) - period):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for swing high
            left_highs = df.iloc[i-period:i]['high']
            right_highs = df.iloc[i+1:i+period+1]['high']
            
            if all(current_high >= h for h in left_highs) and all(current_high > h for h in right_highs):
                swing_points.append(SwingPoint(
                    index=i,
                    price=current_high,
                    type='high',
                    timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else None,
                    strength=period
                ))
            
            # Check for swing low
            left_lows = df.iloc[i-period:i]['low']
            right_lows = df.iloc[i+1:i+period+1]['low']
            
            if all(current_low <= l for l in left_lows) and all(current_low < l for l in right_lows):
                swing_points.append(SwingPoint(
                    index=i,
                    price=current_low,
                    type='low',
                    timestamp=df.index[i] if hasattr(df.index, 'to_pydatetime') else None,
                    strength=period
                ))
        
        return sorted(swing_points, key=lambda x: x.index)
    
    def _detect_structure_breaks(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[StructureBreak]:
        """Detect BOS (Break of Structure) and CHOCH (Change of Character)"""
        structure_breaks = []
        
        if len(swing_points) < 2:
            return structure_breaks
        
        # Group swing points by type
        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']
        
        # Check for breaks of previous swing highs (bullish BOS)
        for i, swing_high in enumerate(swing_highs[:-1]):
            next_swing_high = swing_highs[i + 1]
            
            # Look for price action after the swing high
            start_idx = swing_high.index
            end_idx = min(next_swing_high.index, len(df) - 1)
            
            # Check if price breaks above the swing high
            for j in range(start_idx + 1, end_idx):
                if df.iloc[j]['close'] > swing_high.price + self.min_structure_break_pips:
                    # Determine if it's BOS or CHOCH
                    break_type = 'BOS'  # Most common
                    
                    # CHOCH: if previous trend was bearish and now breaks bullish
                    if i > 0:
                        prev_swing = swing_highs[i - 1]
                        if prev_swing.price > swing_high.price:  # Previous downtrend
                            break_type = 'CHOCH'
                    
                    structure_breaks.append(StructureBreak(
                        break_index=j,
                        break_price=df.iloc[j]['close'],
                        break_type=break_type,
                        direction='bullish',
                        previous_swing=swing_high,
                        strength=self._calculate_break_strength(df, j, swing_high.price, 'bullish'),
                        description=f"Bullish {break_type}: Break above ${swing_high.price:.2f}"
                    ))
                    break
        
        # Check for breaks of previous swing lows (bearish BOS)
        for i, swing_low in enumerate(swing_lows[:-1]):
            next_swing_low = swing_lows[i + 1]
            
            start_idx = swing_low.index
            end_idx = min(next_swing_low.index, len(df) - 1)
            
            # Check if price breaks below the swing low
            for j in range(start_idx + 1, end_idx):
                if df.iloc[j]['close'] < swing_low.price - self.min_structure_break_pips:
                    # Determine if it's BOS or CHOCH
                    break_type = 'BOS'
                    
                    if i > 0:
                        prev_swing = swing_lows[i - 1]
                        if prev_swing.price < swing_low.price:  # Previous uptrend
                            break_type = 'CHOCH'
                    
                    structure_breaks.append(StructureBreak(
                        break_index=j,
                        break_price=df.iloc[j]['close'],
                        break_type=break_type,
                        direction='bearish',
                        previous_swing=swing_low,
                        strength=self._calculate_break_strength(df, j, swing_low.price, 'bearish'),
                        description=f"Bearish {break_type}: Break below ${swing_low.price:.2f}"
                    ))
                    break
        
        return sorted(structure_breaks, key=lambda x: x.break_index)
    
    def _find_order_blocks(self, df: pd.DataFrame, structure_breaks: List[StructureBreak]) -> List[OrderBlock]:
        """Find institutional order blocks"""
        order_blocks = []
        
        for structure_break in structure_breaks:
            # Look for the last opposing candle before the break
            break_idx = structure_break.break_index
            direction = structure_break.direction
            
            # Search backwards from the break point
            search_start = max(0, break_idx - self.order_block_lookback)
            
            for i in range(break_idx - 1, search_start - 1, -1):
                candle = df.iloc[i]
                
                if direction == 'bullish':
                    # Look for last bearish candle before bullish break
                    if candle['close'] < candle['open']:  # Bearish candle
                        order_blocks.append(OrderBlock(
                            start_index=i,
                            end_index=i,
                            high=candle['high'],
                            low=candle['low'],
                            type='bullish',
                            origin_type=structure_break.break_type.lower(),
                            strength=structure_break.strength * 0.8,
                            tested=False,
                            broken=False
                        ))
                        break
                
                elif direction == 'bearish':
                    # Look for last bullish candle before bearish break
                    if candle['close'] > candle['open']:  # Bullish candle
                        order_blocks.append(OrderBlock(
                            start_index=i,
                            end_index=i,
                            high=candle['high'],
                            low=candle['low'],
                            type='bearish',
                            origin_type=structure_break.break_type.lower(),
                            strength=structure_break.strength * 0.8,
                            tested=False,
                            broken=False
                        ))
                        break
        
        # Remove duplicate/overlapping order blocks
        order_blocks = self._filter_order_blocks(order_blocks)
        
        return order_blocks
    
    def _find_liquidity_pools(self, df: pd.DataFrame, swing_points: List[SwingPoint]) -> List[LiquidityPool]:
        """Find areas of liquidity concentration"""
        liquidity_pools = []
        tolerance = 2.0  # Price tolerance for equal levels in XAUUSD
        
        # Group swing points by type
        swing_highs = [sp for sp in swing_points if sp.type == 'high']
        swing_lows = [sp for sp in swing_points if sp.type == 'low']
        
        # Find equal highs
        for i, swing1 in enumerate(swing_highs):
            equal_highs = [swing1]
            
            for j, swing2 in enumerate(swing_highs[i+1:], i+1):
                if abs(swing1.price - swing2.price) <= tolerance:
                    equal_highs.append(swing2)
            
            if len(equal_highs) >= 2:  # At least 2 equal highs
                avg_price = sum(s.price for s in equal_highs) / len(equal_highs)
                liquidity_type = 'equal_highs'
                
                if len(equal_highs) >= 3:
                    liquidity_type = 'triple_top'
                
                liquidity_pools.append(LiquidityPool(
                    price=avg_price,
                    type=liquidity_type,
                    strength=min(1.0, len(equal_highs) / 4.0),  # Max strength at 4 touches
                    swept=False,
                    index=equal_highs[0].index
                ))
        
        # Find equal lows
        for i, swing1 in enumerate(swing_lows):
            equal_lows = [swing1]
            
            for j, swing2 in enumerate(swing_lows[i+1:], i+1):
                if abs(swing1.price - swing2.price) <= tolerance:
                    equal_lows.append(swing2)
            
            if len(equal_lows) >= 2:
                avg_price = sum(s.price for s in equal_lows) / len(equal_lows)
                liquidity_type = 'equal_lows'
                
                if len(equal_lows) >= 3:
                    liquidity_type = 'triple_bottom'
                
                liquidity_pools.append(LiquidityPool(
                    price=avg_price,
                    type=liquidity_type,
                    strength=min(1.0, len(equal_lows) / 4.0),
                    swept=False,
                    index=equal_lows[0].index
                ))
        
        # Remove duplicates and sort by strength
        liquidity_pools = list({lp.price: lp for lp in liquidity_pools}.values())
        liquidity_pools.sort(key=lambda x: x.strength, reverse=True)
        
        return liquidity_pools[:10]  # Top 10 liquidity pools
    
    def _find_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Find Fair Value Gaps (FVG) - Imbalances"""
        fair_value_gaps = []
        
        if len(df) < 3:
            return fair_value_gaps
        
        for i in range(1, len(df) - 1):
            candle1 = df.iloc[i - 1]  # Previous candle
            candle2 = df.iloc[i]      # Current candle  
            candle3 = df.iloc[i + 1]  # Next candle
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if (candle2['close'] > candle2['open'] and  # Current candle bullish
                candle3['low'] > candle1['high']):      # Gap exists
                
                gap_size = candle3['low'] - candle1['high']
                if gap_size > 1.0:  # Minimum gap size for XAUUSD
                    fair_value_gaps.append(FairValueGap(
                        start_index=i - 1,
                        end_index=i + 1,
                        top=candle3['low'],
                        bottom=candle1['high'],
                        type='bullish',
                        size=gap_size,
                        filled=False
                    ))
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            elif (candle2['close'] < candle2['open'] and  # Current candle bearish
                  candle3['high'] < candle1['low']):      # Gap exists
                
                gap_size = candle1['low'] - candle3['high']
                if gap_size > 1.0:
                    fair_value_gaps.append(FairValueGap(
                        start_index=i - 1,
                        end_index=i + 1,
                        top=candle1['low'],
                        bottom=candle3['high'],
                        type='bearish',
                        size=gap_size,
                        filled=False
                    ))
        
        return fair_value_gaps
    
    def _determine_market_bias(self, df: pd.DataFrame, structure_breaks: List[StructureBreak], 
                              order_blocks: List[OrderBlock]) -> Dict[str, Any]:
        """Determine overall market bias"""
        bias = {
            'direction': 'NEUTRAL',
            'strength': 0.5,
            'reasoning': [],
            'last_break': None,
            'active_obs': 0
        }
        
        if not structure_breaks:
            return bias
        
        # Analyze recent structure breaks (last 3)
        recent_breaks = structure_breaks[-3:] if len(structure_breaks) >= 3 else structure_breaks
        
        bullish_breaks = sum(1 for b in recent_breaks if b.direction == 'bullish')
        bearish_breaks = sum(1 for b in recent_breaks if b.direction == 'bearish')
        
        # Last break has more weight
        last_break = structure_breaks[-1]
        bias['last_break'] = last_break
        
        if last_break.direction == 'bullish':
            bias['direction'] = 'BULLISH'
            bias['strength'] = 0.6 + (last_break.strength * 0.3)
            bias['reasoning'].append(f"Last break: Bullish {last_break.break_type}")
        else:
            bias['direction'] = 'BEARISH'
            bias['strength'] = 0.6 + (last_break.strength * 0.3)
            bias['reasoning'].append(f"Last break: Bearish {last_break.break_type}")
        
        # Factor in recent break pattern
        if bullish_breaks > bearish_breaks:
            if bias['direction'] == 'BULLISH':
                bias['strength'] = min(0.9, bias['strength'] + 0.1)
            bias['reasoning'].append(f"Recent pattern: {bullish_breaks}B/{bearish_breaks}B")
        elif bearish_breaks > bullish_breaks:
            if bias['direction'] == 'BEARISH':
                bias['strength'] = min(0.9, bias['strength'] + 0.1)
            bias['reasoning'].append(f"Recent pattern: {bullish_breaks}B/{bearish_breaks}B")
        
        # Count active order blocks
        current_price = df.iloc[-1]['close']
        active_obs = 0
        
        for ob in order_blocks:
            if not ob.broken and ob.low <= current_price <= ob.high:
                active_obs += 1
        
        bias['active_obs'] = active_obs
        if active_obs > 0:
            bias['reasoning'].append(f"{active_obs} active order blocks")
        
        return bias
    
    def _generate_smc_signals(self, df: pd.DataFrame, structure_breaks: List[StructureBreak],
                             order_blocks: List[OrderBlock], liquidity_pools: List[LiquidityPool],
                             fair_value_gaps: List[FairValueGap]) -> List[Dict[str, Any]]:
        """Generate trading signals based on SMC analysis"""
        signals = []
        current_price = df.iloc[-1]['close']
        
        # Signal 1: Order Block Retest
        for ob in order_blocks[-5:]:  # Check last 5 order blocks
            if not ob.broken:
                ob_mid = (ob.high + ob.low) / 2
                price_in_ob = ob.low <= current_price <= ob.high
                
                if price_in_ob:
                    direction = 'BUY' if ob.type == 'bullish' else 'SELL'
                    signals.append({
                        'type': 'ORDER_BLOCK_RETEST',
                        'direction': direction,
                        'strength': ob.strength,
                        'price': current_price,
                        'description': f'{ob.type.title()} order block retest at ${ob_mid:.2f}',
                        'entry_zone': {'high': ob.high, 'low': ob.low}
                    })
        
        # Signal 2: Liquidity Grab
        for lp in liquidity_pools:
            if not lp.swept:
                distance = abs(current_price - lp.price)
                if distance <= 5.0:  # Within 5 pips of liquidity
                    # Expect price to grab liquidity then reverse
                    direction = 'SELL' if lp.type in ['equal_highs', 'triple_top'] else 'BUY'
                    signals.append({
                        'type': 'LIQUIDITY_GRAB',
                        'direction': direction,
                        'strength': lp.strength,
                        'price': lp.price,
                        'description': f'Potential {lp.type} liquidity grab at ${lp.price:.2f}',
                        'target': lp.price
                    })
        
        # Signal 3: Fair Value Gap Fill
        for fvg in fair_value_gaps[-3:]:  # Last 3 FVGs
            if not fvg.filled:
                fvg_mid = (fvg.top + fvg.bottom) / 2
                
                if fvg.type == 'bullish' and fvg.bottom <= current_price <= fvg.top:
                    signals.append({
                        'type': 'FVG_FILL',
                        'direction': 'BUY',
                        'strength': 0.6,
                        'price': current_price,
                        'description': f'Bullish FVG fill opportunity at ${fvg_mid:.2f}',
                        'zone': {'high': fvg.top, 'low': fvg.bottom}
                    })
                elif fvg.type == 'bearish' and fvg.bottom <= current_price <= fvg.top:
                    signals.append({
                        'type': 'FVG_FILL', 
                        'direction': 'SELL',
                        'strength': 0.6,
                        'price': current_price,
                        'description': f'Bearish FVG fill opportunity at ${fvg_mid:.2f}',
                        'zone': {'high': fvg.top, 'low': fvg.bottom}
                    })
        
        # Signal 4: Recent Structure Break Continuation
        if structure_breaks:
            last_break = structure_breaks[-1]
            bars_since_break = len(df) - 1 - last_break.break_index
            
            if bars_since_break <= 10 and last_break.strength > 0.7:  # Recent strong break
                signals.append({
                    'type': 'STRUCTURE_CONTINUATION',
                    'direction': 'BUY' if last_break.direction == 'bullish' else 'SELL',
                    'strength': last_break.strength,
                    'price': current_price,
                    'description': f'{last_break.direction.title()} {last_break.break_type} continuation',
                    'break_level': last_break.break_price
                })
        
        return signals
    
    def _calculate_break_strength(self, df: pd.DataFrame, break_index: int, break_level: float, direction: str) -> float:
        """Calculate the strength of a structure break"""
        if break_index >= len(df):
            return 0.5
        
        break_candle = df.iloc[break_index]
        
        # Factors that increase break strength:
        strength = 0.5  # Base strength
        
        # 1. Size of the breaking candle
        candle_size = abs(break_candle['close'] - break_candle['open'])
        avg_candle_size = df['close'].rolling(20).apply(lambda x: abs(x.diff()).mean()).iloc[break_index]
        
        if avg_candle_size > 0:
            size_factor = min(0.2, candle_size / avg_candle_size * 0.1)
            strength += size_factor
        
        # 2. Volume (if available)
        if 'volume' in df.columns and df['volume'].iloc[break_index] > 0:
            avg_volume = df['volume'].rolling(20).mean().iloc[break_index]
            if avg_volume > 0 and df['volume'].iloc[break_index] > avg_volume * 1.5:
                strength += 0.15  # High volume break
        
        # 3. Clean break (no immediate pullback)
        if break_index < len(df) - 3:
            next_3_candles = df.iloc[break_index+1:break_index+4]
            if direction == 'bullish':
                clean_break = all(candle['low'] > break_level for _, candle in next_3_candles.iterrows())
            else:
                clean_break = all(candle['high'] < break_level for _, candle in next_3_candles.iterrows())
            
            if clean_break:
                strength += 0.15
        
        return min(1.0, strength)
    
    def _filter_order_blocks(self, order_blocks: List[OrderBlock]) -> List[OrderBlock]:
        """Filter out overlapping or weak order blocks"""
        if not order_blocks:
            return []
        
        # Sort by strength
        order_blocks.sort(key=lambda x: x.strength, reverse=True)
        
        filtered = []
        for ob in order_blocks:
            # Check if it overlaps significantly with existing blocks
            overlaps = False
            for existing in filtered:
                overlap_high = min(ob.high, existing.high)
                overlap_low = max(ob.low, existing.low)
                
                if overlap_high > overlap_low:  # There is overlap
                    overlap_size = overlap_high - overlap_low
                    ob_size = ob.high - ob.low
                    
                    if overlap_size > ob_size * 0.5:  # More than 50% overlap
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(ob)
        
        return filtered[:8]  # Keep top 8 order blocks
    
    def _empty_smc_result(self) -> Dict[str, Any]:
        """Return empty SMC result structure"""
        return {
            'swing_points': [],
            'structure_breaks': [],
            'order_blocks': [],
            'liquidity_pools': [],
            'fair_value_gaps': [],
            'market_bias': {'direction': 'NEUTRAL', 'strength': 0.5, 'reasoning': []},
            'signals': []
        }

class SMCSignalGenerator:
    """Generate trading signals from SMC analysis"""
    
    def __init__(self):
        self.analyzer = SMCAnalyzer()
    
    def get_smc_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get SMC-based trading signals"""
        smc_analysis = self.analyzer.analyze_market_structure(df)
        return smc_analysis['signals']
    
    def get_market_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get current market bias from SMC perspective"""
        smc_analysis = self.analyzer.analyze_market_structure(df)
        return smc_analysis['market_bias']
    
    def get_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Get current order blocks"""
        smc_analysis = self.analyzer.analyze_market_structure(df)
        return smc_analysis['order_blocks']

# Global SMC analyzer
smc_analyzer = SMCAnalyzer()
smc_signal_generator = SMCSignalGenerator()

# Convenience functions
def analyze_smc(df: pd.DataFrame) -> Dict[str, Any]:
    """Run complete SMC analysis"""
    return smc_analyzer.analyze_market_structure(df)

def get_smc_bias(df: pd.DataFrame) -> str:
    """Get SMC market bias"""
    bias = smc_signal_generator.get_market_bias(df)
    return bias['direction']

def get_active_order_blocks(df: pd.DataFrame) -> List[OrderBlock]:
    """Get currently active order blocks"""
    return smc_signal_generator.get_order_blocks(df)