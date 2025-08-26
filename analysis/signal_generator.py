 """
XAUUSD Signal Generator - Multi-Strategy Signal Generation Engine
Combines all analysis methods to generate high-quality trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from config.settings import settings
from utils.logger import get_module_logger
from data import get_historical_data, get_current_price, is_market_open
from data.price_validator import validate_data_quality
from analysis.technical_indicators import add_all_indicators, get_current_signals
from analysis.smc_analysis import analyze_smc
from analysis.pattern_detection import get_pattern_signals

logger = get_module_logger('signal_generator')

@dataclass
class TradingSignal:
    """Complete trading signal with all details"""
    direction: str          # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    confidence: float       # 0-100
    timeframe: str
    timestamp: datetime
    
    # Analysis details
    triggered_strategies: List[str]
    strategy_scores: Dict[str, float]
    reasoning: List[str]
    
    # Risk management
    risk_reward_ratios: List[float]
    position_size: float
    max_risk_usd: float
    
    # Market context
    market_session: str
    trend_context: str
    volatility_level: str
    
    # Additional metadata
    data_quality: float
    signal_id: str

class MultiStrategyAnalyzer:
    """Combines all analysis strategies into unified scoring system"""
    
    def __init__(self):
        self.strategy_weights = settings.strategy_weights.to_dict()
        self.min_signal_score = settings.trading.min_signal_score
        self.signal_counter = 0
        
    def analyze_all_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all analysis strategies and combine results"""
        if df.empty or len(df) < 50:
            logger.warning("⚠️ Insufficient data for multi-strategy analysis")
            return self._empty_analysis_result()
        
        logger.debug(f"🔍 Running multi-strategy analysis on {len(df)} bars")
        
        try:
            # 1. Technical Indicators Analysis
            df_with_indicators = add_all_indicators(df)
            technical_signals = get_current_signals(df_with_indicators)
            technical_score = self._score_technical_signals(technical_signals, df_with_indicators)
            
            # 2. Smart Money Concepts Analysis  
            smc_analysis = analyze_smc(df)
            smc_score = self._score_smc_analysis(smc_analysis, df)
            
            # 3. Pattern Analysis
            pattern_analysis = get_pattern_signals(df)
            pattern_score = self._score_pattern_analysis(pattern_analysis, df)
            
            # 4. Volume Analysis
            volume_score = self._score_volume_analysis(df_with_indicators)
            
            # 5. Price Action Analysis
            price_action_score = self._score_price_action(df)
            
            # 6. Support/Resistance Analysis
            sr_score = self._score_support_resistance(df_with_indicators)
            
            # 7. Fair Value Gap Analysis
            fvg_score = self._score_fair_value_gaps(smc_analysis.get('fair_value_gaps', []), df)
            
            # 8. Candlestick Analysis
            candlestick_score = self._score_candlestick_patterns(pattern_analysis.get('candlestick_patterns', []), df)
            
            # Combine all strategy scores
            strategy_scores = {
                'technical_indicators': technical_score,
                'smc': smc_score,
                'patterns': pattern_score,
                'volume': volume_score,
                'price_action': price_action_score,
                'support_resistance': sr_score,
                'fvg': fvg_score,
                'candlesticks': candlestick_score
            }
            
            # Calculate weighted composite score
            composite_score = self._calculate_composite_score(strategy_scores)
            
            # Determine signal direction
            signal_direction = self._determine_signal_direction(strategy_scores, df)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(strategy_scores, signal_direction)
            
            result = {
                'strategy_scores': strategy_scores,
                'composite_score': composite_score,
                'signal_direction': signal_direction,
                'reasoning': reasoning,
                'triggered_strategies': [k for k, v in strategy_scores.items() if abs(v['score']) > 10],
                'market_context': self._get_market_context(df_with_indicators),
                'data_quality': validate_data_quality(df, df.index[-1].strftime('%M') if hasattr(df.index[-1], 'strftime') else '15')
            }
            
            logger.debug(f"✅ Multi-strategy analysis complete: Score {composite_score:.1f}, Direction {signal_direction}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Multi-strategy analysis failed: {e}")
            return self._empty_analysis_result()
    
    def _score_technical_signals(self, signals: List, df: pd.DataFrame) -> Dict[str, Any]:
        """Score technical indicator signals"""
        if not signals:
            return {'direction': 'NEUTRAL', 'score': 0, 'details': []}
        
        buy_score = 0
        sell_score = 0
        details = []
        
        for signal in signals:
            if signal.direction == 'BUY':
                buy_score += signal.strength * 25  # Max 25 points per signal
                details.append(f"Bullish {signal.name}: {signal.description}")
            elif signal.direction == 'SELL':
                sell_score += signal.strength * 25
                details.append(f"Bearish {signal.name}: {signal.description}")
        
        # Add trend alignment bonus
        if len(df) > 50:
            trend_bonus = self._calculate_trend_bonus(df)
            if buy_score > sell_score:
                buy_score *= (1 + trend_bonus)
            else:
                sell_score *= (1 + trend_bonus)
        
        net_score = buy_score - sell_score
        direction = 'BUY' if net_score > 5 else ('SELL' if net_score < -5 else 'NEUTRAL')
        
        return {
            'direction': direction,
            'score': abs(net_score),
            'details': details[:3],  # Top 3 details
            'buy_score': buy_score,
            'sell_score': sell_score
        }
    
    def _score_smc_analysis(self, smc_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Score Smart Money Concepts analysis"""
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        # Market bias from SMC
        market_bias = smc_analysis.get('market_bias', {})
        if market_bias.get('direction') != 'NEUTRAL':
            bias_score = market_bias.get('strength', 0.5) * 30  # Max 30 points
            score += bias_score
            direction = market_bias['direction']
            details.append(f"SMC Bias: {direction} ({market_bias['strength']:.1%})")
        
        # Structure breaks
        structure_breaks = smc_analysis.get('structure_breaks', [])
        if structure_breaks:
            latest_break = structure_breaks[-1]
            break_score = latest_break.strength * 25  # Max 25 points
            score += break_score
            break_direction = 'BUY' if latest_break.direction == 'bullish' else 'SELL'
            details.append(f"Recent {latest_break.break_type}: {break_direction}")
            
            # Override direction if structure break is stronger
            if break_score > 20:
                direction = break_direction
        
        # Order blocks
        order_blocks = smc_analysis.get('order_blocks', [])
        active_obs = [ob for ob in order_blocks if not ob.broken]
        if active_obs:
            current_price = df.iloc[-1]['close']
            for ob in active_obs[-2:]:  # Last 2 order blocks
                if ob.low <= current_price <= ob.high:
                    ob_score = ob.strength * 20  # Max 20 points
                    score += ob_score
                    ob_direction = 'BUY' if ob.type == 'bullish' else 'SELL'
                    details.append(f"Order Block: {ob.type.title()}")
        
        # Liquidity pools
        liquidity_pools = smc_analysis.get('liquidity_pools', [])
        if liquidity_pools:
            current_price = df.iloc[-1]['close']
            for lp in liquidity_pools[:2]:  # Top 2 liquidity pools
                distance = abs(current_price - lp.price) / current_price
                if distance < 0.002:  # Within 0.2%
                    liq_score = lp.strength * 15
                    score += liq_score
                    liq_direction = 'SELL' if lp.type in ['equal_highs', 'triple_top'] else 'BUY'
                    details.append(f"Liquidity: {lp.type}")
        
        return {
            'direction': direction,
            'score': min(score, 100),  # Cap at 100
            'details': details[:3]
        }
    
    def _score_pattern_analysis(self, pattern_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Score chart and candlestick patterns"""
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        # Chart patterns
        chart_patterns = pattern_analysis.get('chart_patterns', [])
        for pattern in chart_patterns[-2:]:  # Latest 2 patterns
            if pattern.confidence > 0.6:
                pattern_score = pattern.confidence * 30  # Max 30 points
                score += pattern_score
                
                if pattern.direction == 'bullish':
                    direction = 'BUY' if direction == 'NEUTRAL' else direction
                elif pattern.direction == 'bearish':
                    direction = 'SELL' if direction == 'NEUTRAL' else direction
                
                details.append(f"Chart: {pattern.name} ({pattern.confidence:.0%})")
        
        # Candlestick patterns
        candlestick_patterns = pattern_analysis.get('candlestick_patterns', [])
        recent_candles = [p for p in candlestick_patterns if p.index >= len(df) - 5]  # Last 5 bars
        
        for pattern in recent_candles:
            if pattern.strength > 0.6 and pattern.reliability in ['high', 'medium']:
                candle_score = pattern.strength * 20  # Max 20 points
                score += candle_score
                
                if pattern.direction == 'bullish':
                    direction = 'BUY' if direction == 'NEUTRAL' else direction
                elif pattern.direction == 'bearish':
                    direction = 'SELL' if direction == 'NEUTRAL' else direction
                
                details.append(f"Candle: {pattern.name} ({pattern.strength:.0%})")
        
        return {
            'direction': direction,
            'score': min(score, 80),  # Cap at 80
            'details': details[:2]
        }
    
    def _score_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Score volume-based signals"""
        if 'volume' not in df.columns:
            return {'direction': 'NEUTRAL', 'score': 0, 'details': ['No volume data']}
        
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        try:
            latest = df.iloc[-1]
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # High volume confirmation
            if recent_volume > avg_volume * 1.5:
                price_change = (latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
                
                if price_change > 0.001:  # 0.1% up with high volume
                    score += 25
                    direction = 'BUY'
                    details.append(f"High volume bullish ({recent_volume/avg_volume:.1f}x avg)")
                elif price_change < -0.001:  # 0.1% down with high volume
                    score += 25
                    direction = 'SELL'
                    details.append(f"High volume bearish ({recent_volume/avg_volume:.1f}x avg)")
            
            # Volume trend analysis
            if len(df) > 10:
                volume_trend = np.polyfit(range(10), df['volume'].tail(10), 1)[0]
                price_trend = np.polyfit(range(10), df['close'].tail(10), 1)[0]
                
                # Volume-Price divergence
                if volume_trend > 0 and price_trend < 0:
                    score += 15
                    direction = 'BUY' if direction == 'NEUTRAL' else direction
                    details.append("Volume-Price bullish divergence")
                elif volume_trend < 0 and price_trend > 0:
                    score += 15  
                    direction = 'SELL' if direction == 'NEUTRAL' else direction
                    details.append("Volume-Price bearish divergence")
        
        except Exception as e:
            logger.warning(f"⚠️ Volume analysis error: {e}")
        
        return {
            'direction': direction,
            'score': min(score, 50),  # Cap at 50
            'details': details
        }
    
    def _score_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Score pure price action signals"""
        if len(df) < 20:
            return {'direction': 'NEUTRAL', 'score': 0, 'details': []}
        
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        try:
            # Higher highs, higher lows (bullish)
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)
            
            hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
            hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] > recent_lows.iloc[i-1])
            
            if hh_count >= 6 and hl_count >= 6:  # Strong uptrend
                score += 30
                direction = 'BUY'
                details.append("Higher Highs & Higher Lows pattern")
            
            # Lower highs, lower lows (bearish)
            lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] < recent_highs.iloc[i-1])
            ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])
            
            if lh_count >= 6 and ll_count >= 6:  # Strong downtrend
                score += 30
                direction = 'SELL'
                details.append("Lower Highs & Lower Lows pattern")
            
            # Breakout analysis
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            current_price = df.iloc[-1]['close']
            
            # Breakout above recent high
            if current_price > recent_high * 1.001:  # 0.1% above
                score += 20
                direction = 'BUY' if direction == 'NEUTRAL' else direction
                details.append(f"Breakout above ${recent_high:.2f}")
            
            # Breakdown below recent low
            elif current_price < recent_low * 0.999:  # 0.1% below
                score += 20
                direction = 'SELL' if direction == 'NEUTRAL' else direction
                details.append(f"Breakdown below ${recent_low:.2f}")
            
            # Momentum analysis
            if len(df) >= 5:
                momentum = (df.iloc[-1]['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']
                
                if momentum > 0.005:  # 0.5% momentum
                    score += 15
                    details.append(f"Bullish momentum ({momentum:.1%})")
                elif momentum < -0.005:
                    score += 15
                    details.append(f"Bearish momentum ({abs(momentum):.1%})")
        
        except Exception as e:
            logger.warning(f"⚠️ Price action analysis error: {e}")
        
        return {
            'direction': direction,
            'score': min(score, 70),
            'details': details[:2]
        }
    
    def _score_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Score support/resistance levels"""
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        try:
            current_price = df.iloc[-1]['close']
            
            # Check pivot points if available
            if 'pivot' in df.columns:
                pivot = df['pivot'].iloc[-1]
                r1 = df['r1'].iloc[-1] if 'r1' in df.columns else None
                s1 = df['s1'].iloc[-1] if 's1' in df.columns else None
                
                # Price near support (bullish)
                if s1 and abs(current_price - s1) / current_price < 0.002:  # Within 0.2%
                    score += 25
                    direction = 'BUY'
                    details.append(f"Near S1 support ${s1:.2f}")
                
                # Price near resistance (bearish)
                elif r1 and abs(current_price - r1) / current_price < 0.002:
                    score += 25
                    direction = 'SELL'
                    details.append(f"Near R1 resistance ${r1:.2f}")
                
                # Above/below pivot
                elif current_price > pivot * 1.002:
                    score += 10
                    direction = 'BUY' if direction == 'NEUTRAL' else direction
                    details.append("Above pivot point")
                elif current_price < pivot * 0.998:
                    score += 10
                    direction = 'SELL' if direction == 'NEUTRAL' else direction
                    details.append("Below pivot point")
            
            # Dynamic S/R levels
            if 'dynamic_support' in df.columns and 'dynamic_resistance' in df.columns:
                support = df['dynamic_support'].iloc[-1]
                resistance = df['dynamic_resistance'].iloc[-1]
                
                if abs(current_price - support) / current_price < 0.003:
                    score += 20
                    direction = 'BUY' if direction == 'NEUTRAL' else direction
                    details.append(f"At dynamic support ${support:.2f}")
                
                elif abs(current_price - resistance) / current_price < 0.003:
                    score += 20
                    direction = 'SELL' if direction == 'NEUTRAL' else direction
                    details.append(f"At dynamic resistance ${resistance:.2f}")
            
            # Round number levels (psychological)
            round_numbers = [2000, 2050, 2100, 2150, 2200, 2250, 2300]
            nearest_round = min(round_numbers, key=lambda x: abs(x - current_price))
            distance_to_round = abs(current_price - nearest_round)
            
            if distance_to_round < 3:  # Within $3 of round number
                score += 15
                direction = 'SELL' if current_price > nearest_round else 'BUY'
                details.append(f"Near ${nearest_round} psychological level")
        
        except Exception as e:
            logger.warning(f"⚠️ Support/Resistance analysis error: {e}")
        
        return {
            'direction': direction,
            'score': min(score, 60),
            'details': details[:2]
        }
    
    def _score_fair_value_gaps(self, fvgs: List, df: pd.DataFrame) -> Dict[str, Any]:
        """Score Fair Value Gap signals"""
        if not fvgs:
            return {'direction': 'NEUTRAL', 'score': 0, 'details': []}
        
        score = 0
        direction = 'NEUTRAL'
        details = []
        current_price = df.iloc[-1]['close']
        
        # Check recent unfilled FVGs
        for fvg in fvgs[-3:]:  # Last 3 FVGs
            if not fvg.filled:
                fvg_mid = (fvg.top + fvg.bottom) / 2
                
                # Price in FVG zone
                if fvg.bottom <= current_price <= fvg.top:
                    fvg_score = min(20, fvg.size * 5)  # Max 20 points
                    score += fvg_score
                    
                    direction = 'BUY' if fvg.type == 'bullish' else 'SELL'
                    details.append(f"{fvg.type.title()} FVG fill at ${fvg_mid:.2f}")
                
                # Price approaching FVG
                elif abs(current_price - fvg_mid) / current_price < 0.003:  # Within 0.3%
                    score += 10
                    direction = 'BUY' if fvg.type == 'bullish' else 'SELL'
                    details.append(f"Approaching {fvg.type} FVG")
        
        return {
            'direction': direction,
            'score': min(score, 40),
            'details': details[:2]
        }
    
    def _score_candlestick_patterns(self, patterns: List, df: pd.DataFrame) -> Dict[str, Any]:
        """Score recent candlestick patterns"""
        if not patterns:
            return {'direction': 'NEUTRAL', 'score': 0, 'details': []}
        
        score = 0
        direction = 'NEUTRAL'
        details = []
        
        # Focus on very recent patterns (last 3 bars)
        recent_patterns = [p for p in patterns if p.index >= len(df) - 3]
        
        for pattern in recent_patterns:
            if pattern.reliability == 'high' and pattern.strength > 0.7:
                pattern_score = pattern.strength * 25  # Max 25 points
                score += pattern_score
                
                if pattern.direction == 'bullish':
                    direction = 'BUY'
                elif pattern.direction == 'bearish':
                    direction = 'SELL'
                
                details.append(f"{pattern.name} ({pattern.reliability})")
        
        return {
            'direction': direction,
            'score': min(score, 35),
            'details': details[:2]
        }
    
    def _calculate_composite_score(self, strategy_scores: Dict[str, Any]) -> float:
        """Calculate weighted composite score"""
        total_score = 0
        
        for strategy_name, result in strategy_scores.items():
            weight = self.strategy_weights.get(strategy_name, 0)
            strategy_score = result.get('score', 0)
            
            # Apply weight to strategy score
            weighted_score = strategy_score * weight
            total_score += weighted_score
        
        return min(100, total_score)
    
    def _determine_signal_direction(self, strategy_scores: Dict[str, Any], df: pd.DataFrame) -> str:
        """Determine overall signal direction"""
        buy_weight = 0
        sell_weight = 0
        
        for strategy_name, result in strategy_scores.items():
            weight = self.strategy_weights.get(strategy_name, 0)
            direction = result.get('direction', 'NEUTRAL')
            score = result.get('score', 0)
            
            weighted_score = score * weight
            
            if direction == 'BUY':
                buy_weight += weighted_score
            elif direction == 'SELL':
                sell_weight += weighted_score
        
        # Determine direction with minimum threshold
        net_score = buy_weight - sell_weight
        
        if net_score > 5:  # Minimum 5 point difference
            return 'BUY'
        elif net_score < -5:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _generate_reasoning(self, strategy_scores: Dict[str, Any], direction: str) -> List[str]:
        """Generate human-readable reasoning"""
        reasoning = []
        
        # Sort strategies by score
        sorted_strategies = sorted(
            [(k, v) for k, v in strategy_scores.items() if v['direction'] == direction and v['score'] > 10],
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        for strategy_name, result in sorted_strategies[:4]:  # Top 4 contributing strategies
            score = result['score']
            details = result.get('details', [])
            
            if details:
                reasoning.append(f"{strategy_name.replace('_', ' ').title()}: {details[0]}")
            else:
                reasoning.append(f"{strategy_name.replace('_', ' ').title()}: {score:.0f} points")
        
        return reasoning
    
    def _get_market_context(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get current market context"""
        context = {
            'session': 'unknown',
            'trend': 'neutral',
            'volatility': 'normal'
        }
        
        try:
            # Market session
            if hasattr(df.index[-1], 'hour'):
                hour = df.index[-1].hour
                if 22 <= hour or hour < 8:
                    context['session'] = 'asian'
                elif 8 <= hour < 16:
                    context['session'] = 'london'
                else:
                    context['session'] = 'newyork'
            
            # Trend context
            if len(df) >= 20:
                recent_closes = df['close'].tail(20)
                trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
                
                if trend_slope > 1:
                    context['trend'] = 'uptrend'
                elif trend_slope < -1:
                    context['trend'] = 'downtrend'
            
            # Volatility
            if 'atr' in df.columns:
                current_atr = df['atr'].iloc[-1]
                avg_atr = df['atr'].rolling(20).mean().iloc[-1]
                
                if current_atr > avg_atr * 1.3:
                    context['volatility'] = 'high'
                elif current_atr < avg_atr * 0.7:
                    context['volatility'] = 'low'
        
        except Exception as e:
            logger.warning(f"⚠️ Market context error: {e}")
        
        return context
    
    def _calculate_trend_bonus(self, df: pd.DataFrame) -> float:
        """Calculate trend alignment bonus"""
        try:
            if 'ema_8' in df.columns and 'ema_21' in df.columns and 'ema_50' in df.columns:
                latest = df.iloc[-1]
                
                # Perfect bullish alignment
                if latest['ema_8'] > latest['ema_21'] > latest['ema_50']:
                    return 0.2  # 20% bonus
                
                # Perfect bearish alignment
                elif latest['ema_8'] < latest['ema_21'] < latest['ema_50']:
                    return 0.2  # 20% bonus
        except:
            pass
        
        return 0
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            'strategy_scores': {},
            'composite_score': 0,
            'signal_direction': 'NEUTRAL',
            'reasoning': [],
            'triggered_strategies': [],
            'market_context': {'session': 'unknown', 'trend': 'neutral', 'volatility': 'normal'},
            'data_quality': 0
        }

class SignalGenerator:
    """Main signal generator class"""
    
    def __init__(self):
        self.analyzer = MultiStrategyAnalyzer()
        self.last_signal_time = None
        self.min_signal_interval = settings.trading.min_hours_between_signals * 3600  # Convert to seconds
        
    async def generate_signal(self, timeframe: str = None) -> Optional[TradingSignal]:
        """Generate trading signal for specified timeframe"""
        
        # Check if market is open
        if not is_market_open():
            logger.info("📴 Market closed - no signal generation")
            return None
        
        # Check signal interval
        if self._should_skip_due_to_interval():
            return None
        
        # Default to first configured timeframe
        if not timeframe:
            timeframe = settings.trading.timeframes[0]
        
        logger.info(f"🎯 Generating signal for {timeframe}min timeframe")
        
        try:
            # Get historical data
            df = await get_historical_data(timeframe, 500)
            if df is None or len(df) < 100:
                logger.warning("⚠️ Insufficient historical data for signal generation")
                return None
            
            # Validate data quality
            data_quality = validate_data_quality(df, timeframe)
            if data_quality < 0.6:
                logger.warning(f"⚠️ Poor data quality ({data_quality:.1%}) - skipping signal")
                return None
            
            # Run multi-strategy analysis
            analysis_result = self.analyzer.analyze_all_strategies(df)
            
            # Check if signal meets minimum criteria
            if (analysis_result['composite_score'] < self.analyzer.min_signal_score or
                analysis_result['signal_direction'] == 'NEUTRAL'):
                logger.debug(f"📊 Signal below threshold: {analysis_result['composite_score']:.1f}/{self.analyzer.min_signal_score}")
                return None
            
            # Get current price
            current_price = await get_current_price()
            if not current_price:
                logger.error("❌ Cannot get current price for signal")
                return None
            
            # Generate complete trading signal
            signal = await self._create_trading_signal(
                analysis_result, current_price, timeframe, data_quality
            )
            
            if signal:
                self.last_signal_time = datetime.now()
                logger.info(f"✅ {signal.direction} signal generated: Score {signal.confidence:.1f}")
                return signal
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
        
        return None
    
    async def _create_trading_signal(self, analysis: Dict[str, Any], current_price: float, 
                                   timeframe: str, data_quality: float) -> Optional[TradingSignal]:
        """Create complete trading signal from analysis"""
        
        try:
            direction = analysis['signal_direction']
            confidence = analysis['composite_score']
            
            # Calculate stop loss and take profits
            sl, tps = self._calculate_sl_tp(current_price, direction, analysis)
            
            # Calculate position sizing
            position_size = self._calculate_position_size(current_price, sl)
            
            # Calculate risk metrics
            risk_amount = abs(current_price - sl) * position_size
            rr_ratios = [(abs(tp - current_price) / abs(current_price - sl)) for tp in tps]
            
            # Generate signal ID
            self.analyzer.signal_counter += 1
            signal_id = f"XAUUSD_{direction}_{timeframe}M_{self.analyzer.signal_counter:04d}"
            
            # Create signal object
            signal = TradingSignal(
                direction=direction,
                entry_price=current_price,
                stop_loss=sl,
                take_profits=tps,
                confidence=confidence,
                timeframe=f"M{timeframe}",
                timestamp=datetime.now(),
                
                # Analysis details
                triggered_strategies=analysis['triggered_strategies'],
                strategy_scores=analysis['strategy_scores'],
                reasoning=analysis['reasoning'],
                
                # Risk management
                risk_reward_ratios=rr_ratios,
                position_size=position_size,
                max_risk_usd=risk_amount,
                
                # Market context
                market_session=analysis['market_context']['session'],
                trend_context=analysis['market_context']['trend'],
                volatility_level=analysis['market_context']['volatility'],
                
                # Metadata
                data_quality=data_quality,
                signal_id=signal_id
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Signal creation failed: {e}")
            return None
    
    def _calculate_sl_tp(self, entry_price: float, direction: str, analysis: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate stop loss and take profit levels"""
        
        # Base stop loss from config
        base_sl_pips = settings.trading.stop_loss_pips
        
        # Adjust SL based on volatility
        volatility_level = analysis['market_context']['volatility']
        if volatility_level == 'high':
            sl_multiplier = 1.3
        elif volatility_level == 'low':
            sl_multiplier = 0.8
        else:
            sl_multiplier = 1.0
        
        adjusted_sl_pips = base_sl_pips * sl_multiplier
        
        # Calculate actual levels
        if direction == 'BUY':
            stop_loss = entry_price - adjusted_sl_pips
            take_profits = [entry_price + tp for tp in settings.trading.tp_levels]
        else:  # SELL
            stop_loss = entry_price + adjusted_sl_pips
            take_profits = [entry_price - tp for tp in settings.trading.tp_levels]
        
        return stop_loss, take_profits
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        
        # Risk per trade (2% default)
        risk_percentage = settings.trading.risk_percentage / 100
        account_balance = 10000  # Default account size (configurable)
        
        # Risk amount in USD
        risk_amount_usd = account_balance * risk_percentage
        
        # Stop loss distance
        sl_distance = abs(entry_price - stop_loss)
        
        # Position size in lots (100oz per lot for XAUUSD)
        position_size_oz = risk_amount_usd / sl_distance
        position_size_lots = position_size_oz / 100
        
        # Limit position size
        max_position_lots = 1.0  # Maximum 1 lot
        min_position_lots = 0.01  # Minimum 0.01 lots
        
        return max(min_position_lots, min(max_position_lots, position_size_lots))
    
    def _should_skip_due_to_interval(self) -> bool:
        """Check if we should skip signal generation due to minimum interval"""
        if not self.last_signal_time:
            return False
        
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds()
        if time_since_last < self.min_signal_interval:
            minutes_to_wait = (self.min_signal_interval - time_since_last) / 60
            logger.debug(f"⏰ Minimum interval not met, wait {minutes_to_wait:.0f} more minutes")
            return True
        
        return False
    
    async def analyze_multiple_timeframes(self) -> Dict[str, Optional[TradingSignal]]:
        """Analyze all configured timeframes"""
        results = {}
        
        for timeframe in settings.trading.timeframes:
            logger.debug(f"📊 Analyzing {timeframe}min timeframe")
            signal = await self.generate_signal(timeframe)
            results[f"M{timeframe}"] = signal
        
        return results
    
    def get_signal_summary(self, signal: TradingSignal) -> str:
        """Get human-readable signal summary"""
        if not signal:
            return "No signal"
        
        rr_text = "/".join([f"1:{rr:.1f}" for rr in signal.risk_reward_ratios])
        
        summary = f"""
🎯 {signal.direction} Signal - {signal.confidence:.1f}% Confidence
💰 Entry: ${signal.entry_price:.2f}
🛑 Stop Loss: ${signal.stop_loss:.2f}
🎯 Take Profits: {' | '.join([f'${tp:.2f}' for tp in signal.take_profits])}
📊 Risk:Reward: {rr_text}
💎 Position: {signal.position_size:.3f} lots
⚡ Timeframe: {signal.timeframe}
🧠 Strategies: {len(signal.triggered_strategies)} triggered
"""
        return summary.strip()

class SignalValidator:
    """Validates signals before sending"""
    
    def __init__(self):
        self.max_daily_signals = settings.trading.max_signals_per_day
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
    
    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Validate signal meets all criteria"""
        
        # Reset daily counter if new day
        self._reset_daily_counter_if_needed()
        
        # Check daily limit
        if self.daily_signal_count >= self.max_daily_signals:
            return False, f"Daily signal limit reached ({self.max_daily_signals})"
        
        # Validate price levels
        if not self._validate_price_levels(signal):
            return False, "Invalid stop loss or take profit levels"
        
        # Validate risk amount
        if signal.max_risk_usd > 500:  # Max $500 risk per trade
            return False, f"Risk too high: ${signal.max_risk_usd:.2f}"
        
        # Validate confidence
        if signal.confidence < settings.trading.min_signal_score:
            return False, f"Confidence too low: {signal.confidence:.1f}%"
        
        # Check minimum risk:reward
        if signal.risk_reward_ratios[0] < 1.0:  # TP1 should be at least 1:1
            return False, f"Poor risk:reward ratio: {signal.risk_reward_ratios[0]:.1f}"
        
        return True, "Signal validated successfully"
    
    def record_signal_sent(self):
        """Record that a signal was sent"""
        self._reset_daily_counter_if_needed()
        self.daily_signal_count += 1
    
    def _validate_price_levels(self, signal: TradingSignal) -> bool:
        """Validate stop loss and take profit levels make sense"""
        entry = signal.entry_price
        sl = signal.stop_loss
        
        # Check stop loss is on correct side
        if signal.direction == 'BUY' and sl >= entry:
            return False
        if signal.direction == 'SELL' and sl <= entry:
            return False
        
        # Check take profits are on correct side
        for tp in signal.take_profits:
            if signal.direction == 'BUY' and tp <= entry:
                return False
            if signal.direction == 'SELL' and tp >= entry:
                return False
        
        # Check take profits are in ascending order (for BUY) or descending (for SELL)
        if signal.direction == 'BUY':
            if not all(signal.take_profits[i] <= signal.take_profits[i+1] for i in range(len(signal.take_profits)-1)):
                return False
        else:
            if not all(signal.take_profits[i] >= signal.take_profits[i+1] for i in range(len(signal.take_profits)-1)):
                return False
        
        return True
    
    def _reset_daily_counter_if_needed(self):
        """Reset daily counter if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_signal_count = 0
            self.last_reset_date = current_date

# Global instances
signal_generator = SignalGenerator()
signal_validator = SignalValidator()

# Convenience functions
async def generate_trading_signal(timeframe: str = None) -> Optional[TradingSignal]:
    """Generate a trading signal"""
    return await signal_generator.generate_signal(timeframe)

async def analyze_all_timeframes() -> Dict[str, Optional[TradingSignal]]:
    """Analyze all configured timeframes"""
    return await signal_generator.analyze_multiple_timeframes()

def validate_trading_signal(signal: TradingSignal) -> Tuple[bool, str]:
    """Validate a trading signal"""
    return signal_validator.validate_signal(signal)

def get_signal_stats() -> Dict[str, Any]:
    """Get signal generation statistics"""
    return {
        'daily_signals_sent': signal_validator.daily_signal_count,
        'daily_limit': signal_validator.max_daily_signals,
        'last_signal_time': signal_generator.last_signal_time,
        'min_interval_hours': settings.trading.min_hours_between_signals,
        'min_signal_score': settings.trading.min_signal_score
    }