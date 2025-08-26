"""
XAUUSD Strategy Optimizer - Machine Learning for Trading Strategy Optimization
Continuously learns from trade results and optimizes strategy weights and parameters
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

from config.settings import settings, StrategyWeights
from utils.logger import get_module_logger
from trading.performance_tracker import performance_tracker, PerformanceMetrics

logger = get_module_logger('strategy_optimizer')

@dataclass
class OptimizationResult:
    """Result of strategy optimization"""
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    improvement_score: float
    confidence: float
    trades_analyzed: int
    optimization_date: datetime
    reasoning: List[str]

@dataclass
class LearningInsight:
    """Learning insights from performance analysis"""
    insight_type: str  # 'strategy', 'timing', 'risk', 'market'
    description: str
    impact_score: float  # 0-1
    recommended_action: str
    supporting_data: Dict[str, Any]

class StrategyOptimizer:
    """Self-learning strategy optimization system"""
    
    def __init__(self):
        self.weights_file = settings.storage.weights_file
        self.optimization_history = []
        self.min_trades_for_optimization = 20
        self.learning_rate = 0.1  # How fast to adapt weights
        self.confidence_threshold = 0.6  # Minimum confidence for weight changes
        
        # Performance tracking
        self.performance_memory = []
        self.last_optimization_date = None
        
        # Learning parameters
        self.strategy_effectiveness_window = 50  # Last N trades for effectiveness
        self.market_condition_memory = 30  # Days to remember market conditions
        
        # Load optimization history
        self._load_optimization_history()
        
        logger.info("🧠 Strategy Optimizer initialized")
    
    def optimize_strategy_weights(self, force: bool = False) -> Optional[OptimizationResult]:
        """Optimize strategy weights based on recent performance"""
        try:
            # Get recent performance data
            learning_data = performance_tracker.get_learning_feedback(days=30)
            
            if not learning_data or not learning_data.get('strategy_feedback'):
                logger.info("📊 No learning data available for optimization")
                return None
            
            strategy_feedback = learning_data['strategy_feedback']
            
            # Check if we have enough data
            total_trades = sum(data['trades'] for data in strategy_feedback.values())
            if total_trades < self.min_trades_for_optimization and not force:
                logger.info(f"📊 Need {self.min_trades_for_optimization} trades for optimization, have {total_trades}")
                return None
            
            # Get current weights
            current_weights = settings.strategy_weights.to_dict()
            
            # Calculate new weights based on performance
            new_weights = self._calculate_optimal_weights(strategy_feedback, current_weights)
            
            # Validate and apply constraints
            new_weights = self._apply_weight_constraints(new_weights)
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(
                strategy_feedback, current_weights, new_weights
            )
            
            # Generate reasoning
            reasoning = self._generate_optimization_reasoning(
                strategy_feedback, current_weights, new_weights
            )
            
            # Calculate confidence in optimization
            confidence = self._calculate_optimization_confidence(strategy_feedback, total_trades)
            
            # Create optimization result
            result = OptimizationResult(
                old_weights=current_weights.copy(),
                new_weights=new_weights.copy(),
                improvement_score=improvement_score,
                confidence=confidence,
                trades_analyzed=total_trades,
                optimization_date=datetime.now(),
                reasoning=reasoning
            )
            
            # Apply weights if confidence is high enough
            if confidence >= self.confidence_threshold or force:
                self._update_strategy_weights(new_weights)
                self.optimization_history.append(result)
                self._save_optimization_history()
                
                logger.info(f"🧠 Strategy weights optimized: {improvement_score:+.1%} improvement, {confidence:.0%} confidence")
                for strategy, (old, new) in zip(current_weights.keys(), 
                                              zip(current_weights.values(), new_weights.values())):
                    if abs(new - old) > 0.01:  # Only log significant changes
                        logger.info(f"   📈 {strategy}: {old:.3f} → {new:.3f} ({((new-old)/old*100):+.1f}%)")
            else:
                logger.info(f"🧠 Optimization skipped: confidence {confidence:.0%} < {self.confidence_threshold:.0%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Strategy optimization failed: {e}")
            return None
    
    def learn_from_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Learn from recent performance and suggest improvements"""
        try:
            learning_data = performance_tracker.get_learning_feedback(days=days)
            
            if not learning_data:
                return {'error': 'No learning data available'}
            
            # Analyze strategy effectiveness
            strategy_analysis = self._analyze_strategy_effectiveness(learning_data)
            
            # Analyze parameter optimization opportunities
            parameter_analysis = self._analyze_parameter_optimization(learning_data)
            
            # Detect market regime changes
            market_regime_analysis = self._analyze_market_regime_changes(learning_data)
            
            # Generate learning insights
            insights = self._generate_learning_insights(strategy_analysis, parameter_analysis, market_regime_analysis)
            
            # Generate recommendations
            recommendations = self._generate_learning_recommendations(insights)
            
            return {
                'analysis_period_days': days,
                'strategy_analysis': strategy_analysis,
                'parameter_analysis': parameter_analysis,
                'market_regime_analysis': market_regime_analysis,
                'learning_insights': insights,
                'recommendations': recommendations,
                'next_optimization_date': self._calculate_next_optimization_date(),
                'learning_score': self._calculate_learning_score(insights)
            }
            
        except Exception as e:
            logger.error(f"❌ Learning analysis failed: {e}")
            return {'error': str(e)}
    
    def backtest_weight_changes(self, proposed_weights: Dict[str, float], 
                               days_back: int = 30) -> Dict[str, Any]:
        """Simulate how proposed weight changes would have performed historically"""
        try:
            # Get historical performance data
            performance_data = performance_tracker.get_performance_metrics()
            
            if not performance_data.strategy_performances:
                return {'error': 'No historical strategy performance data'}
            
            # Simulate performance with old vs new weights
            old_weights = settings.strategy_weights.to_dict()
            
            # Calculate weighted performance scores
            old_weighted_score = self._calculate_weighted_performance_score(
                performance_data.strategy_performances, old_weights
            )
            new_weighted_score = self._calculate_weighted_performance_score(
                performance_data.strategy_performances, proposed_weights
            )
            
            improvement = (new_weighted_score - old_weighted_score) / old_weighted_score if old_weighted_score > 0 else 0
            
            # Calculate confidence based on data quality and consistency
            backtest_confidence = self._calculate_backtest_confidence(performance_data)
            
            # Simulate trade outcomes with new weights
            simulated_results = self._simulate_trade_outcomes(proposed_weights, days_back)
            
            return {
                'old_weighted_score': old_weighted_score,
                'new_weighted_score': new_weighted_score,
                'projected_improvement': improvement,
                'confidence_level': backtest_confidence,
                'simulated_results': simulated_results,
                'recommended_apply': improvement > 0.05 and backtest_confidence > 0.7,
                'risk_assessment': self._assess_weight_change_risk(old_weights, proposed_weights)
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            return {'error': str(e)}
    
    def get_optimization_history(self, limit: int = 10) -> List[OptimizationResult]:
        """Get recent optimization history"""
        return sorted(self.optimization_history, key=lambda x: x.optimization_date, reverse=True)[:limit]
    
    def analyze_strategy_evolution(self) -> Dict[str, Any]:
        """Analyze how strategies have evolved over time"""
        try:
            if not self.optimization_history:
                return {'error': 'No optimization history available'}
            
            # Track weight evolution
            weight_evolution = {}
            for opt in self.optimization_history:
                date = opt.optimization_date.strftime('%Y-%m-%d')
                weight_evolution[date] = opt.new_weights
            
            # Calculate strategy momentum (which strategies are gaining/losing weight)
            strategy_momentum = self._calculate_strategy_momentum()
            
            # Identify winning and losing strategies
            strategy_rankings = self._rank_strategies_by_evolution()
            
            # Detect learning patterns
            learning_patterns = self._detect_learning_patterns()
            
            return {
                'weight_evolution': weight_evolution,
                'strategy_momentum': strategy_momentum,
                'strategy_rankings': strategy_rankings,
                'learning_patterns': learning_patterns,
                'optimization_frequency': len(self.optimization_history),
                'avg_improvement': np.mean([opt.improvement_score for opt in self.optimization_history]) if self.optimization_history else 0,
                'learning_velocity': self._calculate_learning_velocity()
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy evolution analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_optimal_weights(self, strategy_feedback: Dict[str, Any], 
                                 current_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate optimal strategy weights based on performance feedback"""
        
        new_weights = current_weights.copy()
        
        for strategy, feedback in strategy_feedback.items():
            if strategy not in new_weights:
                continue
            
            current_weight = new_weights[strategy]
            
            # Performance metrics
            win_rate = feedback['win_rate']
            avg_pips = feedback['avg_pips']
            trades_count = feedback['trades']
            
            # Normalize avg_pips to 0-1 scale
            normalized_pips = max(0, min(1, (avg_pips + 50) / 100))  # -50 to +50 pips range
            
            # Combined performance score
            performance_score = (win_rate * 0.6) + (normalized_pips * 0.4)
            
            # Confidence factor based on sample size
            confidence_factor = min(1.0, trades_count / 15)  # Full confidence at 15+ trades
            
            # Learning rate adjustment based on recent performance trend
            trend_factor = self._calculate_strategy_trend(strategy, feedback)
            adjusted_learning_rate = self.learning_rate * trend_factor
            
            # Weight adjustment calculation
            target_performance = 0.6  # Target 60% performance score
            performance_deviation = performance_score - target_performance
            
            # Apply non-linear scaling for extreme performers
            if performance_score > 0.8:  # Exceptional performance
                adjustment = performance_deviation * adjusted_learning_rate * 1.5
            elif performance_score < 0.3:  # Poor performance
                adjustment = performance_deviation * adjusted_learning_rate * 2.0
            else:  # Normal performance
                adjustment = performance_deviation * adjusted_learning_rate
            
            # Apply confidence weighting
            weighted_adjustment = adjustment * confidence_factor
            
            # Calculate new weight
            new_weight = current_weight + weighted_adjustment
            
            # Apply bounds (5% min, 40% max)
            new_weights[strategy] = max(0.05, min(0.4, new_weight))
        
        return new_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply constraints and normalize weights"""
        
        # Apply individual constraints
        constrained_weights = {}
        for strategy, weight in weights.items():
            constrained_weights[strategy] = max(0.05, min(0.4, weight))
        
        # Normalize to sum to 1.0
        total_weight = 0
        
        for strategy, weight in weights.items():
            if strategy in strategy_performances:
                perf = strategy_performances[strategy]
                # Performance score combining win rate and profitability
                score = (perf.win_rate * 0.7) + (max(0, min(1, (perf.avg_pips_per_trade + 25) / 50)) * 0.3)
                total_score += weight * score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_backtest_confidence(self, performance_data: PerformanceMetrics) -> float:
        """Calculate confidence level for backtest results"""
        
        # Base confidence on number of trades
        trade_confidence = min(1.0, performance_data.total_trades / 100)
        
        # Factor in time span of data
        if performance_data.monthly_pnl:
            months_of_data = len(performance_data.monthly_pnl)
            time_confidence = min(1.0, months_of_data / 6)
        else:
            time_confidence = 0.5
        
        # Factor in strategy diversity
        strategy_count = len(performance_data.strategy_performances)
        diversity_confidence = min(1.0, strategy_count / 5)
        
        return (trade_confidence * 0.5) + (time_confidence * 0.3) + (diversity_confidence * 0.2)
    
    def _simulate_trade_outcomes(self, proposed_weights: Dict[str, float], days_back: int) -> Dict[str, Any]:
        """Simulate trade outcomes with proposed weights"""
        
        # Simplified simulation
        current_weights = settings.strategy_weights.to_dict()
        
        # Calculate expected improvement based on weight changes
        expected_changes = {}
        for strategy, new_weight in proposed_weights.items():
            old_weight = current_weights.get(strategy, 0)
            weight_change = new_weight - old_weight
            expected_changes[strategy] = weight_change
        
        # Simulate outcomes
        total_expected_improvement = sum(abs(change) for change in expected_changes.values()) * 0.1
        
        return {
            'expected_win_rate_change': total_expected_improvement * 0.5,
            'expected_pip_improvement': total_expected_improvement * 10,
            'confidence_interval': [total_expected_improvement * 0.8, total_expected_improvement * 1.2]
        }
    
    def _assess_weight_change_risk(self, old_weights: Dict[str, float], 
                                 new_weights: Dict[str, float]) -> Dict[str, Any]:
        """Assess risk of proposed weight changes"""
        
        total_change = sum(abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in old_weights.keys())
        
        risk_level = "LOW"
        if total_change > 0.3:
            risk_level = "HIGH"
        elif total_change > 0.15:
            risk_level = "MEDIUM"
        
        return {
            'total_weight_change': total_change,
            'risk_level': risk_level,
            'max_single_change': max(abs(new_weights.get(s, 0) - old_weights.get(s, 0)) for s in old_weights.keys()),
            'recommendations': self._get_risk_mitigation_recommendations(risk_level)
        }
    
    def _calculate_strategy_momentum(self) -> Dict[str, float]:
        """Calculate momentum for each strategy"""
        
        if len(self.optimization_history) < 2:
            return {}
        
        # Get recent weight changes
        recent_opts = self.optimization_history[-3:]  # Last 3 optimizations
        strategy_momentum = {}
        
        for strategy in settings.strategy_weights.to_dict().keys():
            weight_changes = []
            for i in range(1, len(recent_opts)):
                old_weight = recent_opts[i-1].new_weights.get(strategy, 0)
                new_weight = recent_opts[i].new_weights.get(strategy, 0)
                weight_changes.append(new_weight - old_weight)
            
            # Calculate momentum as average change
            momentum = np.mean(weight_changes) if weight_changes else 0
            strategy_momentum[strategy] = momentum
        
        return strategy_momentum
    
    def _rank_strategies_by_evolution(self) -> Dict[str, Any]:
        """Rank strategies by their evolution over time"""
        
        if not self.optimization_history:
            return {}
        
        strategy_evolution = {}
        for strategy in settings.strategy_weights.to_dict().keys():
            initial_weight = self.optimization_history[0].old_weights.get(strategy, 0) if self.optimization_history else 0
            current_weight = settings.strategy_weights.to_dict().get(strategy, 0)
            
            evolution_score = (current_weight - initial_weight) / initial_weight if initial_weight > 0 else 0
            strategy_evolution[strategy] = {
                'initial_weight': initial_weight,
                'current_weight': current_weight,
                'evolution_score': evolution_score,
                'trend': 'gaining' if evolution_score > 0.1 else ('losing' if evolution_score < -0.1 else 'stable')
            }
        
        # Sort by evolution score
        sorted_strategies = sorted(strategy_evolution.items(), key=lambda x: x[1]['evolution_score'], reverse=True)
        
        return {
            'winners': [s for s, data in sorted_strategies if data['evolution_score'] > 0.1],
            'losers': [s for s, data in sorted_strategies if data['evolution_score'] < -0.1],
            'stable': [s for s, data in sorted_strategies if abs(data['evolution_score']) <= 0.1],
            'evolution_details': dict(sorted_strategies)
        }
    
    def _detect_learning_patterns(self) -> List[str]:
        """Detect patterns in learning behavior"""
        
        patterns = []
        
        if len(self.optimization_history) >= 3:
            # Check for improvement trend
            recent_improvements = [opt.improvement_score for opt in self.optimization_history[-3:]]
            if all(imp > 0 for imp in recent_improvements):
                patterns.append("Consistent improvement trend detected")
            elif all(imp < 0 for imp in recent_improvements):
                patterns.append("Declining optimization effectiveness")
            
            # Check for confidence trend
            recent_confidences = [opt.confidence for opt in self.optimization_history[-3:]]
            if all(conf > 0.8 for conf in recent_confidences):
                patterns.append("High confidence learning phase")
            elif all(conf < 0.6 for conf in recent_confidences):
                patterns.append("Low confidence period - may need more data")
        
        if len(self.optimization_history) >= 5:
            # Check for cyclical patterns
            improvements = [opt.improvement_score for opt in self.optimization_history[-5:]]
            if len(set(np.sign(improvements))) > 1:  # Mixed positive/negative
                patterns.append("Cyclical optimization pattern detected")
        
        return patterns
    
    def _calculate_learning_velocity(self) -> float:
        """Calculate how fast the system is learning"""
        
        if len(self.optimization_history) < 2:
            return 0.0
        
        # Calculate time between optimizations
        time_diffs = []
        for i in range(1, len(self.optimization_history)):
            prev_opt = self.optimization_history[i-1]
            curr_opt = self.optimization_history[i]
            time_diff = (curr_opt.optimization_date - prev_opt.optimization_date).days
            time_diffs.append(time_diff)
        
        avg_time_between = np.mean(time_diffs)
        
        # Calculate average improvement per optimization
        avg_improvement = np.mean([opt.improvement_score for opt in self.optimization_history])
        
        # Learning velocity = improvement per day
        learning_velocity = avg_improvement / avg_time_between if avg_time_between > 0 else 0
        
        return max(0, learning_velocity)
    
    def _calculate_learning_score(self, insights: List[LearningInsight]) -> float:
        """Calculate overall learning effectiveness score"""
        
        if not insights:
            return 0.5  # Neutral score
        
        # Weight insights by impact score
        weighted_score = sum(insight.impact_score for insight in insights) / len(insights)
        
        # Adjust based on insight diversity
        insight_types = set(insight.insight_type for insight in insights)
        diversity_bonus = len(insight_types) / 4.0  # Max 4 types
        
        # Adjust based on actionability
        actionable_insights = sum(1 for insight in insights if insight.recommended_action)
        actionability_score = actionable_insights / len(insights)
        
        final_score = (weighted_score * 0.6) + (diversity_bonus * 0.2) + (actionability_score * 0.2)
        
        return min(1.0, max(0.0, final_score))
    
    # Additional helper methods
    def _analyze_timeframe_optimization(self, optimal_params: Dict[str, Any]) -> List[str]:
        """Analyze timeframe optimization opportunities"""
        
        recommendations = []
        tf_performance = optimal_params.get('timeframe_performance', {})
        
        if tf_performance:
            best_tf = max(tf_performance.keys(), key=lambda k: tf_performance[k]['win_rate'])
            best_win_rate = tf_performance[best_tf]['win_rate']
            
            if best_win_rate > 0.65:
                recommendations.append(f"Focus more on {best_tf} timeframe (win rate: {best_win_rate:.1%})")
        
        return recommendations
    
    def _analyze_session_optimization(self, optimal_params: Dict[str, Any]) -> List[str]:
        """Analyze session optimization opportunities"""
        
        recommendations = []
        session_performance = optimal_params.get('session_performance', {})
        
        if session_performance:
            best_session = max(session_performance.keys(), key=lambda k: session_performance[k]['win_rate'])
            best_win_rate = session_performance[best_session]['win_rate']
            
            if best_win_rate > 0.65:
                recommendations.append(f"Prioritize {best_session} session (win rate: {best_win_rate:.1%})")
        
        return recommendations
    
    def _analyze_risk_optimization(self, risk_analysis: Dict[str, Any]) -> List[str]:
        """Analyze risk management optimization opportunities"""
        
        recommendations = []
        
        sl_hit_rate = risk_analysis.get('sl_hit_rate', 0)
        if sl_hit_rate > 0.4:
            recommendations.append("Consider wider stop losses - high SL hit rate detected")
        
        tp_hit_rates = risk_analysis.get('tp_hit_rates', {})
        if tp_hit_rates.get('tp1', 0) < 0.3:
            recommendations.append("Consider closer TP1 - low hit rate detected")
        
        return recommendations
    
    def _prioritize_parameter_optimizations(self, recommendations: List[str]) -> List[str]:
        """Prioritize parameter optimization recommendations"""
        
        # Simple prioritization based on keywords
        priority_keywords = ['stop loss', 'timeframe', 'session', 'tp1']
        prioritized = []
        
        for keyword in priority_keywords:
            for rec in recommendations:
                if keyword.lower() in rec.lower() and rec not in prioritized:
                    prioritized.append(rec)
        
        # Add remaining recommendations
        for rec in recommendations:
            if rec not in prioritized:
                prioritized.append(rec)
        
        return prioritized
    
    def _detect_volatility_regime(self) -> Dict[str, Any]:
        """Detect current volatility regime"""
        # Simplified volatility regime detection
        return {
            'current_regime': 'normal',
            'regime_strength': 0.6,
            'recent_change': False
        }
    
    def _detect_trend_regime(self) -> Dict[str, Any]:
        """Detect current trend regime"""
        # Simplified trend regime detection
        return {
            'current_regime': 'ranging',
            'regime_strength': 0.7,
            'recent_change': False
        }
    
    def _analyze_regime_performance(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy performance in different market regimes"""
        # Simplified regime performance analysis
        return {
            'trending_performance': 0.6,
            'ranging_performance': 0.7,
            'high_volatility_performance': 0.5,
            'low_volatility_performance': 0.8
        }
    
    def _calculate_regime_stability(self) -> float:
        """Calculate market regime stability"""
        return 0.8  # Assume stable regime
    
    def _assess_adaptation_needs(self, volatility_regime: Dict, trend_regime: Dict) -> bool:
        """Assess if strategy adaptation is needed for regime changes"""
        return volatility_regime.get('recent_change', False) or trend_regime.get('recent_change', False)
    
    def _get_risk_mitigation_recommendations(self, risk_level: str) -> List[str]:
        """Get risk mitigation recommendations"""
        
        if risk_level == "HIGH":
            return [
                "Implement changes gradually over multiple optimizations",
                "Monitor performance closely after implementation",
                "Consider rolling back if performance degrades"
            ]
        elif risk_level == "MEDIUM":
            return [
                "Monitor performance after implementation",
                "Be prepared to adjust if needed"
            ]
        else:
            return ["Low risk change - proceed with implementation"]

class AutoLearningSystem:
    """Automated learning system that runs periodically"""
    
    def __init__(self):
        self.optimizer = StrategyOptimizer()
        self.last_learning_run = None
        self.learning_interval_hours = 24
        
    def should_run_learning(self) -> bool:
        """Check if learning should run now"""
        if not self.last_learning_run:
            return True
        
        time_since_last = datetime.now() - self.last_learning_run
        return time_since_last.total_seconds() > (self.learning_interval_hours * 3600)
    
    def run_automated_learning(self) -> Dict[str, Any]:
        """Run automated learning process"""
        
        if not self.should_run_learning():
            return {'skipped': True, 'reason': 'Too soon since last run'}
        
        logger.info("🧠 Running automated learning process...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_result': None,
            'learning_insights': None,
            'actions_taken': []
        }
        
        try:
            # 1. Run learning analysis
            learning_insights = self.optimizer.learn_from_recent_performance(days=14)
            results['learning_insights'] = learning_insights
            
            # 2. Optimize strategy weights if needed
            optimization_result = self.optimizer.optimize_strategy_weights()
            results['optimization_result'] = optimization_result
            
            if optimization_result and optimization_result.confidence >= self.optimizer.confidence_threshold:
                results['actions_taken'].append('Updated strategy weights')
                logger.info(f"🧠 Strategy weights updated with {optimization_result.confidence:.0%} confidence")
            
            # 3. Log key insights and recommendations
            if learning_insights.get('recommendations'):
                for rec in learning_insights['recommendations'][:3]:
                    logger.info(f"💡 Learning recommendation: {rec}")
                results['actions_taken'].append('Generated learning recommendations')
            
            self.last_learning_run = datetime.now()
            logger.info("✅ Automated learning process completed")
            
        except Exception as e:
            logger.error(f"❌ Automated learning failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status"""
        
        next_run = None
        if self.last_learning_run:
            next_run = self.last_learning_run + timedelta(hours=self.learning_interval_hours)
        
        return {
            'last_learning_run': self.last_learning_run.isoformat() if self.last_learning_run else None,
            'next_scheduled_run': next_run.isoformat() if next_run else 'ASAP',
            'should_run_now': self.should_run_learning(),
            'optimization_history_count': len(self.optimizer.optimization_history),
            'current_weights': settings.strategy_weights.to_dict(),
            'learning_velocity': self.optimizer._calculate_learning_velocity()
        }

# Global instances
strategy_optimizer = StrategyOptimizer()
auto_learning_system = AutoLearningSystem()

# Convenience functions
def optimize_strategy_weights(force: bool = False) -> Optional[OptimizationResult]:
    """Optimize strategy weights based on performance"""
    return strategy_optimizer.optimize_strategy_weights(force)

def learn_from_performance(days: int = 7) -> Dict[str, Any]:
    """Learn from recent performance"""
    return strategy_optimizer.learn_from_recent_performance(days)

def run_automated_learning() -> Dict[str, Any]:
    """Run the automated learning process"""
    return auto_learning_system.run_automated_learning()

def get_optimization_history(limit: int = 10) -> List[OptimizationResult]:
    """Get recent optimization history"""
    return strategy_optimizer.get_optimization_history(limit)

def get_learning_status() -> Dict[str, Any]:
    """Get learning system status"""
    return auto_learning_system.get_learning_status()

def analyze_strategy_evolution() -> Dict[str, Any]:
    """Analyze strategy evolution over time"""
    return strategy_optimizer.analyze_strategy_evolution()

def backtest_proposed_weights(proposed_weights: Dict[str, float], days_back: int = 30) -> Dict[str, Any]:
    """Backtest proposed weight changes"""
    return strategy_optimizer.backtest_weight_changes(proposed_weights, days_back) sum(constrained_weights.values())
        if total_weight > 0:
            for strategy in constrained_weights:
                constrained_weights[strategy] = constrained_weights[strategy] / total_weight
        
        # Ensure no strategy gets too dominant
        max_single_weight = 0.35
        while max(constrained_weights.values()) > max_single_weight:
            # Find the dominant strategy
            dominant_strategy = max(constrained_weights.keys(), key=lambda k: constrained_weights[k])
            excess = constrained_weights[dominant_strategy] - max_single_weight
            constrained_weights[dominant_strategy] = max_single_weight
            
            # Redistribute excess to other strategies
            other_strategies = [k for k in constrained_weights.keys() if k != dominant_strategy]
            redistribution = excess / len(other_strategies)
            for strategy in other_strategies:
                constrained_weights[strategy] += redistribution
        
        return constrained_weights
    
    def _calculate_improvement_score(self, strategy_feedback: Dict[str, Any],
                                   old_weights: Dict[str, float], 
                                   new_weights: Dict[str, float]) -> float:
        """Calculate expected improvement from weight changes"""
        
        old_score = 0
        new_score = 0
        total_weight = 0
        
        for strategy, feedback in strategy_feedback.items():
            if strategy not in old_weights or strategy not in new_weights:
                continue
            
            # Strategy performance score (0-1)
            win_rate = feedback['win_rate']
            avg_pips = feedback['avg_pips']
            normalized_pips = max(0, min(1, (avg_pips + 50) / 100))
            performance_score = (win_rate * 0.6) + (normalized_pips * 0.4)
            
            # Weight contribution to overall score
            old_contribution = old_weights[strategy] * performance_score
            new_contribution = new_weights[strategy] * performance_score
            
            old_score += old_contribution
            new_score += new_contribution
            total_weight += new_weights[strategy]
        
        # Normalize scores
        if total_weight > 0:
            old_score /= sum(old_weights[s] for s in strategy_feedback.keys() if s in old_weights)
            new_score /= total_weight
        
        return (new_score - old_score) / old_score if old_score > 0 else 0
    
    def _calculate_optimization_confidence(self, strategy_feedback: Dict[str, Any], 
                                         total_trades: int) -> float:
        """Calculate confidence in the optimization"""
        
        # Base confidence on sample size
        sample_confidence = min(1.0, total_trades / 100)  # Full confidence at 100+ trades
        
        # Confidence based on performance consistency
        win_rates = [data['win_rate'] for data in strategy_feedback.values()]
        consistency_confidence = 1.0 - (np.std(win_rates) / np.mean(win_rates)) if np.mean(win_rates) > 0 else 0.5
        consistency_confidence = max(0, min(1, consistency_confidence))
        
        # Confidence based on data recency
        recency_confidence = 0.9  # Assume recent data for now
        
        # Confidence based on market conditions stability
        market_confidence = self._assess_market_stability()
        
        # Combined confidence
        weights = [0.3, 0.3, 0.2, 0.2]  # sample, consistency, recency, market
        confidences = [sample_confidence, consistency_confidence, recency_confidence, market_confidence]
        
        overall_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _generate_optimization_reasoning(self, strategy_feedback: Dict[str, Any],
                                       old_weights: Dict[str, float],
                                       new_weights: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning for optimization"""
        
        reasoning = []
        
        for strategy, feedback in strategy_feedback.items():
            if strategy not in old_weights or strategy not in new_weights:
                continue
            
            old_weight = old_weights[strategy]
            new_weight = new_weights[strategy]
            change_pct = ((new_weight - old_weight) / old_weight * 100) if old_weight > 0 else 0
            
            if abs(change_pct) > 3:  # Significant change (>3%)
                win_rate = feedback['win_rate']
                avg_pips = feedback['avg_pips']
                trades = feedback['trades']
                
                strategy_name = strategy.replace('_', ' ').title()
                
                if change_pct > 0:
                    reasoning.append(
                        f"Increased {strategy_name} weight by {change_pct:.1f}% "
                        f"(Win rate: {win_rate:.1%}, Avg pips: {avg_pips:+.1f}, {trades} trades)"
                    )
                else:
                    reasoning.append(
                        f"Reduced {strategy_name} weight by {abs(change_pct):.1f}% "
                        f"(Win rate: {win_rate:.1%}, Avg pips: {avg_pips:+.1f}, {trades} trades)"
                    )
        
        if not reasoning:
            reasoning.append("Minor weight adjustments based on recent performance patterns")
        
        # Add market context if relevant
        market_context = self._get_market_context_for_reasoning()
        if market_context:
            reasoning.append(f"Market context: {market_context}")
        
        return reasoning[:5]  # Limit to top 5 reasons
    
    def _update_strategy_weights(self, new_weights: Dict[str, float]):
        """Update the global strategy weights"""
        
        # Update settings object
        for strategy, weight in new_weights.items():
            if hasattr(settings.strategy_weights, strategy):
                setattr(settings.strategy_weights, strategy, weight)
        
        # Normalize to ensure sum = 1.0
        settings.strategy_weights.normalize()
        
        # Save to file
        self._save_strategy_weights()
        
        # Record the optimization
        self.last_optimization_date = datetime.now()
    
    def _save_strategy_weights(self):
        """Save strategy weights to file"""
        try:
            weights_data = {
                'weights': settings.strategy_weights.to_dict(),
                'last_updated': datetime.now().isoformat(),
                'optimization_count': len(self.optimization_history),
                'optimization_history_summary': [
                    {
                        'date': opt.optimization_date.isoformat(),
                        'improvement': opt.improvement_score,
                        'confidence': opt.confidence
                    }
                    for opt in self.optimization_history[-10:]  # Last 10 optimizations
                ]
            }
            
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2)
            
            logger.debug("💾 Strategy weights saved to file")
            
        except Exception as e:
            logger.error(f"❌ Failed to save strategy weights: {e}")
    
    def _load_optimization_history(self):
        """Load optimization history from file"""
        try:
            history_file = settings.storage.base_dir / 'optimization_history.json'
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.optimization_history = []
            for item in data.get('optimizations', []):
                item['optimization_date'] = datetime.fromisoformat(item['optimization_date'])
                self.optimization_history.append(OptimizationResult(**item))
            
            logger.info(f"📁 Loaded {len(self.optimization_history)} optimization records")
            
        except Exception as e:
            logger.error(f"❌ Failed to load optimization history: {e}")
    
    def _save_optimization_history(self):
        """Save optimization history to file"""
        try:
            history_file = settings.storage.base_dir / 'optimization_history.json'
            
            data = {
                'optimizations': [],
                'last_updated': datetime.now().isoformat(),
                'total_optimizations': len(self.optimization_history)
            }
            
            for result in self.optimization_history:
                result_dict = asdict(result)
                result_dict['optimization_date'] = result.optimization_date.isoformat()
                data['optimizations'].append(result_dict)
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("💾 Optimization history saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save optimization history: {e}")
    
    def _analyze_strategy_effectiveness(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which strategies are most effective"""
        
        strategy_feedback = learning_data.get('strategy_feedback', {})
        
        if not strategy_feedback:
            return {}
        
        # Calculate effectiveness scores for each strategy
        strategy_rankings = []
        
        for strategy, data in strategy_feedback.items():
            # Multi-factor effectiveness score
            win_rate_score = data['win_rate']
            profitability_score = max(0, min(1, (data['avg_pips'] + 50) / 100))
            volume_score = min(1.0, data['trades'] / 20)  # Normalize by expected volume
            
            # Weighted effectiveness score
            effectiveness_score = (
                win_rate_score * 0.5 +
                profitability_score * 0.3 +
                volume_score * 0.2
            )
            
            strategy_rankings.append({
                'strategy': strategy,
                'effectiveness_score': effectiveness_score,
                'win_rate': data['win_rate'],
                'avg_pips': data['avg_pips'],
                'trades': data['trades'],
                'trend': self._calculate_strategy_trend(strategy, data)
            })
        
        # Sort by effectiveness
        strategy_rankings.sort(key=lambda x: x['effectiveness_score'], reverse=True)
        
        # Identify top and bottom performers
        top_performers = strategy_rankings[:2]  # Top 2
        bottom_performers = strategy_rankings[-2:]  # Bottom 2
        
        # Calculate performance spread
        performance_spread = (strategy_rankings[0]['effectiveness_score'] - 
                            strategy_rankings[-1]['effectiveness_score'])
        
        return {
            'best_strategies': top_performers,
            'worst_strategies': bottom_performers,
            'all_rankings': strategy_rankings,
            'performance_spread': performance_spread,
            'effectiveness_summary': {
                'highest_score': strategy_rankings[0]['effectiveness_score'],
                'lowest_score': strategy_rankings[-1]['effectiveness_score'],
                'average_score': np.mean([s['effectiveness_score'] for s in strategy_rankings])
            }
        }
    
    def _analyze_parameter_optimization(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter optimization opportunities"""
        
        optimal_params = learning_data.get('optimal_parameters', {})
        risk_analysis = learning_data.get('risk_analysis', {})
        
        recommendations = []
        
        # Timeframe analysis
        timeframe_recommendations = self._analyze_timeframe_optimization(optimal_params)
        recommendations.extend(timeframe_recommendations)
        
        # Session analysis
        session_recommendations = self._analyze_session_optimization(optimal_params)
        recommendations.extend(session_recommendations)
        
        # Risk management analysis
        risk_recommendations = self._analyze_risk_optimization(risk_analysis)
        recommendations.extend(risk_recommendations)
        
        return {
            'parameter_recommendations': recommendations,
            'timeframe_analysis': optimal_params.get('timeframe_performance', {}),
            'session_analysis': optimal_params.get('session_performance', {}),
            'risk_analysis': risk_analysis,
            'optimization_priority': self._prioritize_parameter_optimizations(recommendations)
        }
    
    def _analyze_market_regime_changes(self, learning_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market regime changes and their impact"""
        
        recent_performance = learning_data.get('recent_performance', {})
        
        # Detect volatility regime changes
        volatility_regime = self._detect_volatility_regime()
        
        # Detect trend regime changes
        trend_regime = self._detect_trend_regime()
        
        # Analyze strategy performance in different regimes
        regime_performance = self._analyze_regime_performance(learning_data)
        
        return {
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'regime_performance': regime_performance,
            'regime_stability': self._calculate_regime_stability(),
            'adaptation_needed': self._assess_adaptation_needs(volatility_regime, trend_regime)
        }
    
    def _generate_learning_insights(self, strategy_analysis: Dict[str, Any], 
                                  parameter_analysis: Dict[str, Any],
                                  market_regime_analysis: Dict[str, Any]) -> List[LearningInsight]:
        """Generate actionable learning insights"""
        
        insights = []
        
        # Strategy insights
        if strategy_analysis.get('best_strategies'):
            best_strategy = strategy_analysis['best_strategies'][0]
            insights.append(LearningInsight(
                insight_type='strategy',
                description=f"{best_strategy['strategy'].replace('_', ' ').title()} shows exceptional performance",
                impact_score=best_strategy['effectiveness_score'],
                recommended_action=f"Increase allocation to {best_strategy['strategy']} strategy",
                supporting_data={
                    'win_rate': best_strategy['win_rate'],
                    'avg_pips': best_strategy['avg_pips'],
                    'trades': best_strategy['trades']
                }
            ))
        
        if strategy_analysis.get('worst_strategies'):
            worst_strategy = strategy_analysis['worst_strategies'][-1]
            if worst_strategy['effectiveness_score'] < 0.4:
                insights.append(LearningInsight(
                    insight_type='strategy',
                    description=f"{worst_strategy['strategy'].replace('_', ' ').title()} underperforming",
                    impact_score=1 - worst_strategy['effectiveness_score'],
                    recommended_action=f"Reduce allocation to {worst_strategy['strategy']} strategy",
                    supporting_data={
                        'win_rate': worst_strategy['win_rate'],
                        'avg_pips': worst_strategy['avg_pips'],
                        'trades': worst_strategy['trades']
                    }
                ))
        
        # Parameter insights
        param_recs = parameter_analysis.get('parameter_recommendations', [])
        for rec in param_recs[:2]:  # Top 2 parameter recommendations
            insights.append(LearningInsight(
                insight_type='timing',
                description=rec,
                impact_score=0.6,  # Medium impact
                recommended_action=f"Implement parameter adjustment: {rec}",
                supporting_data={}
            ))
        
        # Market regime insights
        if market_regime_analysis.get('adaptation_needed', False):
            insights.append(LearningInsight(
                insight_type='market',
                description="Market regime change detected - strategy adaptation recommended",
                impact_score=0.8,
                recommended_action="Adjust strategy weights for new market conditions",
                supporting_data=market_regime_analysis
            ))
        
        return sorted(insights, key=lambda x: x.impact_score, reverse=True)
    
    def _generate_learning_recommendations(self, insights: List[LearningInsight]) -> List[str]:
        """Generate overall learning recommendations"""
        
        recommendations = []
        
        # High-impact insights first
        high_impact_insights = [i for i in insights if i.impact_score > 0.7]
        for insight in high_impact_insights:
            recommendations.append(insight.recommended_action)
        
        # Add general recommendations
        if len(insights) > 3:
            recommendations.append("Multiple learning opportunities detected - consider gradual implementation")
        
        if not high_impact_insights:
            recommendations.append("Performance is stable - maintain current strategy mix")
        
        return recommendations[:5]  # Top 5 recommendations
    
    # Helper methods for analysis
    def _calculate_strategy_trend(self, strategy: str, feedback: Dict[str, Any]) -> float:
        """Calculate trend factor for strategy (recent vs historical performance)"""
        # Simplified trend calculation
        # In practice, this would compare recent performance to historical average
        return 1.0  # Neutral trend
    
    def _assess_market_stability(self) -> float:
        """Assess market stability for confidence calculation"""
        # Simplified market stability assessment
        return 0.8  # Assume stable market
    
    def _get_market_context_for_reasoning(self) -> str:
        """Get market context for optimization reasoning"""
        # Simplified market context
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 16:
            return "Active London session conditions"
        elif 13 <= current_hour <= 21:
            return "NY session overlap detected"
        return ""
    
    def _calculate_next_optimization_date(self) -> datetime:
        """Calculate when next optimization should occur"""
        if not self.optimization_history:
            return datetime.now() + timedelta(days=7)
        
        last_opt = max(self.optimization_history, key=lambda x: x.optimization_date)
        
        # Adaptive scheduling based on performance
        if last_opt.improvement_score > 0.1:
            days_until_next = 14  # Good improvement, wait longer
        elif last_opt.improvement_score > 0.05:
            days_until_next = 10  # Moderate improvement
        else:
            days_until_next = 7   # Minimal improvement, try again soon
        
        return last_opt.optimization_date + timedelta(days=days_until_next)
    
    def _calculate_weighted_performance_score(self, strategy_performances: Dict[str, Any], 
                                            weights: Dict[str, float]) -> float:
        """Calculate weighted performance score"""
        
        total_score = 0
        total_weight =