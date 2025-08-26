 """
XAUUSD Price Validator - Ensures data quality and detects anomalies
Validates prices against multiple criteria to prevent false signals
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
import json

from config.settings import settings
from utils.logger import get_module_logger

logger = get_module_logger('price_validator')

@dataclass
class ValidationResult:
    """Price validation result"""
    is_valid: bool
    confidence: float  # 0-1, how confident we are in the data
    issues: List[str]
    adjusted_price: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    has_anomaly: bool
    anomaly_type: str
    severity: float  # 0-1
    description: str
    affected_bars: List[int]

class PriceRangeValidator:
    """Validates prices are within realistic ranges"""
    
    def __init__(self):
        self.config = settings.xauusd_market
        # Dynamic range based on recent market conditions
        self.adaptive_min = self.config.min_realistic_price
        self.adaptive_max = self.config.max_realistic_price
        self.last_price_update = None
    
    def validate_single_price(self, price: float, timestamp: datetime = None) -> ValidationResult:
        """Validate a single price point"""
        issues = []
        confidence = 1.0
        
        # Basic range check
        if price < self.adaptive_min:
            issues.append(f"Price ${price:.2f} below minimum ${self.adaptive_min:.2f}")
            confidence *= 0.1
        elif price > self.adaptive_max:
            issues.append(f"Price ${price:.2f} above maximum ${self.adaptive_max:.2f}")
            confidence *= 0.1
        
        # Check for obvious errors (like prices in wrong units)
        if price < 10:  # Probably in thousands, should be in dollars
            adjusted_price = price * 1000
            if self.adaptive_min <= adjusted_price <= self.adaptive_max:
                issues.append("Price appears to be in wrong units, adjusted by x1000")
                return ValidationResult(True, 0.8, issues, adjusted_price)
        
        # Check for decimal place errors
        if price > 10000:  # Too high, might have extra decimal places
            adjusted_price = price / 10
            if self.adaptive_min <= adjusted_price <= self.adaptive_max:
                issues.append("Price appears to have extra decimal place, adjusted by /10")
                return ValidationResult(True, 0.7, issues, adjusted_price)
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, confidence, issues)
    
    def update_adaptive_range(self, recent_prices: List[float]):
        """Update adaptive price range based on recent market data"""
        if not recent_prices:
            return
        
        try:
            valid_prices = [p for p in recent_prices if 1000 <= p <= 5000]
            if not valid_prices:
                return
            
            # Calculate new range based on recent data
            price_mean = np.mean(valid_prices)
            price_std = np.std(valid_prices)
            
            # Adaptive range: mean ± 3 standard deviations
            buffer_factor = 3.0
            new_min = max(1000, price_mean - buffer_factor * price_std)
            new_max = min(5000, price_mean + buffer_factor * price_std)
            
            # Only update if the new range makes sense
            if new_min < new_max and (new_max - new_min) > 500:
                self.adaptive_min = new_min
                self.adaptive_max = new_max
                self.last_price_update = datetime.now()
                
                logger.debug(f"📊 Updated adaptive price range: ${new_min:.2f} - ${new_max:.2f}")
                
        except Exception as e:
            logger.warning(f"⚠️ Adaptive range update failed: {e}")

class OHLCValidator:
    """Validates OHLC data relationships and patterns"""
    
    def validate_ohlc_relationships(self, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLC data has correct relationships"""
        issues = []
        confidence = 1.0
        invalid_bars = 0
        
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, 0.0, issues)
        
        # Check OHLC relationships
        for i, row in df.iterrows():
            bar_issues = []
            
            # High should be >= Open, Close, Low
            if row['high'] < row['open']:
                bar_issues.append("High < Open")
            if row['high'] < row['close']:
                bar_issues.append("High < Close")
            if row['high'] < row['low']:
                bar_issues.append("High < Low")
            
            # Low should be <= Open, Close, High
            if row['low'] > row['open']:
                bar_issues.append("Low > Open")
            if row['low'] > row['close']:
                bar_issues.append("Low > Close")
            if row['low'] > row['high']:
                bar_issues.append("Low > High")
            
            if bar_issues:
                issues.append(f"Bar {i}: {', '.join(bar_issues)}")
                invalid_bars += 1
        
        # Calculate confidence based on invalid bars ratio
        if len(df) > 0:
            invalid_ratio = invalid_bars / len(df)
            confidence = max(0.0, 1.0 - invalid_ratio * 2)
        
        is_valid = invalid_bars == 0
        return ValidationResult(is_valid, confidence, issues)
    
    def detect_price_gaps(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Detect unusual price gaps between bars"""
        gaps = []
        
        if len(df) < 2:
            return gaps
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            current_open = df.iloc[i]['open']
            current_close = df.iloc[i]['close']
            
            # Calculate gap percentage
            gap_pct = abs(current_open - prev_close) / prev_close * 100
            
            # Flag significant gaps (>0.5% for XAUUSD is unusual in normal timeframes)
            if gap_pct > 0.5:
                gap_type = "up_gap" if current_open > prev_close else "down_gap"
                gaps.append((i, gap_pct, gap_type))
        
        return gaps

class VolatilityValidator:
    """Validates price volatility patterns"""
    
    def __init__(self):
        self.max_single_bar_change = 0.05  # 5% max change per bar
        self.max_hourly_volatility = 0.10   # 10% max hourly volatility
    
    def validate_volatility(self, df: pd.DataFrame) -> ValidationResult:
        """Check for excessive volatility that might indicate bad data"""
        issues = []
        confidence = 1.0
        excessive_moves = 0
        
        if len(df) < 2:
            return ValidationResult(True, 1.0, [])
        
        # Calculate price changes
        df_temp = df.copy()
        df_temp['price_change'] = df_temp['close'].pct_change().abs()
        df_temp['bar_range'] = (df_temp['high'] - df_temp['low']) / df_temp['close']
        
        # Check for excessive single-bar moves
        extreme_changes = df_temp[df_temp['price_change'] > self.max_single_bar_change]
        if not extreme_changes.empty:
            excessive_moves = len(extreme_changes)
            max_change = extreme_changes['price_change'].max()
            issues.append(f"{excessive_moves} bars with excessive price changes (max: {max_change:.1%})")
        
        # Check for excessive intra-bar ranges
        extreme_ranges = df_temp[df_temp['bar_range'] > self.max_single_bar_change]
        if not extreme_ranges.empty:
            max_range = extreme_ranges['bar_range'].max()
            issues.append(f"{len(extreme_ranges)} bars with excessive ranges (max: {max_range:.1%})")
            excessive_moves += len(extreme_ranges)
        
        # Adjust confidence based on excessive moves
        if len(df) > 0:
            excessive_ratio = excessive_moves / len(df)
            confidence = max(0.2, 1.0 - excessive_ratio * 3)
        
        is_valid = excessive_moves == 0
        return ValidationResult(is_valid, confidence, issues)

class DataCompletenessValidator:
    """Validates data completeness and consistency"""
    
    def validate_completeness(self, df: pd.DataFrame, expected_timeframe: str) -> ValidationResult:
        """Check for missing bars and data gaps"""
        issues = []
        confidence = 1.0
        
        if df.empty:
            return ValidationResult(False, 0.0, ["Empty dataset"])
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts.to_dict()}")
            confidence *= 0.7
        
        # Check for duplicate timestamps
        if hasattr(df.index, 'duplicated'):
            duplicates = df.index.duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate timestamps found")
                confidence *= 0.8
        
        # Check for reasonable data length
        if len(df) < 50:
            issues.append(f"Dataset too short: {len(df)} bars (minimum 50 recommended)")
            confidence *= 0.5
        
        # Check timestamp consistency (if index is datetime)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            time_diffs = df.index.to_series().diff()[1:]
            
            # Expected time difference based on timeframe
            expected_minutes = {
                '1': 1, '5': 5, '15': 15, '30': 30, 
                '60': 60, '240': 240, '1D': 1440
            }
            
            expected_diff = timedelta(minutes=expected_minutes.get(expected_timeframe, 15))
            
            # Count significant deviations
            tolerance = expected_diff * 0.1  # 10% tolerance
            deviations = abs(time_diffs - expected_diff) > tolerance
            deviation_count = deviations.sum()
            
            if deviation_count > len(df) * 0.1:  # More than 10% deviations
                issues.append(f"{deviation_count} timestamp deviations from expected {expected_timeframe}min intervals")
                confidence *= 0.9
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, confidence, issues)

class AnomalyDetector:
    """Detects various types of data anomalies"""
    
    def __init__(self):
        self.z_score_threshold = 3.0  # Z-score threshold for outlier detection
        self.isolation_contamination = 0.1  # Expected contamination rate
    
    def detect_statistical_outliers(self, df: pd.DataFrame) -> AnomalyDetection:
        """Detect statistical outliers using Z-score and IQR methods"""
        anomalies = []
        
        if len(df) < 10:
            return AnomalyDetection(False, "statistical", 0.0, "Insufficient data for outlier detection", [])
        
        # Z-score based detection on closing prices
        z_scores = np.abs(stats.zscore(df['close']))
        outlier_indices = np.where(z_scores > self.z_score_threshold)[0].tolist()
        
        # IQR based detection
        Q1 = df['close'].quantile(0.25)
        Q3 = df['close'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = df[(df['close'] < lower_bound) | (df['close'] > upper_bound)].index.tolist()
        
        # Combine both methods
        all_outliers = list(set(outlier_indices + [df.index.get_loc(idx) for idx in iqr_outliers]))
        
        if all_outliers:
            severity = min(1.0, len(all_outliers) / len(df) * 5)  # Scale severity
            description = f"Found {len(all_outliers)} statistical outliers using Z-score and IQR methods"
            return AnomalyDetection(True, "statistical", severity, description, all_outliers)
        
        return AnomalyDetection(False, "statistical", 0.0, "No statistical outliers detected", [])

class MasterValidator:
    """Main validator that orchestrates all validation checks"""
    
    def __init__(self):
        self.price_range_validator = PriceRangeValidator()
        self.ohlc_validator = OHLCValidator()
        self.volatility_validator = VolatilityValidator()
        self.completeness_validator = DataCompletenessValidator()
        self.anomaly_detector = AnomalyDetector()
        
        # Validation history for learning
        self.validation_history = []
    
    def validate_current_price(self, price: float, timestamp: datetime = None) -> ValidationResult:
        """Comprehensive validation of current price"""
        return self.price_range_validator.validate_single_price(price, timestamp)
    
    def validate_historical_data(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Comprehensive validation of historical data"""
        validation_start = datetime.now()
        
        results = {
            'timestamp': validation_start.isoformat(),
            'timeframe': timeframe,
            'total_bars': len(df),
            'validations': {},
            'overall_score': 0.0,
            'is_usable': False,
            'recommendations': []
        }
        
        if df.empty:
            results['validations']['empty_data'] = ValidationResult(False, 0.0, ["Dataset is empty"])
            return results
        
        # Run all validations
        validations = {}
        
        # 1. OHLC relationships
        validations['ohlc'] = self.ohlc_validator.validate_ohlc_relationships(df)
        
        # 2. Volatility check
        validations['volatility'] = self.volatility_validator.validate_volatility(df)
        
        # 3. Data completeness
        validations['completeness'] = self.completeness_validator.validate_completeness(df, timeframe)
        
        # 4. Price range validation (on sample of prices)
        price_sample = df['close'].sample(min(100, len(df))).tolist()
        self.price_range_validator.update_adaptive_range(price_sample)
        
        price_validation_results = []
        for price in price_sample[:10]:  # Check first 10 prices
            result = self.price_range_validator.validate_single_price(price)
            price_validation_results.append(result)
        
        # Aggregate price validation results
        valid_prices = sum(1 for r in price_validation_results if r.is_valid)
        price_confidence = valid_prices / len(price_validation_results) if price_validation_results else 0
        price_issues = []
        for r in price_validation_results:
            price_issues.extend(r.issues)
        
        validations['price_range'] = ValidationResult(
            valid_prices == len(price_validation_results),
            price_confidence,
            list(set(price_issues))  # Remove duplicates
        )
        
        # 5. Anomaly detection
        anomaly_result = self.anomaly_detector.detect_statistical_outliers(df)
        validations['anomalies'] = ValidationResult(
            not anomaly_result.has_anomaly,
            1.0 - anomaly_result.severity,
            [anomaly_result.description] if anomaly_result.has_anomaly else []
        )
        
        # Store validation results
        results['validations'] = validations
        
        # Calculate overall score (weighted average of confidences)
        weights = {
            'ohlc': 0.25,
            'volatility': 0.20,
            'completeness': 0.25,
            'price_range': 0.20,
            'anomalies': 0.10
        }
        
        overall_score = sum(
            validations[key].confidence * weights[key]
            for key in weights.keys()
        )
        
        results['overall_score'] = overall_score
        results['is_usable'] = overall_score >= 0.7  # 70% threshold for usable data
        
        # Generate recommendations
        recommendations = []
        for key, validation in validations.items():
            if not validation.is_valid:
                if key == 'ohlc':
                    recommendations.append("Clean OHLC relationships before analysis")
                elif key == 'volatility':
                    recommendations.append("Filter extreme volatility bars")
                elif key == 'completeness':
                    recommendations.append("Fill missing data or use different timeframe")
                elif key == 'price_range':
                    recommendations.append("Verify price data source accuracy")
                elif key == 'anomalies':
                    recommendations.append("Investigate statistical outliers")
        
        if overall_score < 0.7:
            recommendations.append("Consider using different data source or timeframe")
        
        results['recommendations'] = recommendations
        
        # Log validation results
        self._log_validation_results(results)
        
        # Store for learning
        self.validation_history.append(results)
        if len(self.validation_history) > 100:  # Keep last 100 validations
            self.validation_history.pop(0)
        
        return results
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """Log validation results appropriately"""
        score = results['overall_score']
        
        if score >= 0.9:
            logger.info(f"✅ Data validation: Excellent quality ({score:.1%})")
        elif score >= 0.7:
            logger.info(f"📊 Data validation: Good quality ({score:.1%})")
        elif score >= 0.5:
            logger.warning(f"⚠️ Data validation: Moderate quality ({score:.1%})")
        else:
            logger.error(f"❌ Data validation: Poor quality ({score:.1%})")
        
        # Log specific issues
        for validation_type, validation in results['validations'].items():
            if validation.issues:
                logger.debug(f"📋 {validation_type}: {'; '.join(validation.issues)}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get statistics from recent validations"""
        if not self.validation_history:
            return {}
        
        recent_scores = [v['overall_score'] for v in self.validation_history[-20:]]
        
        return {
            'average_score': np.mean(recent_scores),
            'score_trend': np.polyfit(range(len(recent_scores)), recent_scores, 1)[0],
            'validations_count': len(self.validation_history),
            'usable_data_rate': sum(1 for v in self.validation_history[-20:] if v['is_usable']) / len(self.validation_history[-20:])
        }

# Global validator instance
validator = MasterValidator()

# Convenience functions
def validate_price(price: float) -> bool:
    """Quick price validation"""
    result = validator.validate_current_price(price)
    return result.is_valid

def validate_data_quality(df: pd.DataFrame, timeframe: str) -> float:
    """Quick data quality check - returns score 0-1"""
    results = validator.validate_historical_data(df, timeframe)
    return results['overall_score']
