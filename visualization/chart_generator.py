"""
XAUUSD Chart Generator - Advanced Chart Creation with Technical Analysis
Creates professional trading charts with indicators, patterns, and signal annotations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    print("⚠️ mplfinance not available - using basic matplotlib")

from config.settings import settings
from utils.logger import get_module_logger
from analysis.signal_generator import TradingSignal

logger = get_module_logger('chart_generator')

@dataclass
class ChartConfig:
    """Chart configuration settings"""
    width: int = 16
    height: int = 10
    dpi: int = 300
    style: str = 'yahoo'
    volume: bool = True
    figsize: Tuple[int, int] = (16, 10)
    tight_layout: bool = True
    
    # Colors
    up_color: str = '#26a69a'
    down_color: str = '#ef5350'
    wick_color: str = '#000000'
    volume_color: str = '#1f77b4'
    
    # Signal colors
    buy_color: str = '#00ff00'
    sell_color: str = '#ff0000'
    tp_color: str = '#ffa500'
    sl_color: str = '#ff6b6b'

class ChartAnnotator:
    """Handles adding annotations and overlays to charts"""
    
    def __init__(self, config: ChartConfig):
        self.config = config
    
    def add_signal_markers(self, ax: plt.Axes, signal: TradingSignal, df: pd.DataFrame):
        """Add signal entry, SL, and TP markers to chart"""
        try:
            x_pos = len(df) - 1
            
            # Entry marker
            entry_color = self.config.buy_color if signal.direction == 'BUY' else self.config.sell_color
            entry_marker = '▲' if signal.direction == 'BUY' else '▼'
            
            ax.scatter(x_pos, signal.entry_price, 
                      c=entry_color, s=200, marker=entry_marker, 
                      zorder=10, edgecolor='white', linewidth=2,
                      label=f'{signal.direction} Entry ${signal.entry_price:.2f}')
            
            # Stop Loss line
            ax.axhline(y=signal.stop_loss, color=self.config.sl_color, 
                      linestyle='--', alpha=0.8, linewidth=2,
                      label=f'Stop Loss ${signal.stop_loss:.2f}')
            
            # Take Profit lines
            for i, tp in enumerate(signal.take_profits, 1):
                alpha = 1.0 - (i * 0.15)
                ax.axhline(y=tp, color=self.config.tp_color, 
                          linestyle=':', alpha=alpha, linewidth=1.5,
                          label=f'TP{i} ${tp:.2f}' if i <= 2 else None)
            
            # Add entry text box
            self._add_signal_info_box(ax, signal, x_pos)
            
        except Exception as e:
            logger.error(f"❌ Failed to add signal markers: {e}")
    
    def add_support_resistance(self, ax: plt.Axes, df: pd.DataFrame):
        """Add support and resistance levels"""
        try:
            if 'pivot' in df.columns:
                latest = df.iloc[-1]
                
                # Pivot point
                if not pd.isna(latest['pivot']):
                    ax.axhline(y=latest['pivot'], color='gray', 
                              linestyle='-', alpha=0.6, linewidth=1.5,
                              label=f'Pivot ${latest["pivot"]:.2f}')
                
                # Support and Resistance levels
                levels = [('s1', 'Support 1'), ('r1', 'Resistance 1'), 
                         ('s2', 'Support 2'), ('r2', 'Resistance 2')]
                
                for level_col, level_name in levels:
                    if level_col in df.columns and not pd.isna(latest[level_col]):
                        color = 'green' if 's' in level_col else 'red'
                        ax.axhline(y=latest[level_col], color=color,
                                  linestyle='--', alpha=0.5, linewidth=1,
                                  label=f'{level_name} ${latest[level_col]:.2f}')
            
        except Exception as e:
            logger.error(f"❌ Failed to add support/resistance: {e}")
    
    def _add_signal_info_box(self, ax: plt.Axes, signal: TradingSignal, x_pos: int):
        """Add signal information box"""
        try:
            info_lines = [
                f"{signal.direction} Signal",
                f"Entry: ${signal.entry_price:.2f}",
                f"SL: ${signal.stop_loss:.2f}",
                f"Confidence: {signal.confidence:.1f}%",
                f"R:R: {signal.risk_reward_ratios[0]:.1f}:1"
            ]
            
            info_text = '\n'.join(info_lines)
            
            x_text = x_pos - len(ax.get_xlim()) * 0.25
            y_text = signal.entry_price + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1
            
            ax.text(x_text, y_text, info_text,
                   fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='lightblue', alpha=0.9,
                            edgecolor='navy', linewidth=1),
                   verticalalignment='bottom')
            
        except Exception as e:
            logger.error(f"❌ Failed to add signal info box: {e}")

class ChartGenerator:
    """Main chart generation system"""
    
    def __init__(self):
        self.config = ChartConfig()
        self.annotator = ChartAnnotator(self.config)
        self.charts_dir = settings.storage.charts_dir
        self.charts_dir.mkdir(exist_ok=True)
        
        logger.info("📊 Chart Generator initialized")
    
    def generate_signal_chart(self, signal: TradingSignal, df: pd.DataFrame, 
                            analysis_data: Dict[str, Any] = None) -> Optional[str]:
        """Generate comprehensive signal chart with all annotations"""
        
        if df.empty:
            logger.error("❌ Cannot generate chart: empty dataframe")
            return None
        
        try:
            logger.info(f"📊 Generating signal chart for {signal.signal_id}")
            
            # Prepare data for plotting
            chart_df = self._prepare_chart_data(df)
            
            if MPLFINANCE_AVAILABLE:
                return self._generate_mplfinance_chart(signal, chart_df, analysis_data)
            else:
                return self._generate_matplotlib_chart(signal, chart_df, analysis_data)
            
        except Exception as e:
            logger.error(f"❌ Chart generation failed: {e}")
            return None
    
    def _generate_mplfinance_chart(self, signal: TradingSignal, df: pd.DataFrame, 
                                 analysis_data: Dict[str, Any] = None) -> Optional[str]:
        """Generate chart using mplfinance"""
        try:
            # Create additional plots for indicators
            additional_plots = self._create_indicator_plots(df)
            
            # Chart style configuration
            mc = mpf.make_marketcolors(
                up=self.config.up_color,
                down=self.config.down_color,
                wick={'up': self.config.wick_color, 'down': self.config.wick_color},
                volume='inherit'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle='-',
                gridcolor='lightgray',
                gridwidth=0.5,
                y_on_right=True
            )
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_{signal.signal_id}_{timestamp}.png"
            filepath = self.charts_dir / filename
            
            # Create the plot
            fig, axes = mpf.plot(
                df,
                type='candle',
                style=style,
                volume=True,
                figsize=self.config.figsize,
                tight_layout=self.config.tight_layout,
                addplot=additional_plots,
                returnfig=True,
                title=f'XAUUSD {signal.direction} Signal - {signal.confidence:.1f}% Confidence',
                savefig=dict(fname=str(filepath), dpi=self.config.dpi, bbox_inches='tight')
            )
            
            # Add custom annotations
            main_ax = axes[0]
            
            # Add signal markers
            self.annotator.add_signal_markers(main_ax, signal, df)
            
            # Add support/resistance
            self.annotator.add_support_resistance(main_ax, df)
            
            # Add legend
            main_ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
            
            # Save the chart
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"✅ Signal chart saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ mplfinance chart generation failed: {e}")
            return None
    
    def _generate_matplotlib_chart(self, signal: TradingSignal, df: pd.DataFrame,
                                 analysis_data: Dict[str, Any] = None) -> Optional[str]:
        """Generate chart using basic matplotlib (fallback)"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
            
            # Main price chart
            self._plot_candlesticks_matplotlib(ax1, df)
            
            # Add signal annotations
            self.annotator.add_signal_markers(ax1, signal, df)
            self.annotator.add_support_resistance(ax1, df)
            
            # Volume chart
            if 'Volume' in df.columns:
                ax2.bar(range(len(df)), df['Volume'], alpha=0.7, color=self.config.volume_color)
                ax2.set_title('Volume')
            
            # Formatting
            ax1.set_title(f'XAUUSD {signal.direction} Signal - {signal.confidence:.1f}% Confidence')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_{signal.signal_id}_{timestamp}.png"
            filepath = self.charts_dir / filename
            
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"✅ Matplotlib chart saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Matplotlib chart generation failed: {e}")
            return None
    
    def _plot_candlesticks_matplotlib(self, ax: plt.Axes, df: pd.DataFrame):
        """Plot candlesticks using matplotlib"""
        try:
            for i, row in df.iterrows():
                idx = i if isinstance(i, int) else len(df) - len(df) + list(df.index).index(i)
                
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                
                # Determine color
                color = self.config.up_color if close_price >= open_price else self.config.down_color
                
                # Draw high-low line
                ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=1)
                
                # Draw body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            ax.set_ylabel('Price (USD)')
            ax.set_title('XAUUSD Price Chart')
            
        except Exception as e:
            logger.error(f"❌ Candlestick plotting failed: {e}")
    
    def _prepare_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for chart generation"""
        
        chart_df = df.copy()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in chart_df.columns:
                logger.error(f"❌ Missing required column: {col}")
                return pd.DataFrame()
        
        # Ensure datetime index
        if not isinstance(chart_df.index, pd.DatetimeIndex):
            if 'timestamp' in chart_df.columns:
                chart_df.index = pd.to_datetime(chart_df['timestamp'])
            else:
                chart_df.index = pd.date_range(start='2024-01-01', periods=len(chart_df), freq='15T')
        
        # Add volume if missing
        if 'volume' not in chart_df.columns:
            chart_df['volume'] = 1000
        
        # Ensure proper column names
        chart_df = chart_df.rename(columns={col: col.capitalize() for col in chart_df.columns})
        
        return chart_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def _create_indicator_plots(self, df: pd.DataFrame) -> List:
        """Create additional plots for technical indicators"""
        
        additional_plots = []
        
        # Try to add moving averages if available
        ma_columns = [col for col in df.columns if 'ema' in col.lower() or 'sma' in col.lower()]
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i, ma_col in enumerate(ma_columns[:5]):
            if ma_col in df.columns and MPLFINANCE_AVAILABLE:
                color = colors[i % len(colors)]
                try:
                    additional_plots.append(
                        mpf.make_addplot(df[ma_col], color=color, width=1.5, alpha=0.7)
                    )
                except:
                    pass
        
        # Add Bollinger Bands if available
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']) and MPLFINANCE_AVAILABLE:
            try:
                additional_plots.extend([
                    mpf.make_addplot(df['bb_upper'], color='gray', linestyle='--', alpha=0.6),
                    mpf.make_addplot(df['bb_lower'], color='gray', linestyle='--', alpha=0.6)
                ])
            except:
                pass
        
        return additional_plots
    
    def create_performance_chart(self, performance_data: Dict[str, Any]) -> Optional[str]:
        """Create performance summary chart"""
        
        try:
            logger.info("📊 Creating performance chart")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('XAUUSD Bot Performance Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Win Rate by Strategy
            strategies = list(performance_data.get('strategy_performances', {}).keys())
            if strategies:
                win_rates = [perf.win_rate * 100 for perf in performance_data.get('strategy_performances', {}).values()]
                
                bars = ax1.bar(strategies, win_rates, color=['green' if wr >= 50 else 'red' for wr in win_rates])
                ax1.set_title('Win Rate by Strategy (%)', fontweight='bold')
                ax1.set_ylabel('Win Rate (%)')
                ax1.tick_params(axis='x', rotation=45)
                
                for bar, rate in zip(bars, win_rates):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{rate:.1f}%', ha='center', va='bottom')
            
            # 2. Monthly P&L
            monthly_pnl = performance_data.get('monthly_pnl', {})
            if monthly_pnl:
                months = list(monthly_pnl.keys())
                pnl_values = list(monthly_pnl.values())
                colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
                
                ax2.bar(months, pnl_values, color=colors, alpha=0.7)
                ax2.set_title('Monthly P&L (USD)', fontweight='bold')
                ax2.set_ylabel('P&L (USD)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 3. Performance Metrics
            metrics_text = f"""
Total Trades: {performance_data.get('total_trades', 0)}
Win Rate: {performance_data.get('win_rate', 0):.1%}
Total Pips: {performance_data.get('total_pips', 0):+.1f}
Total P&L: ${performance_data.get('total_pnl_usd', 0):+.2f}
Avg Pips/Trade: {performance_data.get('avg_pips_per_trade', 0):+.1f}
Max Drawdown: {performance_data.get('max_drawdown_pct', 0):.1f}%
Profit Factor: {performance_data.get('profit_factor', 0):.2f}
Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.2f}
            """
            
            ax3.text(0.1, 0.9, metrics_text.strip(), transform=ax3.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax3.set_title('Key Performance Metrics', fontweight='bold')
            ax3.axis('off')
            
            # 4. Session Performance
            session_data = {
                'Asian': 0.5,
                'London': 0.6,
                'New York': 0.55
            }
            
            sessions = list(session_data.keys())
            rates = list(session_data.values())
            colors = ['gold' if r == max(rates) else 'lightblue' for r in rates]
            
            ax4.pie(rates, labels=sessions, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Performance by Trading Session', fontweight='bold')
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_dashboard_{timestamp}.png"
            filepath = self.charts_dir / filename
            
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"✅ Performance chart saved: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Performance chart generation failed: {e}")
            return None
    
    def cleanup_old_charts(self, days_old: int = 7):
        """Clean up old chart files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for chart_file in self.charts_dir.glob('*.png'):
                if chart_file.stat().st_mtime < cutoff_date.timestamp():
                    chart_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"🗑️ Cleaned up {deleted_count} old chart files")
            
        except Exception as e:
            logger.error(f"❌ Chart cleanup failed: {e}")
    
    def get_latest_chart_path(self, signal_id: str = None) -> Optional[str]:
        """Get path to the latest chart file"""
        try:
            pattern = f"signal_{signal_id}_*.png" if signal_id else "*.png"
            chart_files = list(self.charts_dir.glob(pattern))
            
            if not chart_files:
                return None
            
            latest_file = max(chart_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest chart: {e}")
            return None

# Global chart generator instance
chart_generator = ChartGenerator()

# Convenience functions
def generate_signal_chart(signal: TradingSignal, df: pd.DataFrame, 
                         analysis_data: Dict[str, Any] = None) -> Optional[str]:
    """Generate signal chart with annotations"""
    return chart_generator.generate_signal_chart(signal, df, analysis_data)

def create_performance_dashboard(performance_data: Dict[str, Any]) -> Optional[str]:
    """Create performance dashboard chart"""
    return chart_generator.create_performance_chart(performance_data)

def cleanup_old_charts(days_old: int = 7):
    """Clean up old chart files"""
    chart_generator.cleanup_old_charts(days_old)

def get_latest_chart(signal_id: str = None) -> Optional[str]:
    """Get latest chart path"""
    return chart_generator.get_latest_chart_path(signal_id)