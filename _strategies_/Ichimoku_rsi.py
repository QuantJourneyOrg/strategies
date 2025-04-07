"""
    Ichimoku Cloud + RSI Strategy - LearningBacktester Framework
    ---------------------------------------------------------

    This strategy uses the Ichimoku Cloud system for trend detection and 
    RSI for momentum confirmation.

    Entry Rules:
    - Long: Price crosses above the Cloud (above Senkou Span A & B) AND RSI crosses above 40
    - Short: Not implemented

    Exit Rules:
    - Exit Long: Price crosses below Kijun-sen (baseline) or RSI crosses above 70
      (handled by risk management)
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class IchimokuRSIStrategy(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for Ichimoku Cloud + RSI strategy"""
        # Get features with correct naming convention
        price = self.instruments_data.get_feature("close")
        tenkan = self.instruments_data.get_feature("Tenkan-sen")
        kijun = self.instruments_data.get_feature("Kijun-sen")
        senkou_a = self.instruments_data.get_feature("Senkou Span A")
        senkou_b = self.instruments_data.get_feature("Senkou Span B")
        rsi = self.instruments_data.get_feature("RSI_14_close")
        
        # Previous values for crossover detection
        # prev_price = price.shift(1)
        # prev_senkou_a = senkou_a.shift(1)
        # prev_senkou_b = senkou_b.shift(1)
        # prev_rsi = rsi.shift(1)
        
        # Create signals DataFrame
        entry_signals = pd.DataFrame(0.0, index=price.index, columns=price.columns)
        
        # Calculate valid data mask
        valid_data = (
            price.notna() & 
            senkou_a.notna() & 
            senkou_b.notna() & 
            rsi.notna()
            # prev_price.notna() & 
            # prev_senkou_a.notna() & 
            # prev_senkou_b.notna() & 
            # prev_rsi.notna()
        )
        
        # Price above the cloud
        price_above_cloud = (price > senkou_a) & (price > senkou_b)
        
        # Price crossing above the cloud
        price_cross_above_cloud = (
            # ((prev_price <= prev_senkou_a) | (prev_price <= prev_senkou_b)) & 
            price_above_cloud
        )
        
        # RSI crosses above 40
        rsi_cross_above_40 = (rsi >= 40)
        
        # Combined signal: Price crossing above cloud AND RSI condition
        raw_signals = (price_cross_above_cloud & rsi_cross_above_40).astype(float)
        
        # Apply valid data mask
        entry_signals = raw_signals.where(valid_data, 0.0)
        
        return entry_signals

    def _compute_weights(self) -> pd.DataFrame:
        """Calculate equal weights for active signals"""
        signals = self.instruments_data.get_feature("strategies", self.strategy_name, "signals")
        
        active_signals = signals == 1
        active_signals_sums = active_signals.sum(axis=1)
        equal_weights = active_signals.div(active_signals_sums, axis=0).fillna(0.0)
        
        return equal_weights

    def _compute_positions(self) -> None:
        pass


# Strategy configuration
def get_strategy_config():
    return {
        "strategy_name": "Ichimoku_RSI_Strategy",
        "initial_capital": 100000,
        "instruments": ["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
        "trading_range": {	"start": "2018-01-01",  "end": "2024-06-30"}, 
        "risk_management": {
            "global": {  
                "take_profit": 0.30,  
                "stop_loss": 0.10,  
                "holding_days": 45,  
                "max_position_weight": 0.25,
                "target_volatility": 0.0,
            },
        },
        "indicators_config": [
            {
                "function": "ICHIMOKU", 
                "price_cols": ["high","low","close"], 
                "params": {
                    "tenkan_period": 9,
                    "kijun_period": 26,
                    "senkou_period": 52,
                    "displacement": 26
                },
                "color": ["#FF5733", "#33A1FF", "#33FFA1", "#A133FF", "#FFA133"],
                "chart": [1, 1, 1, 1, 1],  # Display on main price chart
            },
            {
                "function": "RSI", 
                "price_cols": ["close"], 
                "params": {"periods": 14},
                "color": ["#FF9500"],
                "chart": [2, 2],  # Display on second chart panel
            },
        ],
        "market_data_provider": {
            "source": "yfinance",
            "granularity": "1d",
            "write_to_db": True,
            "read_from_db": True,
        },
        "market_data_processor": {
            "eligibility": {
                "volatility_threshold": 0.0002,
                "liquidity_threshold": 1000000,
                "active_threshold": 0.05,
                "max_return_threshold": 0.20,
            },
            "resample_freq": "1D"
        },
        "strategy_performance_analysis": {
            "show_text_reports": True,
            "save_text_reports": True,
            "save_portfolio_plots": False,
            "show_portfolio_plots": False,
            "save_instrument_plots": False,
            "show_instrument_plots": False,
            "theme_plots": PlotTheme.QUANTJOURNEY,
            "reports_directory": "./quantjourney/backtesting/reports",
            "benchmark": {
                "symbol": "^GSPC",
                "name": "S&P 500 Index",
            }
        },
        "interactive_dashboard": {
            "create_interactive_dashboard": True,
            "theme": "dark"
        },
        "strategy_archive": {
            "save_blotter": False,
            "save_portfolio_data": False,
            "save_instruments_data": False,
        }
    }
    
async def run_backtest():
    strategy = IchimokuRSIStrategy(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())