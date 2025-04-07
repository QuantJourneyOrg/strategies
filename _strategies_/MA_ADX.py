"""
    Moving Average Crossover + ADX Strategy - LearningBacktester Framework
    ---------------------------------------------------------

    A trend-following strategy that uses two moving averages to identify trend direction
    and ADX to confirm trend strength.

    Entry Rules:
    - Long: Fast EMA (20) crosses above Slow EMA (50) AND ADX is above 25 (strong trend)
    - Short: Not implemented

    Exit Rules:
    - Exit Long: Fast EMA crosses back below Slow EMA (handled by risk management)
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class MACrossADXStrategy(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for Moving Average Crossover + ADX strategy"""
        # Get features with correct naming convention
        fast_ema = self.instruments_data.get_feature("EMA_20_close")
        slow_ema = self.instruments_data.get_feature("EMA_50_close")
        adx = self.instruments_data.get_feature("ADX_14")
        
        # Create signals DataFrame
        entry_signals = pd.DataFrame(0.0, index=fast_ema.index, columns=fast_ema.columns)
        
        # Calculate valid data mask
        valid_data = (
            fast_ema.notna() & 
            slow_ema.notna() & 
            adx.notna() &
        )
        
        # EMA crossover (fast crosses above slow)
        ema_crossover = (fast_ema > slow_ema)
        
        # Strong trend filter
        strong_trend = adx >= 25
        
        # Combined signal: EMA crossover AND strong trend
        raw_signals = (ema_crossover & strong_trend).astype(float)
        
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
        "strategy_name": "MA_Cross_ADX_Strategy",
        "initial_capital": 100000,
        "instruments": ["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
        "trading_range": {	"start": "2018-01-01",  "end": "2024-06-30"}, 
        "risk_management": {
            "global": {  
                "take_profit": 0.25,  
                "stop_loss": 0.10,  
                "holding_days": 40,  
                "max_position_weight": 0.25,
                "target_volatility": 0.0,
            },
        },
        "indicators_config": [
            {
                "function": "EMA", 
                "price_cols": ["close"], 
                "params": {"periods": [20, 50]},
                "color": ["#FF5733", "#33A1FF"],
                "chart": [1, 1],  # Display on main price chart
            },
            {
                "function": "ADX", 
                "price_cols": ["high", "low", "close"], 
                "params": {"periods": 14},
                "color": ["#A233FF", "#33FF57", "#FF33A1"],
                "chart": [2, 2, 2],  # Display on second chart panel
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
    
# Example run function
async def run_backtest():
    strategy = MACrossADXStrategy(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())