"""
    Supertrend + CCI Strategy - LearningBacktester Framework
    ---------------------------------------------------------

    A trend-following strategy that uses Supertrend for trend direction and
    CCI for confirming momentum and identifying potential price reversals.

    Entry Rules:
    - Long: Supertrend turns bullish (direction changes to 1) AND CCI crosses above -100
    - Short: Not implemented

    Exit Rules:
    - Exit Long: Supertrend turns bearish or CCI crosses above +200 (extreme overbought)
      (handled by risk management)
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class SupertrendCCIStrategy(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for Supertrend + CCI strategy"""
        # Get features with correct naming convention
        supertrend = self.instruments_data.get_feature("SUPERTREND_10_3.0")
        supertrend_dir = self.instruments_data.get_feature("SUPERTREND_10_3.0_DIR")
        cci = self.instruments_data.get_feature("CCI_20_0.015")
        
        # Previous values for change detection
        prev_supertrend_dir = supertrend_dir.shift(1)
        prev_cci = cci.shift(1)
        
        # Create signals DataFrame
        entry_signals = pd.DataFrame(0.0, index=supertrend.index, columns=supertrend.columns)
        
        # Calculate valid data mask
        valid_data = (
            supertrend.notna() & 
            supertrend_dir.notna() & 
            cci.notna() & 
            prev_supertrend_dir.notna() & 
            prev_cci.notna()
        )
        
        # Supertrend turns bullish (direction changes to 1)
        supertrend_bullish = (supertrend_dir > 0)
        
        # CCI crosses above -100 (from oversold to normal)
        cci_oversold_exit = (cci >= -100)
        
        # Combined signal: Supertrend bullish AND CCI crosses above -100
        raw_signals = (supertrend_bullish & cci_oversold_exit).astype(float)
        
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
        "strategy_name": "Supertrend_CCI_Strategy",
        "initial_capital": 100000,
        "instruments": ["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
        "trading_range": {	"start": "2018-01-01",  "end": "2024-06-30"}, 
        "risk_management": {
            "global": {  
                "take_profit": 0.25,  
                "stop_loss": 0.08,  
                "holding_days": 30,  
                "max_position_weight": 0.25,
                "target_volatility": 0.0,
            },
        },
        "indicators_config": [
            {
                "function": "SUPERTREND", 
                "price_cols": ["high", "low", "close"], 
                "params": {
                    "period": 10,
                    "multiplier": 3.0
                },
                "color": ["#FF5733", "#33A1FF"],
                "chart": [1, 1],  # Display on main price chart
            },
            {
                "function": "CCI", 
                "price_cols": ["high", "low", "close"], 
                "params": {
                    "periods": 20,
                    "constant": 0.015
                },
                "color": ["#A233FF"],
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
    strategy = SupertrendCCIStrategy(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())