"""
    MACD + RSI Confirmation Strategy - LearningBacktester Framework
    ---------------------------------------------------------

    A classic combination strategy that waits for both MACD and RSI to confirm a signal 
    before entering a trade.

    Entry Rules:
    - Long: MACD line crosses above signal line AND RSI crosses above 40
    - Short: MACD line crosses below signal line AND RSI crosses below 60 (not implemented)

    Exit Rules:
    - Exit Long: RSI above 70 or MACD crosses back below signal line (handled by risk management)
    - Exit Short: RSI below 30 or MACD crosses back above signal line (not implemented)
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class MACDRSIConfirmation(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for MACD and RSI confirmation strategy"""
        # Get features with correct naming convention
        macd_line = self.instruments_data.get_feature("MACD_12_26_9_MACD_close")
        signal_line = self.instruments_data.get_feature("MACD_12_26_9_SIG_close")
        rsi = self.instruments_data.get_feature("RSI_14_close")
        
        # Create signals DataFrame with proper initialization
        entry_signals = pd.DataFrame(0.0, index=macd_line.index, columns=macd_line.columns)
        
        # Calculate valid data mask
        valid_data = (
            macd_line.notna() & 
            signal_line.notna() & 
            rsi.notna()
        )
        
        # MACD crossover (MACD line crosses above signal line)
        macd_signal_cross = (macd_line > signal_line)
        
        # RSI crosses above 40
        rsi_cross_above_40 = (rsi >= 40)
        
        # Combined signal: MACD over signal AND RSI crosses above 40
        raw_signals = (macd_signal_cross & rsi_cross_above_40).astype(float)
        
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


# Example strategy configuration
def get_strategy_config():
    return {
        "strategy_name": "MACD_RSI_Confirmation",
        "initial_capital": 100000,
        "instruments": ["V", "MA", "PYPL", "WMT", "COST", "TGT", "HD", "LOW"],
		"trading_range": {	"start": "2018-01-01",  "end": "2024-06-30"}, 
        "risk_management": {
            "global": {  
                "take_profit": 0.3,  
                "stop_loss": 0.12,  
                "holding_days": 30,  
                "max_position_weight": 0.25,
                "target_volatility": 0.0,
            },
        },
        "indicators_config": [
            {
                "function": "MACD", 
                "price_cols": ["close"], 
                "params": {
                    "fast_periods": 12,
                    "slow_periods": 26,
                    "signal_periods": 9
                },
                "color": ["#FF5733", "#33A1FF", "#33FF57"],
                "chart": [2, 2, 2],  # Display on second chart panel
            },
            {
                "function": "RSI", 
                "price_cols": ["close"], 
                "params": {"periods": 14},
                "color": ["#A233FF"],
                "chart": [3, 3, 3],  # Display on third chart panel
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
    strategy = MACDRSIConfirmation(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())