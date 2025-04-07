"""
    Bollinger Bands + Stochastic Strategy - LearningBacktester Framework
    ---------------------------------------------------------

    This strategy combines Bollinger Bands for market volatility context with the
    Stochastic Oscillator for momentum confirmation.

    Entry Rules:
    - Long: Price crosses below lower Bollinger Band AND Stochastic K line crosses above 20
    - Short: Not implemented

    Exit Rules:
    - Exit Long: Price reaches upper Bollinger Band or Stochastic K line crosses above 80
      (handled by risk management)
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class BollingerStochasticStrategy(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for Bollinger Bands + Stochastic strategy"""
        # Get indicators with correct naming convention
        bb_lower = self.instruments_data.get_feature("BB_20_2.0_LOWER_close")
        bb_mid = self.instruments_data.get_feature("BB_20_2.0_MID_close")
        price = self.instruments_data.get_feature("close")
        stoch_k = self.instruments_data.get_feature("STOCH_K_14_3")
        
        # Previous values for crossover detection
        prev_price = price.shift(1)
        prev_bb_lower = bb_lower.shift(1)
        prev_stoch_k = stoch_k.shift(1)
        
        # Create signals DataFrame
        entry_signals = pd.DataFrame(0.0, index=price.index, columns=price.columns)
        
        # Calculate valid data mask
        valid_data = (
            bb_lower.notna() & 
            price.notna() & 
            stoch_k.notna() & 
            prev_price.notna() & 
            prev_bb_lower.notna() & 
            prev_stoch_k.notna()
        )
        
        # Price crossing below lower BB
        price_cross_below_bb = (prev_price > prev_bb_lower)
        
        # Stochastic K crossing above 20
        stoch_cross_above_20 = (prev_stoch_k < 20)
        
        # Combined signal
        raw_signals = (price_cross_below_bb & stoch_cross_above_20).astype(float)
        
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
        "strategy_name": "Bollinger_Stochastic_Strategy",
        "initial_capital": 100000,
        "instruments": ["AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN"],
        "trading_range": {"start": "2015-01-01",  "end": "2024-06-30"}, 
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
                "function": "BB", 
                "price_cols": ["close"], 
                "params": {
                    "periods": 20,
                    "stds": 2.0
                },
                "color": ["#95C5ED", "#666666", "#ED95C5"],
                "chart": [1, 1, 1],  # Display on main price chart
            },
            {
                "function": "STOCH", 
                "price_cols": ["high", "low", "close"], 
                "params": {
                    "k_periods": 14,
                    "d_periods": 3
                },
                "color": ["#FF9500", "#4287f5"],
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
    
    
# Example run function
async def run_backtest():
    strategy = BollingerStochasticStrategy(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())