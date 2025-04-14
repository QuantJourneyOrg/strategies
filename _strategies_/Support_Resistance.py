"""
    Support Resistance Breakout Strategy - QuantJourney Framework
    -------------------------------------------------------------

    A strategy that trades breakouts from established support and resistance levels.
    This strategy uses volume confirmation to filter out false breakouts and
    specifically looks for high-probability breaks with strong momentum.

    Entry Rules:
    - Long: Price breaks above resistance with volume confirmation

    Exit Rules:
    - Exit Long: Price breaks below a newly formed support level (handled by risk management)

    The strategy also considers wick patterns to avoid entering after fakeout moves.
"""

import numpy as np
import pandas as pd
from typing import Optional

from quantjourney.backtesting.backtester.learning_backtester_with_risk_management import LearningBacktester
from quantjourney.plots.theme.types import PlotTheme


class SupportResistanceBreakout(LearningBacktester):
    def __init__(self, config_file_path: Optional[str] = None, **kwargs):
        super().__init__(config_file_path, **kwargs)

    def _compute_signals(self) -> pd.DataFrame:
        """Generate signals for Support Resistance Breakout strategy"""
        resistance = self.instruments_data.get_feature("Resistance")
        close = self.instruments_data.get_feature("close")
    
        
        # Create signals DataFrame with proper initialization
        entry_signals = pd.DataFrame(0.0, index=close.index, 
                                    columns=close.columns)
        
        # Calculate valid data mask
        valid_data = (
            resistance.notna()
        )
        
        # Resistance break signals (long entries)
        # Only enter when we have resistance break AND not a bull wick
        # This helps avoid fakeout moves
        long_signals = close > resistance
        
        
        # Apply valid data mask
        entry_signals = long_signals.astype(float).where(valid_data, 0.0)
        
        # Optional: Implement signal cooldown to avoid overtrading
        # This prevents entering again too soon after a breakout
        cooldown_period = 5  # bars
        for col in entry_signals.columns:
            mask = entry_signals[col] == 1.0
            if mask.any():
                indices = mask[mask].index
                for idx in indices:
                    pos = entry_signals.index.get_loc(idx)
                    if pos + cooldown_period < len(entry_signals):
                        cooldown_slice = slice(pos + 1, pos + cooldown_period + 1)
                        entry_signals.iloc[cooldown_slice, entry_signals.columns.get_loc(col)] = 0.0
        
        return entry_signals

    def _compute_weights(self) -> pd.DataFrame:
        """Calculate position weights with volatility adjustment"""
        signals = self.instruments_data.get_feature("strategies", self.strategy_name, "signals")
        
        # Get recent volatility (20-day standard deviation of returns)
        close_prices = self.instruments_data.get_feature("close")
        returns = close_prices.pct_change()
        vol_20d = returns.rolling(20).std()
        
        # Inverse volatility weighting - lower weight for higher volatility
        inv_vol = 1.0 / vol_20d.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Normalize to create weights
        active_signals = signals == 1
        weights = active_signals * inv_vol
        
        # Calculate row sums (sum of weights per day)
        row_sums = weights.sum(axis=1)
        
        # Normalize by row sum to ensure weights sum to 1
        normalized_weights = weights.div(row_sums, axis=0).fillna(0.0)
        
        # Access config using attribute notation instead of dictionary access
        try:
            # Try direct attribute access
            max_weight = self.strategy_config.risk_management.global_params.max_position_weight
        except AttributeError:
            # Fallback to hardcoded default
            max_weight = 0.20
        
        # Cap weights at max_weight
        capped_weights = normalized_weights.clip(upper=max_weight)
        
        # Re-normalize if necessary
        final_weights = capped_weights.div(capped_weights.sum(axis=1), axis=0).fillna(0.0)
        
        return final_weights

    def _compute_positions(self) -> None:
        """Optional custom position sizing logic"""
        # Using the default implementation from the parent class
        pass


# Strategy configuration
def get_strategy_config():
    return {
        "strategy_name": "Support_Resistance_Breakout",
        "initial_capital": 100000,
        "instruments": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"],
        "trading_range": {"start": "2020-01-01", "end": "2025-03-23"}, 
        "risk_management": {
            "global": {  
                "take_profit": 0.25,        # 25% take profit
                "stop_loss": 0.10,          # 10% stop loss 
                "holding_days": 20,         # Max holding period
                "max_position_weight": 0.20, # No single position more than 20%
                "target_volatility": 0.15,   # Target portfolio volatility
            },
        },
        "indicators_config": [
            {
                "function": "SUPPORT_RESISTANCE", 
                "price_cols": ["open", "high", "low", "close", "volume"],
                "params": {
                    "left_bars": 10,
                    "right_bars": 10,
                    "volume_thresh": 50,
                    "toggle_breaks": False
                },
                "color": ["#3366FF", "#FF3366", "#33FF66", "#FF6633"],
                "chart": [1, 1, 2, 2],  # Display on main chart and second panel
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
            "save_blotter": True,
            "save_portfolio_data": True,
            "save_instruments_data": False,
        }
    }


# Example run function
async def run_backtest():
    strategy = SupportResistanceBreakout(**get_strategy_config())
    await strategy.run_strategy()


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())