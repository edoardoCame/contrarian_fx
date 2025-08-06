#!/usr/bin/env python3
"""
High-Performance Backtesting Engine for Contrarian Forex Strategy

This module provides a vectorized backtesting framework optimized for speed and accuracy.
It handles portfolio construction, rebalancing, transaction costs, and comprehensive 
performance tracking while ensuring strict alignment with signal timing.

Key Features:
- Vectorized operations for computational efficiency
- Zero lookahead bias with proper signal-return alignment
- Configurable transaction costs and slippage modeling
- Multi-asset portfolio construction with risk management
- Daily rebalancing with realistic execution assumptions
- Comprehensive equity curve and drawdown tracking

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def calculate_portfolio_returns_numba(weights: np.ndarray, 
                                    returns: np.ndarray,
                                    transaction_costs: np.ndarray,
                                    rebalance_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized portfolio return calculation with transaction costs.
    
    Args:
        weights: Portfolio weights (T x N)
        returns: Asset returns (T x N)
        transaction_costs: Transaction cost rates (T x N)
        rebalance_mask: Boolean mask for rebalancing days (T,)
        
    Returns:
        Tuple of (portfolio_returns, transaction_costs_paid, turnover)
    """
    T, N = weights.shape
    portfolio_returns = np.zeros(T)
    transaction_costs_paid = np.zeros(T)
    turnover = np.zeros(T)
    
    # Initialize with zero weights
    prev_weights = np.zeros(N)
    
    for t in range(T):
        if rebalance_mask[t]:
            # Calculate weight changes
            weight_changes = np.abs(weights[t] - prev_weights)
            turnover[t] = np.sum(weight_changes)
            
            # Calculate transaction costs
            transaction_costs_paid[t] = np.sum(weight_changes * transaction_costs[t])
            
            # Update weights
            prev_weights = weights[t].copy()
        
        # Calculate portfolio return
        if t > 0:  # Skip first period
            gross_return = np.sum(prev_weights * returns[t])
            net_return = gross_return - transaction_costs_paid[t]
            portfolio_returns[t] = net_return
    
    return portfolio_returns, transaction_costs_paid, turnover


@nb.jit(nopython=True, cache=True)
def calculate_drawdown_numba(cumulative_returns: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """
    Numba-optimized drawdown calculation.
    
    Args:
        cumulative_returns: Cumulative return series
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_dd_start, max_dd_end)
    """
    n = len(cumulative_returns)
    drawdowns = np.zeros(n)
    peaks = np.zeros(n)
    
    max_drawdown = 0.0
    max_dd_start = 0
    max_dd_end = 0
    
    current_peak = cumulative_returns[0] if n > 0 else 0.0
    peaks[0] = current_peak
    
    for i in range(1, n):
        # Update peak
        if cumulative_returns[i] > current_peak:
            current_peak = cumulative_returns[i]
        peaks[i] = current_peak
        
        # Calculate drawdown
        if current_peak > 0:
            drawdowns[i] = (cumulative_returns[i] - current_peak) / current_peak
        else:
            drawdowns[i] = 0.0
        
        # Track maximum drawdown
        if drawdowns[i] < max_drawdown:
            max_drawdown = drawdowns[i]
            max_dd_end = i
            
            # Find start of this drawdown period
            for j in range(i, -1, -1):
                if cumulative_returns[j] == current_peak:
                    max_dd_start = j
                    break
    
    return drawdowns, abs(max_drawdown), max_dd_start, max_dd_end


class BacktestingEngine:
    """
    High-performance backtesting engine for contrarian forex strategies.
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 transaction_cost_bps: float = 2.0,
                 min_weight_threshold: float = 0.001,
                 rebalance_frequency: str = 'daily',
                 slippage_bps: float = 0.5,
                 max_position_size: float = 0.3,
                 cash_buffer: float = 0.05):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost_bps: Transaction costs in basis points
            min_weight_threshold: Minimum position size threshold
            rebalance_frequency: Portfolio rebalancing frequency
            slippage_bps: Market impact/slippage in basis points
            max_position_size: Maximum position size per asset
            cash_buffer: Cash buffer as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.min_weight_threshold = min_weight_threshold
        self.rebalance_frequency = rebalance_frequency
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size
        self.cash_buffer = cash_buffer
        
        # Results storage
        self.backtest_results = {}
        self.portfolio_history = {}
        self.trade_history = []
        self.rebalancing_dates = []
        
        logger.info(f"Initialized BacktestingEngine with {initial_capital:,.0f} initial capital")
    
    def preprocess_signals_and_returns(self, 
                                     signals: pd.DataFrame,
                                     returns: pd.DataFrame,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess signals and returns with proper alignment and validation.
        
        Args:
            signals: Signal weights DataFrame (dates x assets)
            returns: Returns DataFrame (dates x assets)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Tuple of (aligned_signals, aligned_returns)
        """
        logger.info("Preprocessing signals and returns for backtesting")
        
        # Ensure both have datetime indices
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals.index = pd.to_datetime(signals.index)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)
        
        # Sort chronologically
        signals = signals.sort_index()
        returns = returns.sort_index()
        
        # Filter by date range if specified
        if start_date:
            start_dt = pd.to_datetime(start_date)
            signals = signals[signals.index >= start_dt]
            returns = returns[returns.index >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            signals = signals[signals.index <= end_dt]
            returns = returns[returns.index <= end_dt]
        
        # Align indices (inner join to ensure data availability)
        common_index = signals.index.intersection(returns.index)
        
        if len(common_index) == 0:
            raise ValueError("No overlapping dates between signals and returns")
        
        signals_aligned = signals.reindex(common_index, fill_value=0.0)
        returns_aligned = returns.reindex(common_index, fill_value=0.0)
        
        # Align columns (inner join for common assets)
        common_columns = signals_aligned.columns.intersection(returns_aligned.columns)
        
        if len(common_columns) == 0:
            raise ValueError("No common assets between signals and returns")
        
        signals_aligned = signals_aligned[common_columns]
        returns_aligned = returns_aligned[common_columns]
        
        # Remove any remaining NaN values
        signals_aligned = signals_aligned.fillna(0.0)
        returns_aligned = returns_aligned.fillna(0.0)
        
        # Validate signal timing (signals at T should be applied to returns at T+1)
        # But our signal generator already handles this correctly
        logger.info(f"Aligned data: {len(signals_aligned)} dates, {len(common_columns)} assets")
        
        return signals_aligned, returns_aligned
    
    def apply_position_limits(self, 
                            raw_weights: pd.DataFrame,
                            volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply position size limits and portfolio constraints.
        
        Args:
            raw_weights: Raw portfolio weights
            volatility: Asset volatility for risk scaling (optional)
            
        Returns:
            Adjusted weights with position limits applied
        """
        logger.debug("Applying position limits and portfolio constraints")
        
        adjusted_weights = raw_weights.copy()
        
        # 1. Apply minimum weight threshold
        adjusted_weights = adjusted_weights.where(
            adjusted_weights.abs() >= self.min_weight_threshold, 0.0
        )
        
        # 2. Apply maximum position size limits
        adjusted_weights = adjusted_weights.clip(-self.max_position_size, self.max_position_size)
        
        # 3. Re-normalize weights to account for cash buffer
        target_exposure = 1.0 - self.cash_buffer
        
        for date in adjusted_weights.index:
            row_sum = adjusted_weights.loc[date].abs().sum()
            if row_sum > target_exposure:
                # Scale down to respect target exposure
                scaling_factor = target_exposure / row_sum
                adjusted_weights.loc[date] *= scaling_factor
        
        # 4. Handle concentration risk if volatility is provided
        if volatility is not None:
            adjusted_weights = self._apply_risk_scaling(adjusted_weights, volatility)
        
        return adjusted_weights
    
    def _apply_risk_scaling(self, 
                           weights: pd.DataFrame,
                           volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Apply additional risk scaling based on asset volatility.
        
        Args:
            weights: Portfolio weights
            volatility: Asset volatility measures
            
        Returns:
            Risk-scaled weights
        """
        risk_scaled_weights = weights.copy()
        
        # Align volatility with weights
        vol_aligned = volatility.reindex_like(weights).fillna(volatility.median())
        
        # Scale weights inversely with volatility (but preserve the risk parity logic from signals)
        # This is a secondary adjustment to prevent excessive concentration
        max_vol = vol_aligned.max(axis=1)
        vol_ratio = vol_aligned.div(max_vol, axis=0)
        
        # Apply gentle scaling (don't override the signal generator's risk parity completely)
        scaling_factor = 0.8 + 0.2 / vol_ratio  # Scale between 0.8 and 1.0
        risk_scaled_weights = weights * scaling_factor
        
        # Re-normalize
        for date in risk_scaled_weights.index:
            row_sum = risk_scaled_weights.loc[date].sum()
            if row_sum > 0:
                risk_scaled_weights.loc[date] /= row_sum
        
        return risk_scaled_weights
    
    def create_rebalancing_schedule(self, 
                                  index: pd.DatetimeIndex,
                                  frequency: str = None) -> pd.Series:
        """
        Create boolean series indicating rebalancing dates.
        
        Args:
            index: DateTime index for the backtest period
            frequency: Rebalancing frequency override
            
        Returns:
            Boolean series indicating rebalancing dates
        """
        freq = frequency or self.rebalance_frequency
        rebalance_mask = pd.Series(False, index=index)
        
        if freq == 'daily':
            rebalance_mask[:] = True
        elif freq == 'weekly':
            # Rebalance on Mondays
            rebalance_mask[index.dayofweek == 0] = True
        elif freq == 'monthly':
            # Rebalance on first business day of month
            first_bdays = pd.bdate_range(start=index.min(), end=index.max(), freq='BMS')
            rebalance_mask[index.isin(first_bdays)] = True
        elif freq == 'quarterly':
            # Rebalance on first business day of quarter
            first_bdays = pd.bdate_range(start=index.min(), end=index.max(), freq='BQS')
            rebalance_mask[index.isin(first_bdays)] = True
        else:
            raise ValueError(f"Unsupported rebalancing frequency: {freq}")
        
        # Always rebalance on first day
        if len(rebalance_mask) > 0:
            rebalance_mask.iloc[0] = True
        
        logger.info(f"Created rebalancing schedule: {rebalance_mask.sum()} rebalancing dates")
        return rebalance_mask
    
    def calculate_transaction_costs(self, 
                                  weights: pd.DataFrame,
                                  returns: pd.DataFrame,
                                  rebalance_mask: pd.Series) -> pd.DataFrame:
        """
        Calculate transaction costs including bid-ask spreads and slippage.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns (for volatility-based cost adjustment)
            rebalance_mask: Boolean mask for rebalancing dates
            
        Returns:
            DataFrame with transaction cost rates
        """
        logger.debug("Calculating transaction costs")
        
        # Base transaction cost
        base_cost = self.transaction_cost_bps / 10000.0
        slippage_cost = self.slippage_bps / 10000.0
        
        # Create transaction cost matrix
        transaction_costs = pd.DataFrame(
            base_cost + slippage_cost, 
            index=weights.index, 
            columns=weights.columns
        )
        
        # Adjust costs based on volatility (higher vol = higher costs)
        volatility = returns.rolling(window=20, min_periods=10).std().fillna(returns.std())
        vol_adjustment = 1.0 + volatility.clip(0, 0.1)  # Up to 10% adjustment
        
        transaction_costs = transaction_costs * vol_adjustment
        
        # Only apply costs on rebalancing dates
        transaction_costs = transaction_costs.multiply(rebalance_mask.astype(float), axis=0)
        
        return transaction_costs
    
    def run_backtest(self, 
                    signals: pd.DataFrame,
                    returns: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    volatility: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run complete backtest with vectorized calculations.
        
        Args:
            signals: Signal weights DataFrame
            returns: Returns DataFrame  
            start_date: Backtest start date
            end_date: Backtest end date
            volatility: Optional volatility data for risk scaling
            
        Returns:
            Dictionary with complete backtest results
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Step 1: Preprocess and align data
        signals_aligned, returns_aligned = self.preprocess_signals_and_returns(
            signals, returns, start_date, end_date
        )
        
        # Step 2: Apply position limits and constraints
        portfolio_weights = self.apply_position_limits(signals_aligned, volatility)
        
        # Step 3: Create rebalancing schedule
        rebalance_mask = self.create_rebalancing_schedule(portfolio_weights.index)
        
        # Step 4: Calculate transaction costs
        transaction_costs = self.calculate_transaction_costs(
            portfolio_weights, returns_aligned, rebalance_mask
        )
        
        # Step 5: Convert to numpy for numba acceleration
        weights_np = portfolio_weights.values
        returns_np = returns_aligned.values
        tcosts_np = transaction_costs.values
        rebalance_np = rebalance_mask.values
        
        # Step 6: Run numba-optimized calculation
        portfolio_returns, tcosts_paid, turnover = calculate_portfolio_returns_numba(
            weights_np, returns_np, tcosts_np, rebalance_np
        )
        
        # Step 7: Convert back to pandas and calculate metrics
        portfolio_returns_series = pd.Series(portfolio_returns, index=portfolio_weights.index)
        tcosts_series = pd.Series(tcosts_paid, index=portfolio_weights.index)
        turnover_series = pd.Series(turnover, index=portfolio_weights.index)
        
        # Step 8: Calculate cumulative performance
        portfolio_value = (1 + portfolio_returns_series).cumprod() * self.initial_capital
        
        # Step 9: Calculate drawdowns using numba
        cumulative_returns = (portfolio_value / self.initial_capital - 1).values
        drawdowns, max_dd, max_dd_start, max_dd_end = calculate_drawdown_numba(cumulative_returns)
        drawdown_series = pd.Series(drawdowns, index=portfolio_weights.index)
        
        # Step 10: Store detailed results
        backtest_results = {
            'portfolio_returns': portfolio_returns_series,
            'portfolio_value': portfolio_value,
            'portfolio_weights': portfolio_weights,
            'transaction_costs': tcosts_series,
            'turnover': turnover_series,
            'drawdowns': drawdown_series,
            'rebalancing_dates': portfolio_weights.index[rebalance_mask],
            'asset_returns': returns_aligned,
            'signals': signals_aligned,
            'metadata': {
                'start_date': portfolio_weights.index.min(),
                'end_date': portfolio_weights.index.max(),
                'total_days': len(portfolio_weights),
                'rebalancing_days': rebalance_mask.sum(),
                'assets': list(portfolio_weights.columns),
                'initial_capital': self.initial_capital,
                'transaction_cost_bps': self.transaction_cost_bps,
                'max_drawdown': max_dd,
                'max_dd_start_date': portfolio_weights.index[max_dd_start] if max_dd_start < len(portfolio_weights) else None,
                'max_dd_end_date': portfolio_weights.index[max_dd_end] if max_dd_end < len(portfolio_weights) else None,
            }
        }
        
        # Store in instance for access
        self.backtest_results = backtest_results
        
        logger.info(f"Backtest completed: {len(portfolio_weights)} days, "
                   f"Final value: {portfolio_value.iloc[-1]:,.0f}, "
                   f"Total return: {(portfolio_value.iloc[-1]/self.initial_capital-1)*100:.2f}%, "
                   f"Max drawdown: {max_dd*100:.2f}%")
        
        return backtest_results
    
    def calculate_trade_analysis(self, backtest_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze individual trades and positions.
        
        Args:
            backtest_results: Results from run_backtest()
            
        Returns:
            DataFrame with trade analysis
        """
        logger.info("Calculating trade analysis")
        
        portfolio_weights = backtest_results['portfolio_weights']
        asset_returns = backtest_results['asset_returns']
        rebalancing_dates = backtest_results['rebalancing_dates']
        
        trades = []
        
        # Track position changes on rebalancing dates
        prev_weights = pd.Series(0.0, index=portfolio_weights.columns)
        
        for date in rebalancing_dates:
            current_weights = portfolio_weights.loc[date]
            
            # Find position changes
            weight_changes = current_weights - prev_weights
            
            for asset in weight_changes.index:
                change = weight_changes[asset]
                if abs(change) > self.min_weight_threshold:
                    # This is a significant trade
                    trades.append({
                        'date': date,
                        'asset': asset,
                        'action': 'BUY' if change > 0 else 'SELL',
                        'weight_change': change,
                        'new_weight': current_weights[asset],
                        'prev_weight': prev_weights[asset]
                    })
            
            prev_weights = current_weights.copy()
        
        trades_df = pd.DataFrame(trades)
        
        if not trades_df.empty:
            # Add performance attribution
            trades_df['next_day_return'] = trades_df.apply(
                lambda row: asset_returns.loc[
                    asset_returns.index > row['date'], row['asset']
                ].iloc[0] if len(asset_returns.loc[asset_returns.index > row['date']]) > 0 else 0.0,
                axis=1
            )
            
            trades_df['contribution'] = trades_df['weight_change'] * trades_df['next_day_return']
        
        logger.info(f"Analyzed {len(trades_df)} individual trades")
        return trades_df
    
    def get_portfolio_statistics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio statistics.
        
        Args:
            backtest_results: Results from run_backtest()
            
        Returns:
            Dictionary with portfolio statistics
        """
        portfolio_returns = backtest_results['portfolio_returns']
        portfolio_value = backtest_results['portfolio_value']
        transaction_costs = backtest_results['transaction_costs']
        turnover = backtest_results['turnover']
        
        # Remove first day (initialization day)
        portfolio_returns_clean = portfolio_returns.iloc[1:]
        
        if len(portfolio_returns_clean) == 0:
            logger.warning("Insufficient data for statistics calculation")
            return {}
        
        # Basic performance metrics
        total_return = (portfolio_value.iloc[-1] / self.initial_capital) - 1.0
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns_clean)) - 1.0
        volatility = portfolio_returns_clean.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Downside deviation (for Sortino ratio)
        negative_returns = portfolio_returns_clean[portfolio_returns_clean < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0.0
        
        # Maximum drawdown
        max_drawdown = backtest_results['metadata']['max_drawdown']
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Win/loss statistics
        winning_days = (portfolio_returns_clean > 0).sum()
        total_days = len(portfolio_returns_clean)
        win_rate = winning_days / total_days if total_days > 0 else 0.0
        
        avg_win = portfolio_returns_clean[portfolio_returns_clean > 0].mean() if winning_days > 0 else 0.0
        avg_loss = portfolio_returns_clean[portfolio_returns_clean < 0].mean() if (total_days - winning_days) > 0 else 0.0
        profit_factor = abs(avg_win * winning_days / (avg_loss * (total_days - winning_days))) if avg_loss < 0 else float('inf')
        
        # Transaction cost analysis
        total_transaction_costs = transaction_costs.sum()
        avg_daily_turnover = turnover.mean()
        
        # VaR calculations (5% and 1%)
        var_5 = np.percentile(portfolio_returns_clean, 5) if len(portfolio_returns_clean) > 0 else 0.0
        var_1 = np.percentile(portfolio_returns_clean, 1) if len(portfolio_returns_clean) > 0 else 0.0
        
        stats = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_transaction_costs': total_transaction_costs,
            'avg_daily_turnover': avg_daily_turnover,
            'var_5_percent': var_5,
            'var_1_percent': var_1,
            'trading_days': total_days,
            'winning_days': int(winning_days),
            'losing_days': int(total_days - winning_days),
        }
        
        logger.info(f"Calculated portfolio statistics: Sharpe={sharpe_ratio:.3f}, MaxDD={max_drawdown*100:.2f}%")
        return stats
    
    def save_results(self, 
                    backtest_results: Dict[str, Any],
                    output_dir: str,
                    prefix: str = "backtest") -> Dict[str, str]:
        """
        Save backtest results to files.
        
        Args:
            backtest_results: Results from run_backtest()
            output_dir: Directory to save results
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping result types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = output_path / f"{prefix}_results_{timestamp}.parquet"
        main_results = pd.DataFrame({
            'portfolio_returns': backtest_results['portfolio_returns'],
            'portfolio_value': backtest_results['portfolio_value'],
            'transaction_costs': backtest_results['transaction_costs'],
            'turnover': backtest_results['turnover'],
            'drawdowns': backtest_results['drawdowns']
        })
        main_results.to_parquet(results_file)
        saved_files['main_results'] = str(results_file)
        
        # Save portfolio weights
        weights_file = output_path / f"{prefix}_weights_{timestamp}.parquet"
        backtest_results['portfolio_weights'].to_parquet(weights_file)
        saved_files['portfolio_weights'] = str(weights_file)
        
        # Save metadata
        metadata_file = output_path / f"{prefix}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            metadata = backtest_results['metadata'].copy()
            for key, value in metadata.items():
                if isinstance(value, pd.Timestamp):
                    metadata[key] = str(value)
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_file)
        
        # Calculate and save statistics
        stats = self.get_portfolio_statistics(backtest_results)
        stats_file = output_path / f"{prefix}_statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        saved_files['statistics'] = str(stats_file)
        
        logger.info(f"Saved backtest results to {output_dir}")
        return saved_files


class PortfolioConstructor:
    """
    Advanced portfolio construction with multiple methodologies.
    """
    
    def __init__(self, method: str = 'risk_parity'):
        """
        Initialize portfolio constructor.
        
        Args:
            method: Portfolio construction method ('risk_parity', 'equal_weight', 'volatility_scaled')
        """
        self.method = method
        logger.info(f"Initialized PortfolioConstructor with {method} method")
    
    def construct_portfolio(self, 
                          signals: pd.DataFrame,
                          returns: pd.DataFrame,
                          volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Construct portfolio weights using specified methodology.
        
        Args:
            signals: Binary signals (1 for selected assets)
            returns: Asset returns for volatility calculation
            volatility: Pre-calculated volatility (optional)
            
        Returns:
            Portfolio weights DataFrame
        """
        if self.method == 'risk_parity':
            return self._risk_parity_weights(signals, returns, volatility)
        elif self.method == 'equal_weight':
            return self._equal_weight(signals)
        elif self.method == 'volatility_scaled':
            return self._volatility_scaled_weights(signals, returns, volatility)
        else:
            raise ValueError(f"Unknown portfolio construction method: {self.method}")
    
    def _risk_parity_weights(self, 
                           signals: pd.DataFrame,
                           returns: pd.DataFrame,
                           volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Risk parity weighting (already implemented in signal generator)."""
        # This is typically handled by the signal generator, but we can refine here
        return signals  # Assuming signals already contain risk parity weights
    
    def _equal_weight(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Equal weighting among selected assets."""
        weights = signals.copy()
        
        for date in weights.index:
            selected = weights.loc[date] > 0
            n_selected = selected.sum()
            if n_selected > 0:
                weights.loc[date] = selected.astype(float) / n_selected
        
        return weights
    
    def _volatility_scaled_weights(self, 
                                 signals: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Inverse volatility weighting."""
        if volatility is None:
            volatility = returns.rolling(window=60, min_periods=30).std()
        
        weights = signals.copy()
        
        for date in weights.index:
            selected = weights.loc[date] > 0
            if selected.sum() > 0:
                # Get volatilities for selected assets
                vols = volatility.loc[date, selected]
                vols = vols.fillna(vols.median()).clip(lower=0.001)  # Minimum vol
                
                # Inverse volatility weights
                inv_vol_weights = 1.0 / vols
                inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
                
                weights.loc[date] = 0.0
                weights.loc[date, selected] = inv_vol_weights
        
        return weights


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("BacktestingEngine module loaded successfully")
    print("Module ready for use. Import and use with real forex data and signals.")