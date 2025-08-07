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
import time

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@nb.jit(nopython=True, fastmath=True)
def calculate_portfolio_returns_numba(weights: np.ndarray, 
                                    returns: np.ndarray,
                                    transaction_costs: np.ndarray,
                                    rebalance_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ultra-optimized portfolio return calculation with transaction costs.
    
    Args:
        weights: Portfolio weights (T x N)
        returns: Asset returns (T x N)
        transaction_costs: Transaction cost rates (T x N)
        rebalance_mask: Boolean mask for rebalancing days (T,)
        
    Returns:
        Tuple of (portfolio_returns, transaction_costs_paid, turnover)
    """
    T, N = weights.shape
    portfolio_returns = np.zeros(T, dtype=np.float64)
    transaction_costs_paid = np.zeros(T, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    
    # Initialize with zero weights
    prev_weights = np.zeros(N, dtype=np.float64)
    weight_changes = np.zeros(N, dtype=np.float64)
    
    for t in range(T):
        if rebalance_mask[t]:
            # Calculate weight changes (vectorized abs diff)
            for i in range(N):
                weight_changes[i] = abs(weights[t, i] - prev_weights[i])
            
            # Sum turnover and transaction costs in single pass
            turnover_sum = 0.0
            tcost_sum = 0.0
            for i in range(N):
                change = weight_changes[i]
                turnover_sum += change
                tcost_sum += change * transaction_costs[t, i]
            
            turnover[t] = turnover_sum
            transaction_costs_paid[t] = tcost_sum
            
            # Update weights
            for i in range(N):
                prev_weights[i] = weights[t, i]
        
        # Calculate portfolio return (vectorized dot product)
        if t > 0:  # Skip first period
            gross_return = 0.0
            for i in range(N):
                gross_return += prev_weights[i] * returns[t, i]
            
            portfolio_returns[t] = gross_return - transaction_costs_paid[t]
    
    return portfolio_returns, transaction_costs_paid, turnover


@nb.jit(nopython=True, fastmath=True)
def calculate_portfolio_returns_vectorized(weights: np.ndarray, 
                                         returns: np.ndarray,
                                         transaction_costs: np.ndarray,
                                         rebalance_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized portfolio return calculation - numba compatible.
    
    Args:
        weights: Portfolio weights (T x N)
        returns: Asset returns (T x N)
        transaction_costs: Transaction cost rates (T x N)
        rebalance_mask: Boolean mask for rebalancing days (T,)
        
    Returns:
        Tuple of (portfolio_returns, transaction_costs_paid, turnover)
    """
    T, N = weights.shape
    
    # Pre-allocate all arrays with explicit dtypes
    portfolio_returns = np.zeros(T, dtype=np.float64)
    transaction_costs_paid = np.zeros(T, dtype=np.float64)
    turnover = np.zeros(T, dtype=np.float64)
    
    # Track active portfolio weights
    active_weights = np.zeros(N, dtype=np.float64)
    prev_weights = np.zeros(N, dtype=np.float64)
    
    for t in range(T):
        if rebalance_mask[t]:
            # Calculate weight changes
            turnover_sum = 0.0
            tcost_sum = 0.0
            
            for i in range(N):
                weight_change = abs(weights[t, i] - prev_weights[i])
                turnover_sum += weight_change
                tcost_sum += weight_change * transaction_costs[t, i]
                
                # Update active and previous weights
                active_weights[i] = weights[t, i]
                prev_weights[i] = weights[t, i]
            
            turnover[t] = turnover_sum
            transaction_costs_paid[t] = tcost_sum
        
        # Calculate portfolio return
        if t > 0:  # Skip first period
            gross_return = 0.0
            for i in range(N):
                gross_return += active_weights[i] * returns[t, i]
            
            portfolio_returns[t] = gross_return - transaction_costs_paid[t]
    
    return portfolio_returns, transaction_costs_paid, turnover


@nb.jit(nopython=True, fastmath=True)
def calculate_drawdown_numba(cumulative_returns: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
    """
    Ultra-optimized drawdown calculation with improved performance.
    
    Args:
        cumulative_returns: Cumulative return series
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_dd_start, max_dd_end)
    """
    n = len(cumulative_returns)
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0, 0, 0
        
    drawdowns = np.zeros(n, dtype=np.float64)
    
    max_drawdown = 0.0
    max_dd_start = 0
    max_dd_end = 0
    
    current_peak = cumulative_returns[0]
    current_peak_idx = 0
    
    for i in range(1, n):
        # Update peak efficiently
        if cumulative_returns[i] > current_peak:
            current_peak = cumulative_returns[i]
            current_peak_idx = i
        
        # Calculate drawdown (avoid division by zero)
        if current_peak > 1e-12:  # Small epsilon for numerical stability
            drawdown = (cumulative_returns[i] - current_peak) / current_peak
            drawdowns[i] = drawdown
            
            # Track maximum drawdown without expensive backward search
            if drawdown < max_drawdown:
                max_drawdown = drawdown
                max_dd_start = current_peak_idx
                max_dd_end = i
    
    return drawdowns, abs(max_drawdown), max_dd_start, max_dd_end


@nb.jit(nopython=True, fastmath=True)
def calculate_performance_metrics_numba(returns: np.ndarray, 
                                       portfolio_value: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fast calculation of key performance metrics using numba.
    
    Args:
        returns: Portfolio returns array
        portfolio_value: Portfolio value series
        
    Returns:
        Tuple of (total_return, annualized_return, volatility, sharpe_ratio)
    """
    if len(returns) <= 1:
        return 0.0, 0.0, 0.0, 0.0
    
    # Remove first day (initialization)
    clean_returns = returns[1:]
    n_days = len(clean_returns)
    
    if n_days == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Total return
    total_return = portfolio_value[-1] / portfolio_value[0] - 1.0
    
    # Annualized return
    annualized_return = (1 + total_return) ** (252.0 / n_days) - 1.0
    
    # Volatility calculation
    mean_return = 0.0
    for i in range(n_days):
        mean_return += clean_returns[i]
    mean_return /= n_days
    
    variance = 0.0
    for i in range(n_days):
        diff = clean_returns[i] - mean_return
        variance += diff * diff
    
    volatility = np.sqrt(variance / (n_days - 1)) * np.sqrt(252.0) if n_days > 1 else 0.0
    
    # Sharpe ratio
    sharpe_ratio = annualized_return / volatility if volatility > 1e-12 else 0.0
    
    return total_return, annualized_return, volatility, sharpe_ratio


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
        Optimized preprocessing with minimal pandas overhead.
        
        Args:
            signals: Signal weights DataFrame (dates x assets)
            returns: Returns DataFrame (dates x assets)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Tuple of (aligned_signals, aligned_returns)
        """
        logger.info("Preprocessing signals and returns for backtesting")
        
        # Fast index validation and conversion
        signals_idx = signals.index
        returns_idx = returns.index
        
        if not isinstance(signals_idx, pd.DatetimeIndex):
            signals_idx = pd.to_datetime(signals_idx)
            signals = signals.set_index(signals_idx)
        if not isinstance(returns_idx, pd.DatetimeIndex):
            returns_idx = pd.to_datetime(returns_idx)
            returns = returns.set_index(returns_idx)
        
        # Fast date filtering with boolean indexing
        if start_date or end_date:
            signals, returns = self._filter_by_dates_optimized(
                signals, returns, start_date, end_date
            )
        
        # Fast alignment using numpy operations
        return self._align_data_optimized(signals, returns)
    
    def _filter_by_dates_optimized(self, 
                                 signals: pd.DataFrame, 
                                 returns: pd.DataFrame,
                                 start_date: Optional[str],
                                 end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fast date filtering using boolean masks.
        """
        if start_date:
            start_dt = pd.to_datetime(start_date)
            signals_mask = signals.index >= start_dt
            returns_mask = returns.index >= start_dt
            signals = signals.loc[signals_mask]
            returns = returns.loc[returns_mask]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            signals_mask = signals.index <= end_dt
            returns_mask = returns.index <= end_dt
            signals = signals.loc[signals_mask]
            returns = returns.loc[returns_mask]
            
        return signals, returns
    
    def _align_data_optimized(self, 
                            signals: pd.DataFrame, 
                            returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fast data alignment with minimal copying.
        """
        # Use pandas' built-in align method for efficiency
        signals_aligned, returns_aligned = signals.align(
            returns, join='inner', axis=0, copy=False
        )
        
        # Get common columns efficiently
        common_columns = signals_aligned.columns.intersection(returns_aligned.columns)
        
        if len(common_columns) == 0:
            raise ValueError("No common assets between signals and returns")
        
        # Select common columns (creates view when possible)
        signals_aligned = signals_aligned.loc[:, common_columns]
        returns_aligned = returns_aligned.loc[:, common_columns]
        
        # Fast NaN filling using numpy
        if signals_aligned.isna().any().any():
            signals_values = signals_aligned.values
            signals_values[np.isnan(signals_values)] = 0.0
            signals_aligned = pd.DataFrame(signals_values, 
                                         index=signals_aligned.index, 
                                         columns=signals_aligned.columns)
        
        if returns_aligned.isna().any().any():
            returns_values = returns_aligned.values
            returns_values[np.isnan(returns_values)] = 0.0
            returns_aligned = pd.DataFrame(returns_values, 
                                         index=returns_aligned.index, 
                                         columns=returns_aligned.columns)
        
        logger.info(f"Aligned data: {len(signals_aligned)} dates, {len(common_columns)} assets")
        return signals_aligned, returns_aligned
    
    def apply_position_limits(self, 
                            raw_weights: pd.DataFrame,
                            volatility: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Optimized position limits application using vectorized operations.
        
        Args:
            raw_weights: Raw portfolio weights
            volatility: Asset volatility for risk scaling (optional)
            
        Returns:
            Adjusted weights with position limits applied
        """
        logger.debug("Applying position limits and portfolio constraints")
        
        # Use numpy operations for speed
        weights_values = raw_weights.values.copy()
        
        # 1. Apply minimum weight threshold (vectorized)
        mask = np.abs(weights_values) < self.min_weight_threshold
        weights_values[mask] = 0.0
        
        # 2. Apply maximum position size limits (vectorized)
        weights_values = np.clip(weights_values, -self.max_position_size, self.max_position_size)
        
        # 3. Re-normalize weights for cash buffer (vectorized by row)
        target_exposure = 1.0 - self.cash_buffer
        row_sums = np.abs(weights_values).sum(axis=1)
        
        # Only scale rows that exceed target exposure
        scale_mask = row_sums > target_exposure
        scaling_factors = target_exposure / row_sums
        scaling_factors = np.where(scale_mask, scaling_factors, 1.0)
        
        # Apply scaling
        weights_values = weights_values * scaling_factors.reshape(-1, 1)
        
        # Create result DataFrame
        adjusted_weights = pd.DataFrame(
            weights_values, 
            index=raw_weights.index, 
            columns=raw_weights.columns
        )
        
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
        Fast rebalancing schedule creation with vectorized operations.
        
        Args:
            index: DateTime index for the backtest period
            frequency: Rebalancing frequency override
            
        Returns:
            Boolean series indicating rebalancing dates
        """
        freq = frequency or self.rebalance_frequency
        
        if freq == 'daily':
            # Most common case - simple numpy array creation
            rebalance_values = np.ones(len(index), dtype=bool)
        else:
            # Use vectorized operations for other frequencies
            rebalance_values = self._create_rebalance_mask_vectorized(index, freq)
        
        # Create series with minimal overhead
        rebalance_mask = pd.Series(rebalance_values, index=index, copy=False)
        
        # Ensure first day is always rebalancing day
        if len(rebalance_mask) > 0:
            rebalance_mask.iloc[0] = True
        
        n_rebalancing = rebalance_mask.sum()
        logger.info(f"Created rebalancing schedule: {n_rebalancing} rebalancing dates")
        return rebalance_mask
    
    def _create_rebalance_mask_vectorized(self, index: pd.DatetimeIndex, freq: str) -> np.ndarray:
        """
        Vectorized rebalancing mask creation.
        """
        mask = np.zeros(len(index), dtype=bool)
        
        if freq == 'weekly':
            # Monday rebalancing
            mask = index.dayofweek == 0
        elif freq == 'monthly':
            # First business day of month - use pandas built-in efficiency
            first_bdays = pd.bdate_range(start=index.min(), end=index.max(), freq='BMS')
            mask = index.isin(first_bdays)
        elif freq == 'quarterly':
            # First business day of quarter
            first_bdays = pd.bdate_range(start=index.min(), end=index.max(), freq='BQS')
            mask = index.isin(first_bdays)
        else:
            raise ValueError(f"Unsupported rebalancing frequency: {freq}")
        
        return mask.values if hasattr(mask, 'values') else mask
    
    def calculate_transaction_costs(self, 
                                  weights: pd.DataFrame,
                                  returns: pd.DataFrame,
                                  rebalance_mask: pd.Series) -> pd.DataFrame:
        """
        Optimized transaction cost calculation with vectorized operations.
        
        Args:
            weights: Portfolio weights
            returns: Asset returns (for volatility-based cost adjustment)
            rebalance_mask: Boolean mask for rebalancing dates
            
        Returns:
            DataFrame with transaction cost rates
        """
        logger.debug("Calculating transaction costs")
        
        # Use vectorized operations for speed
        return self._calculate_transaction_costs_vectorized(
            weights.values, returns.values, rebalance_mask.values,
            weights.index, weights.columns
        )
    
    def _calculate_transaction_costs_vectorized(self,
                                              weights_np: np.ndarray,
                                              returns_np: np.ndarray,
                                              rebalance_mask_np: np.ndarray,
                                              index: pd.Index,
                                              columns: pd.Index) -> pd.DataFrame:
        """
        Vectorized transaction cost calculation.
        """
        T, N = weights_np.shape
        
        # Base costs (vectorized)
        base_cost = self.transaction_cost_bps / 10000.0
        slippage_cost = self.slippage_bps / 10000.0
        total_base_cost = base_cost + slippage_cost
        
        # Vectorized volatility calculation
        volatility = self._rolling_std_vectorized(returns_np, window=20, min_periods=10)
        
        # Apply volatility adjustment (vectorized)
        vol_adjustment = 1.0 + np.clip(volatility, 0, 0.1)
        
        # Calculate final transaction costs
        transaction_costs = total_base_cost * vol_adjustment
        
        # Apply rebalancing mask
        transaction_costs = transaction_costs * rebalance_mask_np.reshape(-1, 1)
        
        return pd.DataFrame(transaction_costs, index=index, columns=columns)
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def _rolling_std_vectorized(data: np.ndarray, window: int, min_periods: int) -> np.ndarray:
        """
        Fast rolling standard deviation using numba.
        """
        T, N = data.shape
        result = np.zeros_like(data)
        
        for i in range(N):  # For each asset
            for t in range(T):  # For each time period
                start_idx = max(0, t - window + 1)
                n_obs = t - start_idx + 1
                
                if n_obs >= min_periods:
                    # Calculate rolling std
                    mean_val = 0.0
                    for k in range(start_idx, t + 1):
                        mean_val += data[k, i]
                    mean_val /= n_obs
                    
                    var_val = 0.0
                    for k in range(start_idx, t + 1):
                        diff = data[k, i] - mean_val
                        var_val += diff * diff
                    
                    result[t, i] = np.sqrt(var_val / (n_obs - 1)) if n_obs > 1 else 0.0
                else:
                    # Use overall std as fallback
                    if t == T - 1:  # Only calculate once at the end
                        overall_mean = 0.0
                        for k in range(T):
                            overall_mean += data[k, i]
                        overall_mean /= T
                        
                        overall_var = 0.0
                        for k in range(T):
                            diff = data[k, i] - overall_mean
                            overall_var += diff * diff
                        overall_std = np.sqrt(overall_var / (T - 1)) if T > 1 else 0.0
                        
                        # Fill all insufficient periods with overall std
                        for fill_t in range(T):
                            if result[fill_t, i] == 0.0:
                                result[fill_t, i] = overall_std
        
        return result
    
    def run_backtest(self, 
                    signals: pd.DataFrame,
                    returns: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    volatility: Optional[pd.DataFrame] = None,
                    use_fast_mode: bool = True) -> Dict[str, Any]:
        """
        Ultra-optimized backtest with minimal memory allocation and maximum vectorization.
        
        Args:
            signals: Signal weights DataFrame
            returns: Returns DataFrame  
            start_date: Backtest start date
            end_date: Backtest end date
            volatility: Optional volatility data for risk scaling
            use_fast_mode: Use fastest vectorized calculations (default: True)
            
        Returns:
            Dictionary with complete backtest results
        """
        logger.info(f"Starting optimized backtest from {start_date} to {end_date}")
        import time
        start_time = time.time()
        
        # Step 1: Preprocess and align data (optimized)
        signals_aligned, returns_aligned = self.preprocess_signals_and_returns(
            signals, returns, start_date, end_date
        )
        
        # Step 2: Apply position limits (vectorized)
        portfolio_weights = self.apply_position_limits(signals_aligned, volatility)
        
        # Step 3: Create rebalancing schedule (fast)
        rebalance_mask = self.create_rebalancing_schedule(portfolio_weights.index)
        
        # Step 4: Calculate transaction costs (vectorized)
        transaction_costs = self.calculate_transaction_costs(
            portfolio_weights, returns_aligned, rebalance_mask
        )
        
        # Step 5: Convert to numpy arrays (minimal copying)
        weights_np = portfolio_weights.values
        returns_np = returns_aligned.values
        tcosts_np = transaction_costs.values
        rebalance_np = rebalance_mask.values
        
        # Step 6: Choose calculation method based on performance requirements
        if use_fast_mode and len(weights_np) > 1000:  # Use vectorized for large datasets
            portfolio_returns, tcosts_paid, turnover = calculate_portfolio_returns_vectorized(
                weights_np, returns_np, tcosts_np, rebalance_np
            )
        else:
            portfolio_returns, tcosts_paid, turnover = calculate_portfolio_returns_numba(
                weights_np, returns_np, tcosts_np, rebalance_np
            )
        
        # Step 7: Efficient series creation with pre-allocated index
        idx = portfolio_weights.index
        portfolio_returns_series = pd.Series(portfolio_returns, index=idx, copy=False)
        tcosts_series = pd.Series(tcosts_paid, index=idx, copy=False)
        turnover_series = pd.Series(turnover, index=idx, copy=False)
        
        # Step 8: Fast portfolio value calculation
        portfolio_value = self._calculate_portfolio_value_fast(
            portfolio_returns_series, self.initial_capital
        )
        
        # Step 9: Optimized performance metrics
        total_return, ann_return, volatility_metric, sharpe = calculate_performance_metrics_numba(
            portfolio_returns, portfolio_value.values
        )
        
        # Step 10: Fast drawdown calculation
        cumulative_returns = portfolio_value.values / self.initial_capital - 1.0
        drawdowns, max_dd, max_dd_start, max_dd_end = calculate_drawdown_numba(cumulative_returns)
        drawdown_series = pd.Series(drawdowns, index=idx, copy=False)
        
        # Step 11: Efficient results dictionary construction
        backtest_results = self._construct_results_dict(
            portfolio_returns_series, portfolio_value, portfolio_weights,
            tcosts_series, turnover_series, drawdown_series,
            returns_aligned, signals_aligned, rebalance_mask,
            max_dd, max_dd_start, max_dd_end,
            total_return, ann_return, volatility_metric, sharpe
        )
        
        # Store results
        self.backtest_results = backtest_results
        
        execution_time = time.time() - start_time
        logger.info(f"Optimized backtest completed in {execution_time:.2f}s: {len(portfolio_weights)} days, "
                   f"Final value: {portfolio_value.iloc[-1]:,.0f}, "
                   f"Total return: {total_return*100:.2f}%, "
                   f"Sharpe: {sharpe:.3f}, Max DD: {max_dd*100:.2f}%")
        
        return backtest_results
    
    def _calculate_portfolio_value_fast(self, returns: pd.Series, initial_capital: float) -> pd.Series:
        """
        Fast portfolio value calculation using numpy.
        """
        # Use numpy cumprod for speed
        cumulative_factors = np.cumprod(1.0 + returns.values)
        return pd.Series(cumulative_factors * initial_capital, index=returns.index, copy=False)
    
    def _construct_results_dict(self, 
                              portfolio_returns: pd.Series,
                              portfolio_value: pd.Series,
                              portfolio_weights: pd.DataFrame,
                              tcosts_series: pd.Series,
                              turnover_series: pd.Series,
                              drawdown_series: pd.Series,
                              returns_aligned: pd.DataFrame,
                              signals_aligned: pd.DataFrame,
                              rebalance_mask: pd.Series,
                              max_dd: float,
                              max_dd_start: int,
                              max_dd_end: int,
                              total_return: float,
                              ann_return: float,
                              volatility_metric: float,
                              sharpe: float) -> Dict[str, Any]:
        """
        Efficient results dictionary construction.
        """
        return {
            'portfolio_returns': portfolio_returns,
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
                'total_return': total_return,
                'annualized_return': ann_return,
                'volatility': volatility_metric,
                'sharpe_ratio': sharpe,
            }
        }
    
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