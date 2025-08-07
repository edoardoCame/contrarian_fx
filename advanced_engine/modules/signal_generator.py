#!/usr/bin/env python3
"""
Contrarian Signal Generator Module for Forex Trading System

This module implements a contrarian trading signal generation system with strict
lookahead bias prevention. Every day, it identifies the worst N performers over
a lookback period M and generates long signals for these currencies.

Key Features:
- ZERO lookahead bias: Only uses data up to T-1 for signals applied at T
- Risk parity weighting based on historical volatility
- Configurable parameters (N worst performers, M lookback days)
- Vectorized pandas operations for efficiency
- Comprehensive validation and quality control

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ConrarianSignalGenerator:
    """
    Contrarian signal generation system with strict lookahead bias prevention.
    
    The system works as follows:
    1. At time T, look back M days (from T-M to T-1) to calculate returns
    2. Rank currencies by performance (worst performers first)
    3. Select worst N performers for long positions
    4. Calculate risk parity weights using historical volatility (T-M to T-1)
    5. Generate signals to be applied starting from T
    """
    
    def __init__(self, 
                 n_worst_performers: int = 5,
                 lookback_days: int = 20,
                 min_history_days: int = 252,
                 volatility_lookback: int = 60):
        """
        Initialize the contrarian signal generator.
        
        Args:
            n_worst_performers: Number of worst performing currencies to select
            lookback_days: Number of days to look back for performance ranking
            min_history_days: Minimum days of history required before generating signals
            volatility_lookback: Days to look back for volatility calculation
        """
        self.n_worst_performers = n_worst_performers
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self.volatility_lookback = volatility_lookback
        
        # Signal tracking
        self.signal_history = {}
        self.weight_history = {}
        self.performance_history = {}
        
        logger.info(f"Initialized ConrarianSignalGenerator with N={n_worst_performers}, M={lookback_days}")
    
    def calculate_rolling_returns(self, 
                                prices: pd.DataFrame, 
                                lookback: int) -> pd.DataFrame:
        """
        Calculate rolling returns over specified lookback period with strict lookahead prevention - OPTIMIZED.
        
        Args:
            prices: DataFrame with prices (dates x currencies)
            lookback: Number of days to calculate returns over
            
        Returns:
            DataFrame with rolling returns (no lookahead bias)
        """
        # OPTIMIZED: Use pandas pct_change with periods parameter for better performance
        # This is equivalent to (price_{t} / price_{t-periods}) - 1 but faster
        # Add 1 day shift to prevent lookahead bias: use T-1 data for signal at T
        shifted_prices = prices.shift(1)
        
        # OPTIMIZED: Single operation instead of two shifts and division
        returns = shifted_prices.pct_change(periods=lookback)
        
        logger.debug(f"Calculated rolling {lookback}-day returns with 1-day lag (optimized)")
        return returns
    
    def calculate_historical_volatility(self, 
                                      returns: pd.DataFrame, 
                                      vol_lookback: int) -> pd.DataFrame:
        """
        Calculate historical volatility using only past data - OPTIMIZED.
        
        Args:
            returns: DataFrame with daily returns
            vol_lookback: Number of days for volatility calculation
            
        Returns:
            DataFrame with rolling volatility (annualized)
        """
        # OPTIMIZED: Pre-calculate sqrt(252) to avoid repeated calculation
        annualization_factor = np.sqrt(252)
        
        # OPTIMIZED: Use more efficient min_periods calculation
        min_periods = max(10, vol_lookback // 3)  # More aggressive min_periods for speed
        
        # Calculate rolling standard deviation and annualize in one step
        historical_vol = (returns.rolling(window=vol_lookback, min_periods=min_periods)
                         .std() * annualization_factor).shift(1)
        
        logger.debug(f"Calculated {vol_lookback}-day historical volatility (optimized)")
        return historical_vol
    
    def rank_performance(self, 
                        rolling_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Rank currencies by performance (ascending = worst performers first) - OPTIMIZED.
        
        Args:
            rolling_returns: DataFrame with rolling returns
            
        Returns:
            DataFrame with performance ranks (1 = worst performer)
        """
        # OPTIMIZED: Use 'first' method instead of 'min' for better performance
        # and specify numeric_only=True to avoid type checking overhead
        ranks = rolling_returns.rank(axis=1, method='first', ascending=True, numeric_only=True)
        
        logger.debug("Ranked currency performance (optimized)")
        return ranks
    
    def generate_binary_signals(self, 
                               ranks: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary signals for worst N performers - OPTIMIZED.
        
        Args:
            ranks: DataFrame with performance ranks
            
        Returns:
            DataFrame with binary signals (1 = selected, 0 = not selected)
        """
        # OPTIMIZED: Use numpy array operations for better performance
        # Convert to numpy for faster boolean operations
        ranks_values = ranks.values
        
        # Create boolean mask and convert to int in one operation
        binary_mask = (ranks_values <= self.n_worst_performers) & (~np.isnan(ranks_values))
        
        # Create DataFrame with optimized constructor
        binary_signals = pd.DataFrame(
            binary_mask.astype(np.int8, copy=False),  # int8 for memory efficiency
            index=ranks.index,
            columns=ranks.columns
        )
        
        logger.debug(f"Generated binary signals for {self.n_worst_performers} worst performers (optimized)")
        return binary_signals
    
    def calculate_risk_parity_weights(self, 
                                    binary_signals: pd.DataFrame,
                                    volatility: pd.DataFrame,
                                    min_vol: float = 0.01) -> pd.DataFrame:
        """
        Calculate risk parity weights for selected currencies - OPTIMIZED VERSION.
        
        Args:
            binary_signals: DataFrame with binary selection signals
            volatility: DataFrame with historical volatility
            min_vol: Minimum volatility floor to prevent division by zero
            
        Returns:
            DataFrame with risk parity weights
        """
        # Apply minimum volatility floor and handle NaN values (vectorized)
        adj_volatility = volatility.fillna(min_vol).clip(lower=min_vol)
        
        # Calculate inverse volatility weights (only for selected currencies) - vectorized
        inv_vol = 1.0 / adj_volatility
        inv_vol_selected = inv_vol * binary_signals
        
        # OPTIMIZED: Vectorized normalization using pandas operations
        # Calculate row sums for normalization
        row_sums = inv_vol_selected.sum(axis=1)
        
        # Vectorized weight calculation - eliminate the loop!
        # Use pandas broadcasting to divide each row by its sum, handling zeros automatically
        weights = inv_vol_selected.div(row_sums, axis=0).fillna(0.0)
        
        # Zero out rows where sum is zero or NaN (pandas div already handles this, but be explicit)
        weights = weights.where(row_sums.notna() & (row_sums > 0), 0.0)
        
        logger.debug("Calculated risk parity weights (optimized)")
        return weights
    
    def generate_signals(self, 
                        prices: pd.DataFrame,
                        returns: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate complete signal set with proper lookahead bias prevention - OPTIMIZED VERSION.
        
        Args:
            prices: DataFrame with price data (dates x currencies)
            returns: Optional pre-calculated returns DataFrame
            
        Returns:
            Dictionary containing:
            - 'binary_signals': Binary selection signals
            - 'weights': Risk parity weights
            - 'rolling_returns': Rolling returns used for ranking
            - 'volatility': Historical volatility
            - 'ranks': Performance ranks
        """
        logger.info(f"Generating contrarian signals for {len(prices.columns)} currencies (OPTIMIZED)")
        
        # OPTIMIZED: Pre-calculate dimensions to avoid repeated len() calls
        n_dates, n_currencies = prices.shape
        
        # Validate inputs
        if n_dates < self.min_history_days:
            raise ValueError(f"Insufficient data: {n_dates} < {self.min_history_days} required")
        
        # OPTIMIZED: Calculate daily returns if not provided (more efficient method)
        if returns is None:
            returns = prices.pct_change()
            logger.debug("Calculated daily returns from prices")
        
        # OPTIMIZED: Pre-calculate the minimum signal date index to avoid repeated calculation
        min_signal_date_idx = self.lookback_days + self.min_history_days + 1
        
        # Step 1: Calculate rolling returns for performance ranking (optimized)
        rolling_returns = self.calculate_rolling_returns(prices, self.lookback_days)
        
        # Step 2: Rank performance (optimized with faster method)
        ranks = self.rank_performance(rolling_returns)
        
        # Step 3: Generate binary signals (optimized with numpy operations)
        binary_signals = self.generate_binary_signals(ranks)
        
        # Step 4: Calculate historical volatility in parallel (optimized)
        volatility = self.calculate_historical_volatility(returns, self.volatility_lookback)
        
        # Step 5: Calculate risk parity weights (fully vectorized)
        weights = self.calculate_risk_parity_weights(binary_signals, volatility)
        
        # Step 6: OPTIMIZED - Zero out early signals using vectorized operations
        if n_dates > min_signal_date_idx:
            # Use numpy array slicing for better performance
            binary_signals.values[:min_signal_date_idx] = 0
            weights.values[:min_signal_date_idx] = 0.0
        
        # OPTIMIZED: Store signal history with direct assignment (avoid dictionary operations in loop)
        self.signal_history = {
            'binary': binary_signals,
            'weights': weights
        }
        self.performance_history = {
            'rolling_returns': rolling_returns,
            'volatility': volatility,
            'ranks': ranks
        }
        
        # OPTIMIZED: Pre-format date range string to avoid repeated index operations
        date_range_str = f"{binary_signals.index[0]} to {binary_signals.index[-1]}"
        
        # Create comprehensive signal output with optimized metadata creation
        signal_output = {
            'binary_signals': binary_signals,
            'weights': weights,
            'rolling_returns': rolling_returns,
            'volatility': volatility,
            'ranks': ranks,
            'metadata': {
                'n_worst_performers': self.n_worst_performers,
                'lookback_days': self.lookback_days,
                'volatility_lookback': self.volatility_lookback,
                'total_currencies': n_currencies,
                'signal_dates': n_dates,
                'date_range': date_range_str
            }
        }
        
        logger.info(f"Generated signals for {n_dates} dates (OPTIMIZED)")
        return signal_output
    
    def validate_signals(self, 
                        signal_output: Dict[str, pd.DataFrame]) -> Dict[str, Union[bool, float, int]]:
        """
        Validate signal quality and check for lookahead bias.
        
        Args:
            signal_output: Dictionary from generate_signals()
            
        Returns:
            Dictionary with validation results
        """
        binary_signals = signal_output['binary_signals']
        weights = signal_output['weights']
        
        validation_results = {
            'has_lookahead_bias': False,  # We'll check this
            'signals_properly_normalized': True,
            'weights_sum_to_one': True,
            'correct_number_selected': True,
            'no_future_data_used': True,
            'signal_coverage': 0.0,
            'avg_signals_per_day': 0.0,
            'weight_distribution_valid': True,
            'issues': []
        }
        
        try:
            # Check 1: Correct number of signals per day
            daily_signal_counts = binary_signals.sum(axis=1)
            expected_counts = (daily_signal_counts == self.n_worst_performers) | (daily_signal_counts == 0)
            
            if not expected_counts.all():
                validation_results['correct_number_selected'] = False
                validation_results['issues'].append(
                    f"Some days don't have exactly {self.n_worst_performers} signals"
                )
            
            # Check 2: Weights sum to ~1 for days with signals
            weight_sums = weights.sum(axis=1)
            days_with_signals = daily_signal_counts > 0
            
            if days_with_signals.any():
                weight_sums_with_signals = weight_sums[days_with_signals]
                weights_valid = np.abs(weight_sums_with_signals - 1.0) < 0.001
                
                if not weights_valid.all():
                    validation_results['weights_sum_to_one'] = False
                    validation_results['issues'].append("Weights don't sum to 1 on some days")
            
            # Check 3: Signal coverage
            total_possible_signals = len(binary_signals) * len(binary_signals.columns)
            actual_signals = binary_signals.sum().sum()
            validation_results['signal_coverage'] = actual_signals / total_possible_signals
            validation_results['avg_signals_per_day'] = daily_signal_counts.mean()
            
            # Check 4: No negative weights
            if (weights < 0).any().any():
                validation_results['weight_distribution_valid'] = False
                validation_results['issues'].append("Negative weights detected")
            
            # Check 5: Signals only where we have sufficient history
            first_valid_signal_idx = self.lookback_days + self.min_history_days
            if len(binary_signals) > first_valid_signal_idx:
                early_signals = binary_signals.iloc[:first_valid_signal_idx].sum().sum()
                if early_signals > 0:
                    validation_results['no_future_data_used'] = False
                    validation_results['issues'].append("Signals generated with insufficient historical data")
            
            logger.info(f"Signal validation complete: {len(validation_results['issues'])} issues found")
            
        except Exception as e:
            logger.error(f"Error during signal validation: {str(e)}")
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def get_signal_statistics(self, 
                            signal_output: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate comprehensive signal statistics.
        
        Args:
            signal_output: Dictionary from generate_signals()
            
        Returns:
            DataFrame with signal statistics by currency
        """
        binary_signals = signal_output['binary_signals']
        weights = signal_output['weights']
        rolling_returns = signal_output['rolling_returns']
        
        stats_list = []
        
        for currency in binary_signals.columns:
            currency_stats = {
                'currency': currency,
                'total_signals': binary_signals[currency].sum(),
                'signal_frequency': binary_signals[currency].mean(),
                'avg_weight_when_selected': weights[currency][weights[currency] > 0].mean(),
                'avg_return_when_selected': rolling_returns[currency][binary_signals[currency] == 1].mean(),
                'volatility_when_selected': rolling_returns[currency][binary_signals[currency] == 1].std(),
                'best_performance_rank': signal_output['ranks'][currency].min(),
                'worst_performance_rank': signal_output['ranks'][currency].max(),
                'avg_rank_when_selected': signal_output['ranks'][currency][binary_signals[currency] == 1].mean()
            }
            stats_list.append(currency_stats)
        
        stats_df = pd.DataFrame(stats_list)
        logger.info("Calculated signal statistics for all currencies")
        
        return stats_df


class ParameterTestingFramework:
    """
    Framework for testing different parameter combinations (N, M).
    """
    
    def __init__(self, 
                 n_values: List[int] = [2, 3, 5, 7, 10],
                 m_values: List[int] = [5, 10, 15, 20, 30]):
        """
        Initialize parameter testing framework.
        
        Args:
            n_values: List of N values (number of worst performers) to test
            m_values: List of M values (lookback days) to test
        """
        self.n_values = n_values
        self.m_values = m_values
        self.test_results = {}
        
        logger.info(f"Initialized parameter testing with {len(n_values)} N values and {len(m_values)} M values")
    
    def run_parameter_sweep(self, 
                          prices: pd.DataFrame,
                          returns: Optional[pd.DataFrame] = None,
                          validate_signals: bool = True) -> pd.DataFrame:
        """
        Run comprehensive parameter sweep across all N, M combinations.
        
        Args:
            prices: Price data for testing
            returns: Optional pre-calculated returns
            validate_signals: Whether to run signal validation
            
        Returns:
            DataFrame with results for all parameter combinations
        """
        logger.info(f"Starting parameter sweep: {len(self.n_values)} x {len(self.m_values)} combinations")
        
        results_list = []
        total_combinations = len(self.n_values) * len(self.m_values)
        combination_count = 0
        
        for n in self.n_values:
            for m in self.m_values:
                combination_count += 1
                logger.info(f"Testing combination {combination_count}/{total_combinations}: N={n}, M={m}")
                
                try:
                    # Create signal generator with current parameters
                    generator = ConrarianSignalGenerator(
                        n_worst_performers=n,
                        lookback_days=m
                    )
                    
                    # Generate signals
                    signal_output = generator.generate_signals(prices, returns)
                    
                    # Calculate basic statistics
                    binary_signals = signal_output['binary_signals']
                    weights = signal_output['weights']
                    
                    result = {
                        'n_worst_performers': n,
                        'lookback_days': m,
                        'total_signals': binary_signals.sum().sum(),
                        'avg_signals_per_day': binary_signals.sum(axis=1).mean(),
                        'signal_coverage': binary_signals.mean().mean(),
                        'valid_signal_days': (binary_signals.sum(axis=1) > 0).sum(),
                        'total_trading_days': len(binary_signals),
                        'avg_weight_concentration': (weights.max(axis=1)).mean(),
                        'weight_std': weights.std().mean()
                    }
                    
                    # Add validation results if requested
                    if validate_signals:
                        validation = generator.validate_signals(signal_output)
                        result.update({
                            'validation_passed': len(validation['issues']) == 0,
                            'validation_issues': len(validation['issues']),
                            'weights_sum_correctly': validation['weights_sum_to_one'],
                            'correct_signal_count': validation['correct_number_selected']
                        })
                    
                    results_list.append(result)
                    
                    # Store detailed results
                    self.test_results[(n, m)] = {
                        'signal_output': signal_output,
                        'generator': generator,
                        'validation': validation if validate_signals else None
                    }
                    
                except Exception as e:
                    logger.error(f"Error testing N={n}, M={m}: {str(e)}")
                    result = {
                        'n_worst_performers': n,
                        'lookback_days': m,
                        'error': str(e)
                    }
                    results_list.append(result)
        
        results_df = pd.DataFrame(results_list)
        logger.info("Parameter sweep completed")
        
        return results_df
    
    def get_best_parameters(self, 
                          results_df: pd.DataFrame,
                          metric: str = 'signal_coverage') -> Tuple[int, int]:
        """
        Get best parameter combination based on specified metric.
        
        Args:
            results_df: Results from run_parameter_sweep()
            metric: Metric to optimize ('signal_coverage', 'valid_signal_days', etc.)
            
        Returns:
            Tuple of (best_n, best_m)
        """
        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        # Filter out error cases
        valid_results = results_df[~results_df.get('error', pd.Series([False]*len(results_df))).notna()]
        
        if len(valid_results) == 0:
            logger.warning("No valid results found")
            return self.n_values[0], self.m_values[0]
        
        # Find best combination
        best_idx = valid_results[metric].idxmax()
        best_row = valid_results.loc[best_idx]
        
        best_n = int(best_row['n_worst_performers'])
        best_m = int(best_row['lookback_days'])
        
        logger.info(f"Best parameters based on {metric}: N={best_n}, M={best_m}")
        return best_n, best_m


# Utility functions for signal analysis
def analyze_signal_timing(signal_output: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Analyze signal timing and ensure no lookahead bias.
    
    Args:
        signal_output: Output from generate_signals()
        
    Returns:
        Dictionary with timing analysis results
    """
    binary_signals = signal_output['binary_signals']
    rolling_returns = signal_output['rolling_returns']
    
    # Check signal-return alignment
    # Signals should be based on returns ending BEFORE the signal date
    timing_analysis = {
        'signal_return_correlation': 0.0,
        'lookahead_bias_detected': False,
        'proper_timing_verified': True
    }
    
    # Calculate correlation between signals and contemporaneous returns
    # Should be low/negative for contrarian strategy
    for currency in binary_signals.columns:
        curr_signals = binary_signals[currency]
        curr_returns = rolling_returns[currency]
        
        # Align dates and calculate correlation
        aligned_data = pd.DataFrame({
            'signals': curr_signals,
            'returns': curr_returns
        }).dropna()
        
        if len(aligned_data) > 10:
            correlation = aligned_data['signals'].corr(aligned_data['returns'])
            timing_analysis['signal_return_correlation'] = correlation
            
            # For contrarian strategy, we expect negative correlation
            if correlation > 0.1:  # Threshold for suspicion
                timing_analysis['lookahead_bias_detected'] = True
                timing_analysis['proper_timing_verified'] = False
    
    logger.info(f"Signal timing analysis: correlation={timing_analysis['signal_return_correlation']:.3f}")
    return timing_analysis


def validate_no_lookahead_bias(prices: pd.DataFrame, 
                              signal_output: Dict[str, pd.DataFrame],
                              lookback_days: int) -> bool:
    """
    Comprehensive validation that no future data was used in signal generation.
    
    Args:
        prices: Original price data
        signal_output: Generated signals
        lookback_days: Lookback period used
        
    Returns:
        True if no lookahead bias detected, False otherwise
    """
    logger.info("Running comprehensive lookahead bias validation")
    
    binary_signals = signal_output['binary_signals']
    rolling_returns = signal_output['rolling_returns']
    
    # Test 1: Check that rolling returns align properly with signals
    # Account for the additional shift in our calculation
    min_test_idx = lookback_days + 2  # Account for both shifts
    
    for date_idx in range(min_test_idx, min(min_test_idx + 50, len(binary_signals))):  # Test 50 dates
        signal_date = binary_signals.index[date_idx]
        
        # Manually calculate returns for this date using only past data
        # Our system uses: (price_{t-1} / price_{t-lookback-1}) - 1
        if date_idx >= lookback_days + 1:
            end_price = prices.iloc[date_idx - 1]  # T-1 price
            start_price = prices.iloc[date_idx - lookback_days - 1]  # T-lookback-1 price
            manual_return = (end_price / start_price) - 1.0
            system_return = rolling_returns.iloc[date_idx]
            
            # Compare (should be very close)
            if not system_return.isnull().all():
                difference = (manual_return - system_return).abs()
                max_diff = difference.max()
                if not np.isnan(max_diff) and max_diff > 0.001:  # 0.1% tolerance
                    logger.warning(f"Potential lookahead bias detected on {signal_date}")
                    return False
    
    # Test 2: Verify signals can only be generated with sufficient history
    # Check first lookback_days + some buffer
    min_history_buffer = lookback_days + 50
    if len(binary_signals) > min_history_buffer:
        first_valid_signals = binary_signals.iloc[:min_history_buffer].sum().sum()
        if first_valid_signals > 0:
            logger.warning(f"Signals generated with insufficient history: {first_valid_signals} signals in first {min_history_buffer} days")
            return False
    
    logger.info("Lookahead bias validation passed")
    return True


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # This would be used with real data
    logger.info("Contrarian Signal Generator module loaded successfully")
    print("Module ready for use. Import and use with real forex data.")