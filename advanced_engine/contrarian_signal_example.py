#!/usr/bin/env python3
"""
Contrarian Signal Generator Example Usage and Testing Script

This script demonstrates how to use the ConrarianSignalGenerator with real forex data,
perform parameter optimization, and validate signal quality.

Features demonstrated:
- Loading forex data using the data loader
- Generating contrarian signals with different parameters
- Running parameter sweep for optimization
- Comprehensive signal validation
- Performance analysis and visualization
- Lookahead bias testing

Author: Claude Code
Date: 2025-08-06
"""

import sys
import os
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / 'modules'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple

# Import our modules
from data_loader import ForexDataLoader, load_forex_data
from signal_generator import (
    ConrarianSignalGenerator, 
    ParameterTestingFramework,
    analyze_signal_timing,
    validate_no_lookahead_bias
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contrarian_signals.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_sample_data(start_date: str = '2010-01-01', 
                    end_date: str = '2024-12-31') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load sample forex data for testing.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Tuple of (prices, returns) DataFrames
    """
    logger.info(f"Loading forex data from {start_date} to {end_date}")
    
    # Load data using the data loader
    loader = ForexDataLoader('advanced_engine/data')
    
    # Load unified prices and returns
    prices = loader.get_data_for_period(start_date, end_date, data_type='prices')
    returns = loader.get_data_for_period(start_date, end_date, data_type='returns')
    
    if prices is None or returns is None:
        logger.error("Failed to load data")
        return None, None
    
    # Remove currencies with too much missing data
    price_coverage = (1 - prices.isnull().mean())
    valid_currencies = price_coverage[price_coverage > 0.8].index  # At least 80% coverage
    
    prices_clean = prices[valid_currencies].fillna(method='ffill').dropna()
    returns_clean = returns[valid_currencies].fillna(0)
    
    # Align the dataframes
    common_dates = prices_clean.index.intersection(returns_clean.index)
    prices_clean = prices_clean.loc[common_dates]
    returns_clean = returns_clean.loc[common_dates]
    
    logger.info(f"Loaded clean data: {prices_clean.shape} prices, {returns_clean.shape} returns")
    logger.info(f"Available currencies: {list(prices_clean.columns)}")
    
    return prices_clean, returns_clean


def basic_signal_generation_example():
    """
    Basic example of generating contrarian signals.
    """
    logger.info("=== BASIC SIGNAL GENERATION EXAMPLE ===")
    
    # Load data
    prices, returns = load_sample_data('2015-01-01', '2023-12-31')
    if prices is None:
        logger.error("Cannot proceed without data")
        return
    
    # Create signal generator
    generator = ConrarianSignalGenerator(
        n_worst_performers=5,  # Select 5 worst performers
        lookback_days=20,      # Look back 20 days
        min_history_days=252,  # Require 1 year of history
        volatility_lookback=60 # Use 60 days for volatility
    )
    
    # Generate signals
    logger.info("Generating contrarian signals...")
    signal_output = generator.generate_signals(prices, returns)
    
    # Display results
    binary_signals = signal_output['binary_signals']
    weights = signal_output['weights']
    
    logger.info(f"Generated signals for {len(binary_signals)} trading days")
    logger.info(f"Signal period: {binary_signals.index.min()} to {binary_signals.index.max()}")
    
    # Basic statistics
    total_signals = binary_signals.sum().sum()
    trading_days = len(binary_signals)
    avg_signals_per_day = binary_signals.sum(axis=1).mean()
    
    logger.info(f"Total signals generated: {total_signals}")
    logger.info(f"Average signals per day: {avg_signals_per_day:.2f}")
    logger.info(f"Signal frequency: {total_signals/(trading_days * len(binary_signals.columns)):.3f}")
    
    # Show signal statistics by currency
    signal_stats = generator.get_signal_statistics(signal_output)
    print("\nSignal Statistics by Currency:")
    print(signal_stats[['currency', 'total_signals', 'signal_frequency', 'avg_weight_when_selected']].head(10))
    
    # Validate signals
    validation = generator.validate_signals(signal_output)
    logger.info(f"Signal validation: {len(validation['issues'])} issues found")
    if validation['issues']:
        for issue in validation['issues']:
            logger.warning(f"Validation issue: {issue}")
    else:
        logger.info("All signal validations passed!")
    
    return signal_output, generator


def parameter_optimization_example():
    """
    Example of parameter optimization using the testing framework.
    """
    logger.info("=== PARAMETER OPTIMIZATION EXAMPLE ===")
    
    # Load data (smaller sample for faster testing)
    prices, returns = load_sample_data('2020-01-01', '2023-12-31')
    if prices is None:
        logger.error("Cannot proceed without data")
        return
    
    # Create parameter testing framework
    tester = ParameterTestingFramework(
        n_values=[2, 3, 5, 7],        # Test different numbers of worst performers
        m_values=[10, 15, 20, 30]     # Test different lookback periods
    )
    
    # Run parameter sweep
    logger.info("Running parameter optimization sweep...")
    results_df = tester.run_parameter_sweep(prices, returns, validate_signals=True)
    
    # Display results
    print("\nParameter Optimization Results:")
    print(results_df[['n_worst_performers', 'lookback_days', 'total_signals', 
                     'avg_signals_per_day', 'signal_coverage', 'validation_passed']].head(10))
    
    # Find best parameters
    best_n, best_m = tester.get_best_parameters(results_df, metric='signal_coverage')
    logger.info(f"Best parameters: N={best_n} worst performers, M={best_m} lookback days")
    
    # Test the best parameters
    logger.info("Testing best parameters...")
    best_generator = ConrarianSignalGenerator(
        n_worst_performers=best_n,
        lookback_days=best_m
    )
    
    best_signals = best_generator.generate_signals(prices, returns)
    best_validation = best_generator.validate_signals(best_signals)
    
    logger.info(f"Best parameter results:")
    logger.info(f"- Total signals: {best_signals['binary_signals'].sum().sum()}")
    logger.info(f"- Validation issues: {len(best_validation['issues'])}")
    
    return results_df, best_generator, best_signals


def comprehensive_signal_validation():
    """
    Comprehensive signal validation including lookahead bias testing.
    """
    logger.info("=== COMPREHENSIVE SIGNAL VALIDATION ===")
    
    # Load data
    prices, returns = load_sample_data('2018-01-01', '2022-12-31')
    if prices is None:
        logger.error("Cannot proceed without data")
        return
    
    # Test multiple parameter combinations
    test_configs = [
        {'n': 3, 'm': 15},
        {'n': 5, 'm': 20},
        {'n': 7, 'm': 30}
    ]
    
    validation_results = []
    
    for config in test_configs:
        logger.info(f"Validating N={config['n']}, M={config['m']}")
        
        # Create generator
        generator = ConrarianSignalGenerator(
            n_worst_performers=config['n'],
            lookback_days=config['m']
        )
        
        # Generate signals
        signal_output = generator.generate_signals(prices, returns)
        
        # Standard validation
        validation = generator.validate_signals(signal_output)
        
        # Timing analysis
        timing = analyze_signal_timing(signal_output)
        
        # Lookahead bias check
        no_lookahead = validate_no_lookahead_bias(prices, signal_output, config['m'])
        
        result = {
            'n': config['n'],
            'm': config['m'],
            'standard_validation_passed': len(validation['issues']) == 0,
            'timing_correlation': timing['signal_return_correlation'],
            'lookahead_bias_detected': timing['lookahead_bias_detected'],
            'no_future_data_used': no_lookahead,
            'issues': len(validation['issues'])
        }
        
        validation_results.append(result)
        
        logger.info(f"Results for N={config['n']}, M={config['m']}:")
        logger.info(f"  - Standard validation: {'PASS' if result['standard_validation_passed'] else 'FAIL'}")
        logger.info(f"  - Timing correlation: {result['timing_correlation']:.3f}")
        logger.info(f"  - No lookahead bias: {'PASS' if result['no_future_data_used'] else 'FAIL'}")
    
    # Summary
    validation_df = pd.DataFrame(validation_results)
    print("\nValidation Summary:")
    print(validation_df)
    
    # Overall assessment
    all_passed = validation_df['standard_validation_passed'].all() and \
                validation_df['no_future_data_used'].all() and \
                not validation_df['lookahead_bias_detected'].any()
    
    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED! Signal generator is working correctly.")
    else:
        logger.warning("⚠️  Some validations failed. Please review the issues.")
    
    return validation_df


def signal_quality_analysis():
    """
    Analyze signal quality and characteristics.
    """
    logger.info("=== SIGNAL QUALITY ANALYSIS ===")
    
    # Load data
    prices, returns = load_sample_data('2019-01-01', '2023-12-31')
    if prices is None:
        logger.error("Cannot proceed without data")
        return
    
    # Generate signals with standard parameters
    generator = ConrarianSignalGenerator(
        n_worst_performers=5,
        lookback_days=20
    )
    
    signal_output = generator.generate_signals(prices, returns)
    binary_signals = signal_output['binary_signals']
    weights = signal_output['weights']
    rolling_returns = signal_output['rolling_returns']
    
    # Analysis 1: Signal distribution over time
    monthly_signals = binary_signals.resample('M').sum().sum(axis=1)
    logger.info(f"Average monthly signals: {monthly_signals.mean():.1f}")
    logger.info(f"Signal volatility (monthly): {monthly_signals.std():.1f}")
    
    # Analysis 2: Currency selection frequency
    selection_frequency = binary_signals.mean().sort_values(ascending=False)
    logger.info("Top 5 most frequently selected currencies:")
    for currency, freq in selection_frequency.head().items():
        logger.info(f"  {currency}: {freq:.3f} ({freq*100:.1f}%)")
    
    # Analysis 3: Weight concentration
    daily_max_weight = weights.max(axis=1)
    daily_weight_std = weights.std(axis=1)
    
    logger.info(f"Average maximum daily weight: {daily_max_weight.mean():.3f}")
    logger.info(f"Average weight dispersion: {daily_weight_std.mean():.3f}")
    
    # Analysis 4: Performance when selected
    performance_when_selected = {}
    for currency in binary_signals.columns:
        mask = binary_signals[currency] == 1
        if mask.sum() > 0:
            perf = rolling_returns[currency][mask].mean()
            performance_when_selected[currency] = perf
    
    avg_performance_when_selected = np.mean(list(performance_when_selected.values()))
    logger.info(f"Average return of selected currencies: {avg_performance_when_selected:.4f} ({avg_performance_when_selected*100:.2f}%)")
    
    # Analysis 5: Signal persistence
    signal_changes = binary_signals.diff().abs().sum(axis=1)
    avg_daily_changes = signal_changes.mean()
    logger.info(f"Average daily signal changes: {avg_daily_changes:.1f}")
    
    return {
        'monthly_signals': monthly_signals,
        'selection_frequency': selection_frequency,
        'performance_when_selected': performance_when_selected,
        'signal_changes': signal_changes
    }


def demonstrate_real_time_usage():
    """
    Demonstrate how the signal generator would be used in real-time trading.
    """
    logger.info("=== REAL-TIME USAGE DEMONSTRATION ===")
    
    # Load full dataset
    prices, returns = load_sample_data('2010-01-01', '2024-07-31')
    if prices is None:
        logger.error("Cannot proceed without data")
        return
    
    # Create generator
    generator = ConrarianSignalGenerator(
        n_worst_performers=5,
        lookback_days=20
    )
    
    # Simulate real-time signal generation for the last month
    simulation_start = prices.index[-30]  # Last 30 days
    logger.info(f"Simulating real-time signals from {simulation_start}")
    
    daily_signals = []
    
    # Simulate day-by-day signal generation
    for i, current_date in enumerate(prices.index[-30:]):
        # Use data up to previous day for signal generation
        data_cutoff = prices.index.get_loc(current_date)
        historical_prices = prices.iloc[:data_cutoff+1]  # Include current day
        historical_returns = returns.iloc[:data_cutoff+1]
        
        if len(historical_prices) < generator.min_history_days:
            continue
        
        try:
            # Generate signals (this would be the daily process)
            signal_output = generator.generate_signals(historical_prices, historical_returns)
            
            # Get today's signals (last row)
            today_signals = signal_output['binary_signals'].iloc[-1]
            today_weights = signal_output['weights'].iloc[-1]
            
            # Store for analysis
            selected_currencies = today_signals[today_signals == 1].index.tolist()
            signal_weights = {curr: today_weights[curr] for curr in selected_currencies}
            
            daily_signals.append({
                'date': current_date,
                'selected_currencies': selected_currencies,
                'weights': signal_weights,
                'n_selected': len(selected_currencies)
            })
            
            # Log every 5th day
            if i % 5 == 0:
                logger.info(f"Signals for {current_date.strftime('%Y-%m-%d')}: {selected_currencies}")
        
        except Exception as e:
            logger.warning(f"Failed to generate signals for {current_date}: {str(e)}")
    
    # Summary of real-time simulation
    if daily_signals:
        avg_selections = np.mean([s['n_selected'] for s in daily_signals])
        logger.info(f"Average daily selections: {avg_selections:.1f}")
        
        # Show last few days
        logger.info("Last 5 days of signals:")
        for signal_day in daily_signals[-5:]:
            date_str = signal_day['date'].strftime('%Y-%m-%d')
            currencies = ', '.join(signal_day['selected_currencies'])
            logger.info(f"  {date_str}: {currencies}")
    
    return daily_signals


def main():
    """
    Main function to run all examples.
    """
    logger.info("Starting Contrarian Signal Generator Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Basic signal generation
        signal_output, generator = basic_signal_generation_example()
        
        print("\n" + "="*60)
        
        # Example 2: Parameter optimization
        param_results, best_generator, best_signals = parameter_optimization_example()
        
        print("\n" + "="*60)
        
        # Example 3: Comprehensive validation
        validation_results = comprehensive_signal_validation()
        
        print("\n" + "="*60)
        
        # Example 4: Signal quality analysis
        quality_analysis = signal_quality_analysis()
        
        print("\n" + "="*60)
        
        # Example 5: Real-time usage demonstration
        realtime_signals = demonstrate_real_time_usage()
        
        logger.info("\n🎉 All examples completed successfully!")
        logger.info("Check the log file 'contrarian_signals.log' for detailed output.")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()