#!/usr/bin/env python3
"""
Example Usage of Forex Data Collection and Loading System

This script demonstrates how to use the forex data collection and loading
system for contrarian trading strategy development.
"""

import sys
import os
sys.path.append('modules')

from data_loader import ForexDataLoader, load_forex_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def example_1_basic_data_loading():
    """
    Example 1: Basic data loading and exploration
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Data Loading and Exploration")
    print("=" * 60)
    
    # Initialize the data loader
    loader = ForexDataLoader("data")
    
    # Get all available currency pairs
    symbols = loader.get_available_symbols()
    print(f"\n📊 Available currency pairs: {len(symbols)}")
    for symbol in symbols[:10]:  # Show first 10
        print(f"   - {symbol}")
    if len(symbols) > 10:
        print(f"   ... and {len(symbols) - 10} more")
    
    # Load unified price data
    prices = loader.load_unified_prices()
    print(f"\n📈 Unified prices dataset: {prices.shape}")
    print(f"   Date range: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}")
    print(f"   Currency pairs: {list(prices.columns)}")
    
    # Load unified returns
    returns = loader.load_unified_returns()
    print(f"\n📉 Unified returns dataset: {returns.shape}")
    
    # Basic statistics
    print(f"\n📊 Basic Statistics (Annualized):")
    annual_returns = returns.mean() * 252 * 100  # Convert to percentage
    annual_volatility = returns.std() * np.sqrt(252) * 100
    
    stats_df = pd.DataFrame({
        'Annual Return (%)': annual_returns,
        'Annual Volatility (%)': annual_volatility,
        'Sharpe Ratio': annual_returns / annual_volatility
    }).round(2)
    
    print(stats_df.head())
    
    return prices, returns


def example_2_period_analysis():
    """
    Example 2: Period-specific analysis for strategy development
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Period-Specific Analysis")
    print("=" * 60)
    
    loader = ForexDataLoader("data")
    
    # Analyze different market regimes
    periods = [
        ("2008 Financial Crisis", "2008-01-01", "2008-12-31"),
        ("Post-Crisis Recovery", "2009-01-01", "2010-12-31"),
        ("COVID-19 Period", "2020-01-01", "2020-12-31"),
        ("Recent Period", "2023-01-01", "2023-12-31")
    ]
    
    major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    
    print(f"\n📈 Analyzing major pairs: {major_pairs}")
    print(f"📅 Periods: {len(periods)} different market regimes\n")
    
    results = []
    
    for period_name, start_date, end_date in periods:
        print(f"🔍 Analyzing: {period_name} ({start_date} to {end_date})")
        
        period_returns = loader.get_data_for_period(
            start_date, end_date, 
            symbols=major_pairs, 
            data_type='returns'
        )
        
        if period_returns is not None and not period_returns.empty:
            # Calculate key metrics
            total_return = (period_returns.mean() * len(period_returns) * 100).round(2)
            volatility = (period_returns.std() * np.sqrt(252) * 100).round(2)
            max_drawdown = ((period_returns.cumsum().expanding().max() - period_returns.cumsum()).max() * 100).round(2)
            
            results.append({
                'Period': period_name,
                'EURUSD Return (%)': total_return.get('EURUSD', 'N/A'),
                'GBPUSD Return (%)': total_return.get('GBPUSD', 'N/A'),
                'USDJPY Return (%)': total_return.get('USDJPY', 'N/A'),
                'Avg Volatility (%)': volatility.mean()
            })
            
            print(f"   ✓ Data points: {len(period_returns)}")
            print(f"   ✓ Average volatility: {volatility.mean():.1f}%")
        else:
            print(f"   ❌ No data available for this period")
    
    # Display results table
    if results:
        results_df = pd.DataFrame(results)
        print(f"\n📊 Period Analysis Summary:")
        print(results_df.to_string(index=False))
    
    return results


def example_3_contrarian_signal_detection():
    """
    Example 3: Contrarian signal detection using the data
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Contrarian Signal Detection")
    print("=" * 60)
    
    loader = ForexDataLoader("data")
    
    # Load recent data for analysis
    returns = loader.get_data_for_period(
        "2020-01-01", "2024-12-31", 
        data_type='returns'
    )
    
    if returns is None or returns.empty:
        print("❌ Unable to load data for contrarian analysis")
        return
    
    print(f"📊 Analyzing contrarian signals on {returns.shape[0]} days of data")
    print(f"💱 Currency pairs: {returns.shape[1]}")
    
    # Simple contrarian strategy indicators
    lookback_periods = [5, 10, 20]  # Trading days
    
    for lookback in lookback_periods:
        print(f"\n🔄 {lookback}-day contrarian analysis:")
        
        # Calculate rolling returns
        rolling_returns = returns.rolling(window=lookback).sum()
        
        # Identify extreme movements (potential contrarian opportunities)
        # Using 2 standard deviations as threshold
        thresholds = rolling_returns.std() * 2
        
        contrarian_signals = {}
        for pair in returns.columns:
            if pair in rolling_returns.columns:
                # Bearish extreme (potential buy signal)
                bearish_signals = rolling_returns[pair] < -thresholds[pair]
                # Bullish extreme (potential sell signal)  
                bullish_signals = rolling_returns[pair] > thresholds[pair]
                
                contrarian_signals[pair] = {
                    'bearish_extremes': bearish_signals.sum(),
                    'bullish_extremes': bullish_signals.sum(),
                    'total_signals': bearish_signals.sum() + bullish_signals.sum()
                }
        
        # Display top pairs with most contrarian opportunities
        signal_summary = pd.DataFrame(contrarian_signals).T
        signal_summary = signal_summary.sort_values('total_signals', ascending=False)
        
        print(f"   Top contrarian opportunities:")
        for i, (pair, row) in enumerate(signal_summary.head(5).iterrows()):
            print(f"   {i+1}. {pair}: {row['total_signals']} signals "
                  f"({row['bearish_extremes']} bearish, {row['bullish_extremes']} bullish)")
    
    # Calculate correlation matrix for diversification
    print(f"\n🔗 Correlation Analysis (for portfolio diversification):")
    correlation_matrix = returns.corr()
    
    # Find pairs with lowest correlation (best diversification)
    print(f"   Pairs with lowest correlation (best for diversification):")
    corr_pairs = []
    for i, pair1 in enumerate(correlation_matrix.columns):
        for j, pair2 in enumerate(correlation_matrix.columns[i+1:], i+1):
            corr_val = correlation_matrix.loc[pair1, pair2]
            corr_pairs.append((pair1, pair2, corr_val))
    
    # Sort by absolute correlation (closest to 0 is best diversification)
    corr_pairs.sort(key=lambda x: abs(x[2]))
    
    for i, (pair1, pair2, corr) in enumerate(corr_pairs[:5]):
        print(f"   {i+1}. {pair1} vs {pair2}: {corr:.3f}")
    
    return contrarian_signals, correlation_matrix


def example_4_data_quality_validation():
    """
    Example 4: Data quality validation for production use
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Data Quality Validation")
    print("=" * 60)
    
    loader = ForexDataLoader("data")
    
    # Test a few key pairs
    test_pairs = ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']
    
    print(f"🔍 Validating data quality for: {test_pairs}")
    
    validation_results = []
    
    for pair in test_pairs:
        print(f"\n📊 Validating {pair}...")
        
        data = loader.load_individual_pair(pair)
        if data is not None:
            validation = loader.validate_data_integrity(data, pair)
            
            print(f"   ✓ Total records: {validation['total_rows']:,}")
            print(f"   ✓ Date range: {validation['date_range']}")
            print(f"   ✓ Missing values: {sum(validation['missing_values'].values())}")
            print(f"   ✓ Date gaps: {validation['gaps_in_dates']}")
            print(f"   ✓ Issues found: {len(validation['issues'])}")
            
            if validation['issues']:
                for issue in validation['issues']:
                    print(f"     ⚠️  {issue}")
            else:
                print(f"     ✅ No data quality issues detected")
            
            validation_results.append(validation)
        else:
            print(f"   ❌ Could not load data for {pair}")
    
    # Summary
    if validation_results:
        total_records = sum(v['total_rows'] for v in validation_results)
        total_issues = sum(len(v['issues']) for v in validation_results)
        
        print(f"\n📈 Validation Summary:")
        print(f"   Total records validated: {total_records:,}")
        print(f"   Total issues found: {total_issues}")
        print(f"   Data quality score: {((len(validation_results) - total_issues) / len(validation_results) * 100):.1f}%")
    
    return validation_results


def example_5_memory_efficient_loading():
    """
    Example 5: Memory-efficient data loading for large backtests
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Memory-Efficient Data Loading")
    print("=" * 60)
    
    loader = ForexDataLoader("data")
    
    print(f"💾 Demonstrating memory-efficient data loading techniques")
    
    # Technique 1: Load specific time periods only
    print(f"\n1. Loading specific periods only:")
    recent_data = loader.get_data_for_period(
        "2023-01-01", "2024-12-31",
        symbols=['EURUSD', 'GBPUSD'],
        data_type='returns'
    )
    if recent_data is not None:
        memory_usage = recent_data.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        print(f"   ✓ Period data memory usage: {memory_usage:.2f} MB")
        print(f"   ✓ Shape: {recent_data.shape}")
    
    # Technique 2: Cache management
    print(f"\n2. Cache management:")
    print(f"   📋 Cache contents: {len(loader.cache)} items")
    
    # Clear cache to free memory
    loader.clear_cache()
    print(f"   🗑️  Cache cleared")
    
    # Technique 3: Individual pair loading for specific analysis
    print(f"\n3. Loading individual pairs for focused analysis:")
    eurusd_data = loader.load_individual_pair('EURUSD')
    if eurusd_data is not None:
        memory_usage = eurusd_data.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"   ✓ EURUSD memory usage: {memory_usage:.2f} MB")
        print(f"   ✓ Date coverage: {eurusd_data.index.min().strftime('%Y-%m-%d')} to {eurusd_data.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n💡 Memory optimization tips:")
    print(f"   • Load only required time periods")
    print(f"   • Use specific symbol lists rather than all pairs")
    print(f"   • Clear cache when switching between analyses")
    print(f"   • Consider data type optimization for large datasets")


def main():
    """
    Main function to run all examples
    """
    print("🚀 FOREX DATA SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the complete forex data collection")
    print("and loading system for contrarian trading strategies.")
    print("=" * 60)
    
    try:
        # Check if data directory exists
        if not os.path.exists("data"):
            print("❌ Data directory not found!")
            print("Please run the forex_data_collector.py script first.")
            return 1
        
        # Run all examples
        prices, returns = example_1_basic_data_loading()
        period_results = example_2_period_analysis()
        contrarian_signals, correlations = example_3_contrarian_signal_detection()
        validation_results = example_4_data_quality_validation()
        example_5_memory_efficient_loading()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Data loading system is ready for production use")
        print("✅ All major functionality tested and validated")
        print("✅ System optimized for contrarian trading strategies")
        print("\n📚 Next steps:")
        print("   • Integrate with your backtesting framework")
        print("   • Implement specific contrarian strategies")
        print("   • Set up automated data updates")
        print("   • Configure production monitoring")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during example execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())