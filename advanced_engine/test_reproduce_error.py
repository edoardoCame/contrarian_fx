#!/usr/bin/env python3
"""
Reproduce IndexError from Notebook

This script reproduces the exact same scenario as the notebook to identify
the root cause of the IndexError at line 384 in portfolio_manager.py
"""

import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path

# Add modules path
sys.path.append(str(Path(__file__).parent / 'modules'))

from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager

def reproduce_notebook_error():
    """Reproduce the exact scenario from the notebook"""
    print("🔍 REPRODUCING NOTEBOOK ERROR")
    print("="*60)
    
    try:
        # 1. Load data exactly as in notebook
        data_loader = ForexDataLoader('data')
        prices = data_loader.load_unified_prices()
        returns = data_loader.load_unified_returns()
        
        print(f"✅ Data loaded: prices {prices.shape}, returns {returns.shape}")
        
        # 2. Initialize with the EXACT same parameters as notebook
        # These are the best parameters from optimization
        best_n = 3
        best_m = 30
        
        signal_generator = ConrarianSignalGenerator(
            n_worst_performers=best_n, 
            lookback_days=best_m
        )
        
        print(f"✅ Signal generator initialized: N={best_n}, M={best_m}")
        
        # 3. Generate signals for FULL dataset (not subset)
        print("📊 Generating signals for FULL dataset...")
        signal_output = signal_generator.generate_signals(prices, returns)
        
        print(f"✅ Signals generated:")
        for key, value in signal_output.items():
            if key != 'metadata':
                print(f"    {key}: {value.shape}")
        
        # 4. Initialize portfolio manager with EXACT same parameters as notebook
        portfolio_manager = PortfolioManager(
            volatility_method='ewma',
            risk_parity_method='erc',  # This is where the error occurs!
            target_volatility=0.12,
            max_position_size=0.3
        )
        
        print(f"✅ Portfolio manager initialized")
        
        # 5. Run portfolio management (this should trigger the error)
        print("🎯 Running portfolio management - this should trigger the IndexError...")
        portfolio_results = portfolio_manager.run_portfolio_management(signal_output, returns)
        
        print("❌ ERROR NOT REPRODUCED - This means something is different!")
        
    except Exception as e:
        print(f"✅ ERROR REPRODUCED: {type(e).__name__}: {e}")
        print("\n📍 FULL TRACEBACK:")
        traceback.print_exc()
        
        # Let's analyze the specific error
        if "index 2 is out of bounds for axis 0 with size 2" in str(e):
            print(f"\n🔍 ANALYSIS OF IndexError:")
            print(f"The error suggests we're trying to access index 2 in an array of size 2")
            print(f"Valid indices for size 2 array are: 0, 1")
            print(f"This happens in correlation matrix indexing at line 384")
            
            # Let's debug what's happening at the specific point of failure
            debug_specific_error(signal_output, returns, portfolio_manager)

def debug_specific_error(signal_output, returns, portfolio_manager):
    """Debug the specific point where the IndexError occurs"""
    print(f"\n🔬 DETAILED ERROR ANALYSIS")
    print("="*40)
    
    try:
        # Find a date where we have actual signals (not all zeros)
        binary_signals = signal_output['binary_signals']
        
        # Look for dates with actual selections
        signal_dates = binary_signals[binary_signals.sum(axis=1) > 0].index
        
        if len(signal_dates) == 0:
            print("❌ No dates with active signals found!")
            return
        
        test_date = signal_dates[0]  # Use first date with signals
        print(f"📅 Testing with date: {test_date}")
        
        # Get the selections for this date
        current_signals = binary_signals.loc[test_date]
        selected_assets = current_signals > 0
        
        print(f"🎯 Selected assets: {selected_assets.sum()}")
        print(f"📋 Asset names: {selected_assets[selected_assets].index.tolist()}")
        
        # Calculate volatility (this works fine)
        volatility_estimates = portfolio_manager.volatility_estimator.estimate_volatility(
            returns, window=60
        )
        
        # Get volatility for the test date
        available_dates = volatility_estimates.index[volatility_estimates.index <= test_date]
        if len(available_dates) == 0:
            print("❌ No volatility data available")
            return
        
        vol_date = available_dates[-1]
        current_volatility = volatility_estimates.loc[vol_date]
        
        # Try to calculate correlation matrix (this is where the error likely occurs)
        print(f"🧪 Testing correlation matrix calculation...")
        
        # Get historical data up to test date
        historical_data = returns.loc[returns.index < test_date]
        print(f"📊 Historical data shape: {historical_data.shape}")
        
        if len(historical_data) < 126:  # correlation_lookback
            print(f"⚠️ Insufficient historical data: {len(historical_data)} < 126")
            return
        
        recent_data = historical_data.tail(126)  # correlation_lookback
        print(f"📊 Recent data shape: {recent_data.shape}")
        
        # Filter to selected assets
        selected_columns = selected_assets.index[selected_assets]
        print(f"🔍 Selected columns: {selected_columns.tolist()}")
        
        correlation_data = recent_data[selected_columns].dropna()
        print(f"📊 Correlation data shape: {correlation_data.shape}")
        
        if len(correlation_data) < 63:  # correlation_lookback // 2
            print(f"⚠️ Insufficient clean correlation data: {len(correlation_data)} < 63")
            return
        
        # Calculate correlation matrix
        correlation_matrix = correlation_data.corr().values
        print(f"✅ Correlation matrix shape: {correlation_matrix.shape}")
        
        # Now test the problematic line
        print(f"🔍 Testing the problematic indexing...")
        
        # Get asset indices (this is the problematic part)
        asset_indices = np.where(selected_assets)[0]
        print(f"📊 Asset indices: {asset_indices}")
        print(f"📊 Asset indices length: {len(asset_indices)}")
        print(f"📊 Correlation matrix shape: {correlation_matrix.shape}")
        
        # This should be the problematic line
        if len(asset_indices) != correlation_matrix.shape[0]:
            print(f"❌ MISMATCH DETECTED!")
            print(f"    Asset indices length: {len(asset_indices)}")
            print(f"    Correlation matrix size: {correlation_matrix.shape[0]}")
            print(f"    This is the source of the IndexError!")
            
            # Show exactly what's different
            print(f"\n🔍 ROOT CAUSE ANALYSIS:")
            print(f"selected_assets (boolean mask):")
            print(selected_assets)
            print(f"\nasset_indices (where selected_assets is True):")
            print(asset_indices)
            print(f"\ncorrelation_data.columns:")
            print(correlation_data.columns.tolist())
            print(f"\noriginal returns.columns:")
            print(returns.columns.tolist())
            
        else:
            # Try the actual indexing
            try:
                filtered_corr = correlation_matrix[np.ix_(asset_indices, asset_indices)]
                print(f"✅ Indexing successful: {filtered_corr.shape}")
            except IndexError as ie:
                print(f"❌ IndexError during indexing: {ie}")
                print(f"Trying to access indices {asset_indices} in matrix of shape {correlation_matrix.shape}")
        
    except Exception as debug_e:
        print(f"❌ Debug error: {debug_e}")
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_notebook_error()