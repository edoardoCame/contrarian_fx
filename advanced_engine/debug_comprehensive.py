#!/usr/bin/env python3
"""
Comprehensive Debug Script for Contrarian Forex Backtesting System

This script performs systematic debugging of the entire pipeline:
1. Data loading and validation
2. Signal generation testing
3. Portfolio manager debugging
4. Integration testing
5. Full pipeline validation

Author: Claude Code
Date: 2025-08-07
"""

import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add modules path
sys.path.append(str(Path(__file__).parent / 'modules'))

try:
    from data_loader import ForexDataLoader
    from signal_generator import ConrarianSignalGenerator
    from portfolio_manager import PortfolioManager
    from backtesting_engine import BacktestingEngine
    logger.info("✅ All modules imported successfully")
except Exception as e:
    logger.error(f"❌ Module import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

def debug_data_loading():
    """Debug data loading component"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING DATA LOADING")
    print("="*60)
    
    try:
        data_loader = ForexDataLoader('data')
        prices = data_loader.load_unified_prices()
        returns = data_loader.load_unified_returns()
        
        print(f"✅ Data loaded successfully")
        print(f"📊 Prices shape: {prices.shape}")
        print(f"📊 Returns shape: {returns.shape}")
        print(f"📅 Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"💱 Currency pairs: {list(prices.columns)}")
        
        # Check for NaN values
        prices_nan = prices.isna().sum().sum()
        returns_nan = returns.isna().sum().sum()
        print(f"🔍 Prices NaN count: {prices_nan}")
        print(f"🔍 Returns NaN count: {returns_nan}")
        
        return prices, returns, data_loader
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return None, None, None

def debug_signal_generation(prices, returns):
    """Debug signal generation component"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING SIGNAL GENERATION")
    print("="*60)
    
    try:
        # Use small test parameters
        signal_generator = ConrarianSignalGenerator(
            n_worst_performers=3,
            lookback_days=10
        )
        
        # Test with subset of data
        test_prices = prices.iloc[-500:]  # Last 500 days
        test_returns = returns.iloc[-500:]
        
        print(f"📊 Test data shape - Prices: {test_prices.shape}, Returns: {test_returns.shape}")
        
        # Generate signals
        signal_output = signal_generator.generate_signals(test_prices, test_returns)
        
        print(f"✅ Signal generation completed")
        print(f"🔑 Signal output keys: {list(signal_output.keys())}")
        
        for key, value in signal_output.items():
            if key == 'metadata':
                print(f"📋 {key}: {value}")
            else:
                df = value
                print(f"📊 {key} shape: {df.shape}")
                print(f"🔍 {key} data types: {df.dtypes.value_counts().to_dict()}")
                print(f"📈 {key} sample:\n{df.head(2)}")
            print()
        
        return signal_output
        
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}")
        traceback.print_exc()
        return None

def debug_portfolio_manager(signal_output, returns):
    """Debug portfolio manager component with detailed analysis"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING PORTFOLIO MANAGER")
    print("="*60)
    
    try:
        # Test different configurations
        configurations = [
            {'volatility_method': 'rolling', 'risk_parity_method': 'inverse_volatility'},
            {'volatility_method': 'ewma', 'risk_parity_method': 'inverse_volatility'},
            {'volatility_method': 'ewma', 'risk_parity_method': 'erc'}
        ]
        
        for i, config in enumerate(configurations, 1):
            print(f"\n🔧 Testing Configuration {i}: {config}")
            
            try:
                portfolio_manager = PortfolioManager(**config)
                
                # Test with subset of data to isolate the issue
                test_returns = returns.iloc[-100:]  # Last 100 days
                
                # Get a specific date for testing
                test_dates = signal_output['binary_signals'].index[-50:]  # Last 50 dates
                
                for j, test_date in enumerate(test_dates[:3]):  # Test first 3 dates
                    print(f"    📅 Testing date {j+1}: {test_date}")
                    
                    try:
                        # Get signals for this date
                        binary_signals = signal_output['binary_signals'].loc[test_date]
                        selected_assets = binary_signals > 0
                        
                        print(f"    🎯 Selected assets: {selected_assets.sum()}/{len(selected_assets)}")
                        print(f"    📋 Selected: {selected_assets[selected_assets].index.tolist()}")
                        
                        if selected_assets.sum() == 0:
                            print(f"    ⚠️ No assets selected, skipping")
                            continue
                        
                        # Test individual components
                        print(f"    🧪 Testing volatility estimation...")
                        volatility_estimates = portfolio_manager.volatility_estimator.estimate_volatility(
                            test_returns, window=30
                        )
                        print(f"    ✅ Volatility shape: {volatility_estimates.shape}")
                        
                        # Get volatility for current date
                        available_dates = volatility_estimates.index[volatility_estimates.index <= test_date]
                        if len(available_dates) == 0:
                            print(f"    ⚠️ No volatility data for {test_date}, skipping")
                            continue
                        
                        current_vol_date = available_dates[-1]
                        current_volatility = volatility_estimates.loc[current_vol_date]
                        
                        print(f"    📊 Using volatility from {current_vol_date}")
                        print(f"    🔢 Volatility stats: min={current_volatility.min():.4f}, max={current_volatility.max():.4f}")
                        
                        # Test correlation matrix calculation
                        if config['risk_parity_method'] == 'erc':
                            print(f"    🧪 Testing correlation matrix calculation...")
                            correlation_matrix = portfolio_manager._calculate_correlation_matrix(
                                test_returns, test_date, selected_assets
                            )
                            
                            if correlation_matrix is not None:
                                print(f"    ✅ Correlation matrix shape: {correlation_matrix.shape}")
                                print(f"    🔢 Expected shape based on selected assets: {selected_assets.sum()} x {selected_assets.sum()}")
                                
                                # This is where the error likely occurs - let's debug in detail
                                print(f"    🔍 DETAILED DEBUG:")
                                print(f"        Selected assets indices: {np.where(selected_assets)[0]}")
                                print(f"        Selected assets count: {len(selected_assets[selected_assets])}")
                                print(f"        Correlation matrix shape: {correlation_matrix.shape}")
                                
                                # Test the risk parity calculation directly
                                print(f"    🧪 Testing risk parity calculation...")
                                
                                # Filter to selected assets only
                                selected_vol = current_volatility[selected_assets]
                                print(f"        Selected volatility shape: {selected_vol.shape}")
                                
                                if len(selected_vol) != correlation_matrix.shape[0]:
                                    print(f"    ❌ MISMATCH FOUND:")
                                    print(f"        Selected volatility length: {len(selected_vol)}")
                                    print(f"        Correlation matrix size: {correlation_matrix.shape}")
                                    print(f"        This is the source of the IndexError!")
                                    
                                    # Find the root cause
                                    print(f"        Selected assets boolean mask:")
                                    print(f"        {selected_assets}")
                                    print(f"        Assets where True: {selected_assets[selected_assets].index.tolist()}")
                                    print(f"        Available in returns: {test_returns.columns.tolist()}")
                                    print(f"        Available in volatility: {current_volatility.index.tolist()}")
                                    
                                    break
                            else:
                                print(f"    ⚠️ Correlation matrix is None")
                        
                        # Test weight calculation
                        print(f"    🧪 Testing weight calculation...")
                        weights = portfolio_manager.construct_portfolio_weights(
                            signal_output, test_returns, current_date=test_date
                        )
                        print(f"    ✅ Weight calculation successful: {weights.shape}")
                        print(f"    📊 Non-zero weights: {(weights > 0.001).sum()}")
                        
                    except Exception as date_e:
                        print(f"    ❌ Error with date {test_date}: {date_e}")
                        traceback.print_exc()
                        return False  # Stop on first error for debugging
                        
                print(f"✅ Configuration {i} completed successfully")
                
            except Exception as config_e:
                print(f"❌ Configuration {i} failed: {config_e}")
                traceback.print_exc()
                continue
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Portfolio manager debugging failed: {e}")
        traceback.print_exc()
        return False

def debug_integration(signal_output, returns):
    """Debug full integration"""
    print("\n" + "="*60)
    print("🔍 DEBUGGING FULL INTEGRATION")
    print("="*60)
    
    try:
        # Use safe configuration
        portfolio_manager = PortfolioManager(
            volatility_method='rolling',
            risk_parity_method='inverse_volatility'
        )
        
        # Test with very small dataset
        test_returns = returns.iloc[-50:]
        
        print(f"📊 Test returns shape: {test_returns.shape}")
        
        # Run portfolio management
        portfolio_results = portfolio_manager.run_portfolio_management(
            signal_output, test_returns
        )
        
        print(f"✅ Integration test successful")
        print(f"🔑 Portfolio results keys: {list(portfolio_results.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    print("🚀 COMPREHENSIVE FOREX BACKTESTING SYSTEM DEBUG")
    print("=" * 80)
    
    debug_results = {}
    
    # 1. Test data loading
    prices, returns, data_loader = debug_data_loading()
    debug_results['data_loading'] = prices is not None
    
    if not debug_results['data_loading']:
        print("❌ Cannot proceed without data")
        return
    
    # 2. Test signal generation
    signal_output = debug_signal_generation(prices, returns)
    debug_results['signal_generation'] = signal_output is not None
    
    if not debug_results['signal_generation']:
        print("❌ Cannot proceed without signals")
        return
    
    # 3. Test portfolio manager (this is where the error occurs)
    debug_results['portfolio_manager'] = debug_portfolio_manager(signal_output, returns)
    
    # 4. Test integration (only if portfolio manager works)
    if debug_results['portfolio_manager']:
        debug_results['integration'] = debug_integration(signal_output, returns)
    else:
        debug_results['integration'] = False
    
    # Summary
    print("\n" + "="*80)
    print("🔍 DEBUG SUMMARY")
    print("="*80)
    
    for component, success in debug_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{component.replace('_', ' ').title():<25}: {status}")
    
    if all(debug_results.values()):
        print("\n🎉 All components working correctly!")
    else:
        failed_components = [k for k, v in debug_results.items() if not v]
        print(f"\n⚠️ Issues found in: {', '.join(failed_components)}")
    
    print("="*80)

if __name__ == "__main__":
    main()