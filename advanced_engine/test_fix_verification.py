#!/usr/bin/env python3
"""
Test that the IndexError fix is working correctly
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'modules'))

from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager

def test_fix():
    """Test that the IndexError is fixed"""
    print("🔧 TESTING INDEXERROR FIX")
    print("="*50)
    
    try:
        # Load data
        data_loader = ForexDataLoader('data')
        prices = data_loader.load_unified_prices()
        returns = data_loader.load_unified_returns()
        
        # Generate signals with parameters that caused the error
        signal_generator = ConrarianSignalGenerator(n_worst_performers=3, lookback_days=30)
        signal_output = signal_generator.generate_signals(prices, returns)
        
        # Initialize portfolio manager with the problematic ERC method
        portfolio_manager = PortfolioManager(
            volatility_method='ewma',
            risk_parity_method='erc',
            target_volatility=0.12,
            max_position_size=0.3
        )
        
        # Test with a specific date that we know has signals
        binary_signals = signal_output['binary_signals']
        signal_dates = binary_signals[binary_signals.sum(axis=1) > 0].index
        
        if len(signal_dates) == 0:
            print("❌ No signal dates found")
            return False
        
        test_date = signal_dates[100]  # Use a date well into the data
        print(f"📅 Testing with date: {test_date}")
        
        # This should work without IndexError
        weights = portfolio_manager.construct_portfolio_weights(
            signal_output, returns, current_date=test_date
        )
        
        print(f"✅ SUCCESS! Portfolio weights calculated without error")
        print(f"📊 Number of positions: {(weights > 0.001).sum()}")
        print(f"📈 Portfolio weights sum: {weights.sum():.6f}")
        print(f"🎯 Non-zero weights:")
        non_zero = weights[weights > 0.001]
        for asset, weight in non_zero.items():
            print(f"    {asset}: {weight:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR STILL EXISTS: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_configurations():
    """Test multiple portfolio manager configurations"""
    print(f"\n🔧 TESTING MULTIPLE CONFIGURATIONS")
    print("="*50)
    
    configurations = [
        {'volatility_method': 'rolling', 'risk_parity_method': 'inverse_volatility'},
        {'volatility_method': 'ewma', 'risk_parity_method': 'inverse_volatility'}, 
        {'volatility_method': 'ewma', 'risk_parity_method': 'erc'},
        {'volatility_method': 'rolling', 'risk_parity_method': 'erc'}
    ]
    
    data_loader = ForexDataLoader('data')
    prices = data_loader.load_unified_prices()
    returns = data_loader.load_unified_returns()
    
    signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=20)
    signal_output = signal_generator.generate_signals(prices, returns)
    
    binary_signals = signal_output['binary_signals']
    signal_dates = binary_signals[binary_signals.sum(axis=1) > 0].index
    test_date = signal_dates[200]  # Use a date with sufficient history
    
    results = {}
    
    for i, config in enumerate(configurations, 1):
        print(f"\n🧪 Configuration {i}: {config}")
        
        try:
            portfolio_manager = PortfolioManager(**config)
            weights = portfolio_manager.construct_portfolio_weights(
                signal_output, returns, current_date=test_date
            )
            
            results[f"Config_{i}"] = {
                'success': True,
                'num_positions': (weights > 0.001).sum(),
                'weights_sum': weights.sum(),
                'config': config
            }
            
            print(f"    ✅ SUCCESS - {(weights > 0.001).sum()} positions")
            
        except Exception as e:
            results[f"Config_{i}"] = {
                'success': False,
                'error': str(e),
                'config': config
            }
            print(f"    ❌ FAILED: {type(e).__name__}: {e}")
    
    return results

if __name__ == "__main__":
    print("🚀 PORTFOLIO MANAGER FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Basic fix verification
    success1 = test_fix()
    
    # Test 2: Multiple configurations
    results = test_multiple_configurations()
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("="*30)
    print(f"Basic fix test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    
    successful_configs = sum(1 for r in results.values() if r['success'])
    total_configs = len(results)
    print(f"Configuration tests: {successful_configs}/{total_configs} passed")
    
    if success1 and successful_configs == total_configs:
        print(f"\n🎉 ALL TESTS PASSED - IndexError is fixed!")
    else:
        print(f"\n⚠️ Some tests failed - more work needed")
        
    print("="*60)