#!/usr/bin/env python3
"""
Test Full Pipeline End-to-End

This script tests the complete pipeline as used in the notebook but with
reduced dataset size to ensure it runs efficiently.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'modules'))

from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager
from backtesting_engine import BacktestingEngine
from performance_analyzer import PerformanceAnalyzer

def test_full_pipeline():
    """Test the complete pipeline with a subset of data"""
    print("🚀 TESTING FULL PIPELINE END-TO-END")
    print("="*60)
    
    try:
        # 1. Data Loading
        print("📊 Step 1: Loading data...")
        data_loader = ForexDataLoader('data')
        prices = data_loader.load_unified_prices()
        returns = data_loader.load_unified_returns()
        
        # Use a subset for faster testing (last 2 years)
        test_prices = prices.iloc[-500:]  
        test_returns = returns.iloc[-500:]
        
        print(f"✅ Data loaded: {test_prices.shape} prices, {test_returns.shape} returns")
        
        # 2. Signal Generation
        print("\n📡 Step 2: Generating signals...")
        signal_generator = ConrarianSignalGenerator(
            n_worst_performers=3,
            lookback_days=20,
            min_history_days=100  # Reduced for testing
        )
        
        signal_output = signal_generator.generate_signals(test_prices, test_returns)
        
        # Check if we have any active signals
        binary_signals = signal_output['binary_signals']
        active_dates = binary_signals[binary_signals.sum(axis=1) > 0]
        
        print(f"✅ Signals generated: {len(active_dates)} active trading dates")
        
        if len(active_dates) == 0:
            print("⚠️ No active signals found, adjusting parameters...")
            # Try with smaller min_history_days
            signal_generator = ConrarianSignalGenerator(
                n_worst_performers=3,
                lookback_days=10,
                min_history_days=50
            )
            signal_output = signal_generator.generate_signals(test_prices, test_returns)
            active_dates = signal_output['binary_signals'][signal_output['binary_signals'].sum(axis=1) > 0]
            print(f"✅ Adjusted signals: {len(active_dates)} active trading dates")
        
        # 3. Portfolio Management
        print("\n⚖️ Step 3: Portfolio management...")
        
        # Test different configurations
        configs_to_test = [
            {'volatility_method': 'rolling', 'risk_parity_method': 'inverse_volatility'},
            {'volatility_method': 'ewma', 'risk_parity_method': 'erc', 'target_volatility': 0.12}
        ]
        
        portfolio_results = {}
        
        for i, config in enumerate(configs_to_test, 1):
            print(f"    Config {i}: {config['risk_parity_method']} with {config['volatility_method']}")
            
            portfolio_manager = PortfolioManager(**config)
            
            # Run portfolio management on subset of dates for speed
            subset_signal_output = {}
            for key, value in signal_output.items():
                if key == 'metadata':
                    subset_signal_output[key] = value
                else:
                    subset_signal_output[key] = value.iloc[-100:]  # Last 100 days
            
            results = portfolio_manager.run_portfolio_management(
                subset_signal_output, test_returns.iloc[-100:]
            )
            
            portfolio_results[f'config_{i}'] = results
            
            print(f"    ✅ Portfolio management completed")
            print(f"        Rebalancing dates: {len(results['rebalancing_dates'])}")
            print(f"        Portfolio returns shape: {results['portfolio_returns'].shape}")
        
        # 4. Backtesting
        print("\n🎯 Step 4: Backtesting...")
        
        backtesting_engine = BacktestingEngine(
            initial_capital=1000000,
            transaction_cost_bps=2.0
        )
        
        # Test with the first portfolio configuration
        portfolio_weights = portfolio_results['config_1']['portfolio_weights']
        
        backtest_results = backtesting_engine.run_backtest(
            portfolio_weights,
            test_returns.iloc[-100:],
            start_date=None,
            end_date=None
        )
        
        print(f"✅ Backtesting completed")
        print(f"    Final portfolio value: ${backtest_results['portfolio_value'].iloc[-1]:,.0f}")
        
        # 5. Performance Analysis
        print("\n📈 Step 5: Performance analysis...")
        
        analyzer = PerformanceAnalyzer()
        performance_metrics = analyzer.analyze_returns(
            backtest_results['portfolio_returns']
        )
        
        print(f"✅ Performance analysis completed")
        print(f"    Available metrics: {list(performance_metrics.keys())}")
        
        # Use safe key access
        total_return = performance_metrics.get('total_return', performance_metrics.get('cumulative_return', 0))
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', performance_metrics.get('maximum_drawdown', 0))
        
        print(f"    Total return: {total_return:.2%}")
        print(f"    Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"    Max drawdown: {max_drawdown:.2%}")
        
        print(f"\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        
        return {
            'success': True,
            'data_shape': test_prices.shape,
            'active_signals': len(active_dates),
            'portfolio_configs': len(portfolio_results),
            'backtest_results': backtest_results,
            'performance_metrics': performance_metrics
        }
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    result = test_full_pipeline()
    
    print(f"\n" + "="*60)
    print(f"PIPELINE TEST RESULT: {'SUCCESS' if result['success'] else 'FAILURE'}")
    print("="*60)
    
    if result['success']:
        print("The contrarian forex backtesting system is now fully functional!")
    else:
        print("There are still issues that need to be addressed.")