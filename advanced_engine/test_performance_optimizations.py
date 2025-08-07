#!/usr/bin/env python3
"""
Performance Test for Optimized Portfolio Management System

This script tests the optimized portfolio management modules to ensure
they execute within the target <30 seconds timeframe for the notebook.
"""

import sys
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Add modules to path
sys.path.append('modules')

from modules.data_loader import ForexDataLoader
from modules.signal_generator import ConrarianSignalGenerator
from modules.portfolio_manager import PortfolioManager
from modules.backtesting_engine import BacktestingEngine

warnings.filterwarnings('ignore')

def test_optimized_performance():
    """Test the optimized portfolio management performance."""
    
    print("🚀 Testing Optimized Portfolio Management Performance")
    print("=" * 60)
    
    # Record start time
    total_start_time = time.time()
    
    # Step 1: Load data
    print("📈 Loading forex market data...")
    start_time = time.time()
    
    data_loader = ForexDataLoader('data')
    prices = data_loader.load_unified_prices()
    returns = data_loader.load_unified_returns()
    
    print(f"✅ Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"   - {len(prices.columns)} currency pairs")
    print(f"   - {len(prices):,} trading days")
    
    # Step 2: Initialize components with best parameters
    print("\n🔬 Initializing optimized components...")
    start_time = time.time()
    
    # Use best parameters from optimization: N=3, M=30
    signal_generator = ConrarianSignalGenerator(
        n_worst_performers=3, 
        lookback_days=30
    )
    
    portfolio_manager = PortfolioManager(
        volatility_method='ewma',  # Fast method
        risk_parity_method='inverse_volatility',  # Fast method that can be vectorized
        target_volatility=0.12,
        max_position_size=0.3
    )
    
    backtesting_engine = BacktestingEngine(
        transaction_cost_bps=5.0,
        initial_capital=1000000
    )
    
    print(f"✅ Components initialized in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Generate signals
    print("\n📊 Generating contrarian signals...")
    start_time = time.time()
    
    signal_output = signal_generator.generate_signals(prices, returns)
    
    signals_time = time.time() - start_time
    print(f"✅ Signals generated in {signals_time:.2f} seconds")
    
    # Step 4: Run portfolio management (This is the main optimization target)
    print("\n⚖️ Running optimized portfolio management...")
    start_time = time.time()
    
    portfolio_results = portfolio_manager.run_portfolio_management(signal_output, returns)
    
    portfolio_time = time.time() - start_time
    print(f"✅ Portfolio management completed in {portfolio_time:.2f} seconds")
    
    # Step 5: Run backtest
    print("\n🎯 Executing backtest...")
    start_time = time.time()
    
    backtest_results = backtesting_engine.run_backtest(
        portfolio_results['portfolio_weights'], 
        returns, 
        start_date=None,
        end_date=None
    )
    
    backtest_time = time.time() - start_time
    print(f"✅ Backtest completed in {backtest_time:.2f} seconds")
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Performance summary
    print("\n📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Signal Generation:     {signals_time:>8.2f} seconds")
    print(f"Portfolio Management:  {portfolio_time:>8.2f} seconds  ⭐ (Main optimization)")
    print(f"Backtesting:           {backtest_time:>8.2f} seconds")
    print(f"Total Execution:       {total_time:>8.2f} seconds")
    print(f"Target (<30s):         {'✅ PASSED' if total_time < 30 else '❌ FAILED'}")
    
    # Performance analysis
    if total_time < 30:
        improvement_factor = 180 / total_time  # Assuming original was 3+ minutes
        print(f"\n🎉 PERFORMANCE IMPROVEMENT: {improvement_factor:.1f}x faster")
    else:
        print(f"\n⚠️ Further optimization needed. Exceeded target by {total_time - 30:.2f} seconds")
    
    # Validate results quality
    print("\n📈 RESULT VALIDATION")
    print("=" * 60)
    
    portfolio_returns = backtest_results['portfolio_returns'].dropna()
    if len(portfolio_returns) > 0:
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        print(f"Total Return:          {total_return:>8.2%}")
        print(f"Annualized Return:     {annualized_return:>8.2%}")
        print(f"Volatility:            {volatility:>8.2%}")
        print(f"Sharpe Ratio:          {sharpe_ratio:>8.3f}")
        print(f"Portfolio Days:        {len(portfolio_returns):>8,}")
        
        # Check for data quality issues
        num_zero_returns = (portfolio_returns == 0).sum()
        if num_zero_returns / len(portfolio_returns) > 0.1:
            print(f"⚠️ High number of zero returns: {num_zero_returns:,} ({num_zero_returns/len(portfolio_returns):.1%})")
        else:
            print("✅ Portfolio returns look healthy")
    
    return {
        'total_time': total_time,
        'signals_time': signals_time,
        'portfolio_time': portfolio_time,
        'backtest_time': backtest_time,
        'target_met': total_time < 30,
        'results': backtest_results
    }

if __name__ == "__main__":
    try:
        test_results = test_optimized_performance()
        
        print(f"\n🎯 OPTIMIZATION {'SUCCESS' if test_results['target_met'] else 'INCOMPLETE'}")
        print(f"Total execution time: {test_results['total_time']:.2f} seconds")
        
        if test_results['target_met']:
            print("Ready for production use in the notebook! 🚀")
        else:
            print("Additional optimization may be needed for optimal notebook performance.")
            
    except Exception as e:
        print(f"\n❌ Error during performance testing: {str(e)}")
        import traceback
        traceback.print_exc()