#!/usr/bin/env python3
"""
Example usage of the Advanced Contrarian Forex Trading System

This script demonstrates how to use the system for:
1. Basic parameter optimization
2. Custom strategy implementation  
3. Performance analysis and visualization

Run this script to see the system in action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the system modules
from modules.data_loader import ForexDataLoader
from modules.signal_generator import ConrarianSignalGenerator
from modules.portfolio_manager import PortfolioManager
from modules.backtesting_engine import BacktestingEngine
from modules.performance_analyzer import PerformanceAnalyzer
from modules.parameter_optimizer import ParameterOptimizer

def main():
    """
    Main example workflow demonstrating system usage
    """
    print("🚀 Advanced Contrarian Forex Trading System - Example Usage")
    print("=" * 70)
    
    # 1. Load Data
    print("\n📊 Step 1: Loading forex data...")
    data_loader = ForexDataLoader('data')
    prices = data_loader.load_unified_prices()
    returns = data_loader.load_unified_returns()
    
    if prices is None or returns is None:
        print("❌ Error: Could not load data. Please ensure data files are present.")
        return
    
    print(f"✅ Loaded {len(prices.columns)} currency pairs, {len(prices)} trading days")
    print(f"   Data period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # 2. Quick Parameter Test
    print("\n🎯 Step 2: Quick parameter optimization (reduced for speed)...")
    optimizer = ParameterOptimizer(optimization_metric='sharpe_ratio')
    
    # Small parameter grid for demonstration
    parameter_grid = {
        'n_worst_performers': [3, 5],     # Test 3 and 5 worst performers
        'lookback_days': [10, 20]         # Test 10 and 20 day lookback
    }
    
    try:
        optimization_results = optimizer.grid_search_optimization(
            data_loader=data_loader,
            signal_generator_class=ConrarianSignalGenerator,
            backtesting_engine_class=BacktestingEngine,
            parameter_grid=parameter_grid,
            start_date="2020-01-01",       # Recent period for speed
            end_date="2023-12-31"
        )
        
        best_results = optimization_results['all_results']
        if 'val_sharpe_ratio' in best_results.columns:
            best_params = best_results.loc[best_results['val_sharpe_ratio'].idxmax()]
            print(f"✅ Best parameters found:")
            print(f"   N (worst performers): {best_params.get('n_worst_performers', 'N/A')}")
            print(f"   M (lookback days): {best_params.get('lookback_days', 'N/A')}")
            print(f"   Validation Sharpe: {best_params.get('val_sharpe_ratio', 0):.3f}")
        else:
            print("✅ Optimization completed (detailed results in optimization_results)")
            
    except Exception as e:
        print(f"⚠️ Optimization encountered an issue: {str(e)}")
        print("   Proceeding with default parameters...")
    
    # 3. Custom Strategy Implementation
    print("\n⚖️ Step 3: Running custom strategy with default parameters...")
    
    # Use default parameters for demonstration
    signal_generator = ConrarianSignalGenerator(
        n_worst_performers=5, 
        lookback_days=20
    )
    
    # Generate signals
    print("   Generating contrarian signals...")
    signal_output = signal_generator.generate_signals(prices, returns)
    
    # Apply portfolio management
    print("   Applying risk parity portfolio management...")
    portfolio_manager = PortfolioManager(
        risk_parity_method='erc',          # Equal Risk Contribution
        target_volatility=0.12,           # 12% target volatility
        max_position_size=0.25             # 25% max per position
    )
    
    portfolio_results = portfolio_manager.run_portfolio_management(signal_output, returns)
    
    # Run backtest
    print("   Executing backtest...")
    backtesting_engine = BacktestingEngine(
        transaction_cost_bps=5.0,          # 5 bps transaction costs
        initial_capital=1000000            # $1M initial capital
    )
    
    backtest_results = backtesting_engine.run_backtest(
        portfolio_results['portfolio_weights'], 
        returns,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    # 4. Performance Analysis
    print("\n📈 Step 4: Performance analysis...")
    analyzer = PerformanceAnalyzer()
    portfolio_metrics = analyzer.calculate_comprehensive_metrics(
        backtest_results['portfolio_returns']
    )
    
    # Display key results
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Total Return:           {portfolio_metrics.get('total_return', 0):.1%}")
    print(f"   Annualized Return:      {portfolio_metrics.get('annual_return', 0):.1%}")
    print(f"   Annualized Volatility:  {portfolio_metrics.get('annual_volatility', 0):.1%}")
    print(f"   Sharpe Ratio:           {portfolio_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Maximum Drawdown:       {portfolio_metrics.get('max_drawdown', 0):.1%}")
    print(f"   Calmar Ratio:           {portfolio_metrics.get('calmar_ratio', 0):.3f}")
    print(f"   Win Rate:               {portfolio_metrics.get('win_rate', 0):.1%}")
    
    # 5. Simple Visualization
    print("\n📊 Step 5: Creating equity curve visualization...")
    try:
        plt.figure(figsize=(12, 6))
        portfolio_values = backtest_results['portfolio_values']
        dates = backtest_results['portfolio_returns'].index
        
        plt.plot(dates, portfolio_values, linewidth=2, color='darkblue', label='Portfolio Value')
        plt.title('Contrarian Forex Strategy - Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add performance annotation
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value / initial_value - 1) * 100
        plt.text(0.02, 0.98, 
                f'Total Return: {total_return:.1f}%\nSharpe: {portfolio_metrics.get("sharpe_ratio", 0):.3f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        print("✅ Visualization created successfully!")
        
    except Exception as e:
        print(f"⚠️ Visualization issue: {str(e)}")
    
    print(f"\n🎉 Example completed successfully!")
    print(f"   Final Portfolio Value: ${backtest_results['portfolio_values'][-1]:,.0f}")
    print(f"   Strategy demonstrated over {len(backtest_results['portfolio_returns'])} trading days")
    print(f"\n📝 Next steps:")
    print(f"   1. Run the notebooks for comprehensive analysis")
    print(f"   2. Experiment with different parameters")
    print(f"   3. Extend the analysis period for full historical testing")

if __name__ == "__main__":
    main()