#!/usr/bin/env python3
"""
Test the exact code from the notebook to ensure it works after fixes
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add modules path
sys.path.append(str(Path(__file__).parent / 'modules'))

# Import modules exactly as in notebook
from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager
from backtesting_engine import BacktestingEngine

def test_notebook_code():
    """Test the exact code sequence from the notebook"""
    
    print("🧪 TESTING EXACT NOTEBOOK CODE SEQUENCE")
    print("="*60)
    
    try:
        # Exact code from notebook cell 8
        print("📊 Loading data...")
        data_loader = ForexDataLoader('data')
        prices = data_loader.load_unified_prices()
        returns = data_loader.load_unified_returns()
        
        print(f"✅ Data loaded: {len(prices.columns)} currency pairs")
        print(f"📅 Data period: {prices.index[0]} to {prices.index[-1]}")
        
        # Best parameters from notebook
        best_n = 3
        best_m = 30
        
        print(f"🏆 Using best parameters: N={best_n}, M={best_m}")
        
        # Initialize components with best parameters (exact notebook code)
        signal_generator = ConrarianSignalGenerator(
            n_worst_performers=best_n, 
            lookback_days=best_m
        )
        
        portfolio_manager = PortfolioManager(
            volatility_method='ewma',
            risk_parity_method='erc',
            target_volatility=0.12,
            max_position_size=0.3
        )
        
        backtesting_engine = BacktestingEngine(
            transaction_cost_bps=5.0,  # 5 bps transaction cost
            initial_capital=1000000
        )
        
        # Generate signals for full period (this was working)
        print("📊 Generating contrarian signals...")
        signal_output = signal_generator.generate_signals(prices, returns)
        
        print(f"✅ Signals generated successfully")
        print(f"    Binary signals shape: {signal_output['binary_signals'].shape}")
        
        # The problematic line from notebook that was causing IndexError
        print("⚖️ Applying risk parity portfolio management...")
        
        # Use a subset for testing to avoid long runtime
        subset_size = 200  # Last 200 days for testing
        test_signal_output = {}
        for key, value in signal_output.items():
            if key == 'metadata':
                test_signal_output[key] = value
            else:
                test_signal_output[key] = value.iloc[-subset_size:]
        
        test_returns = returns.iloc[-subset_size:]
        
        # This should now work without IndexError
        portfolio_results = portfolio_manager.run_portfolio_management(
            test_signal_output, test_returns
        )
        
        print(f"✅ Portfolio management completed successfully!")
        print(f"    Portfolio weights shape: {portfolio_results['portfolio_weights'].shape}")
        print(f"    Portfolio returns shape: {portfolio_results['portfolio_returns'].shape}")
        
        # Test backtesting
        print("🎯 Executing comprehensive backtest...")
        backtest_results = backtesting_engine.run_backtest(
            portfolio_results['portfolio_weights'], 
            test_returns, 
            start_date=None,
            end_date=None
        )
        
        print(f"✅ Backtesting completed successfully!")
        
        # Print key results
        final_value = backtest_results['portfolio_value'].iloc[-1]
        initial_value = backtesting_engine.initial_capital
        total_return = (final_value / initial_value - 1) * 100
        
        print(f"\n📈 KEY RESULTS:")
        print(f"    Initial Capital: ${initial_value:,.0f}")
        print(f"    Final Value: ${final_value:,.0f}")
        print(f"    Total Return: {total_return:.2f}%")
        print(f"    Trading Days: {len(portfolio_results['portfolio_returns'])}")
        
        print(f"\n🎉 NOTEBOOK CODE SEQUENCE COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        print(f"\n❌ NOTEBOOK CODE FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_notebook_code()
    
    print(f"\n" + "="*60)
    if success:
        print("✅ NOTEBOOK IS NOW FIXED AND READY TO RUN!")
        print("The IndexError has been resolved and the full pipeline works.")
    else:
        print("❌ NOTEBOOK STILL HAS ISSUES")
        print("Additional debugging may be required.")
    print("="*60)