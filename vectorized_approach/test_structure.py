#!/usr/bin/env python3
"""
Test script per verificare che la nuova struttura organizzata funzioni correttamente
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test che gli import funzionino"""
    print("🧪 Testing imports...")
    
    try:
        # Test shared utilities
        sys.path.append('shared')
        from utils import clean_data, load_individual_results, setup_matplotlib
        print("✅ Shared utilities import OK")
    except ImportError as e:
        print(f"❌ Shared utilities import FAILED: {e}")
        return False
    
    try:
        # Test modules
        sys.path.append('modules')
        from strategy_contrarian import strategy, rebalance_risk_parity
        print("✅ Strategy module import OK")
    except ImportError as e:
        print(f"❌ Strategy module import FAILED: {e}")
        return False
    
    return True

def test_file_structure():
    """Test che la struttura file sia corretta"""
    print("🧪 Testing file structure...")
    
    expected_structure = {
        'modules/': [
            'strategy_contrarian.py',
            'forex_backtest.py', 
            'commodities_backtest.py',
            'daily_operations_analyzer.py'
        ],
        'forex/notebooks/': [
            'fx_main_educational.ipynb',
            'fx_analysis.ipynb',
            'fx_lookback_analysis.ipynb',
            'fx_daily_operations.ipynb'
        ],
        'forex/data/raw/': [],  # Should have parquet files
        'forex/data/results/': [],  # Should have results files
        'commodities/notebooks/': [
            'commodities_analysis.ipynb',
            'commodities_lookback_analysis.ipynb'
        ],
        'commodities/data/raw/': [],  # Should have parquet files
        'commodities/data/results/': [],  # Should have results files
        'shared/': ['utils.py']
    }
    
    all_good = True
    for folder, expected_files in expected_structure.items():
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"❌ Missing folder: {folder}")
            all_good = False
            continue
            
        if expected_files:  # Only check specific files if listed
            for expected_file in expected_files:
                file_path = folder_path / expected_file
                if not file_path.exists():
                    print(f"❌ Missing file: {file_path}")
                    all_good = False
                else:
                    print(f"✅ Found: {file_path}")
        else:
            # Check that directory has some files
            files = list(folder_path.glob('*'))
            if files:
                print(f"✅ Folder {folder} has {len(files)} files")
            else:
                print(f"⚠️  Folder {folder} is empty (may be expected)")
    
    return all_good

def test_data_accessibility():
    """Test che i dati siano accessibili"""
    print("🧪 Testing data accessibility...")
    
    try:
        import pandas as pd
        
        # Test forex data
        forex_equity = Path('forex/data/results/all_equity_curves.parquet')
        if forex_equity.exists():
            df = pd.read_parquet(forex_equity)
            print(f"✅ Forex equity curves loaded: {df.shape}")
        else:
            print("❌ Forex equity curves not found")
            return False
        
        # Test commodities data
        commodities_equity = Path('commodities/data/results/all_equity_curves.parquet')
        if commodities_equity.exists():
            df = pd.read_parquet(commodities_equity)
            print(f"✅ Commodities equity curves loaded: {df.shape}")
        else:
            print("❌ Commodities equity curves not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data accessibility test FAILED: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing new vectorized_approach structure...")
    print("=" * 50)
    
    # Change to vectorized_approach directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Data Accessibility", test_data_accessibility)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY:")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! New structure is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)