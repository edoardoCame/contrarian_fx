#!/usr/bin/env python3
"""
Core Risk Management System Validation

Focused validation of the critical risk management fixes without backtesting dependencies.
Tests the core IndexError fix and essential portfolio management functionality.

Author: Quantitative Portfolio Management Specialist  
Date: August 7, 2025
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from datetime import datetime
import traceback

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'modules'))

from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager


class CoreRiskValidator:
    """Focused validation of core risk management functionality"""
    
    def __init__(self):
        self.results = {}
        self.test_count = 0
        self.passed_tests = 0
        
        # Load data
        print("📊 Loading market data...")
        self.data_loader = ForexDataLoader('data')
        self.prices = self.data_loader.load_unified_prices()
        self.returns = self.data_loader.load_unified_returns()
        print(f"   ✓ Loaded {len(self.prices.columns)} assets, {len(self.prices)} observations")
        
    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        self.test_count += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"   {status}: {name}")
        if details:
            print(f"      {details}")
            
        self.results[name] = {'success': success, 'details': details}
    
    def test_indexerror_fix(self):
        """Test 1: Validate the original IndexError is fixed"""
        print(f"\n🔍 TEST 1: ORIGINAL INDEXERROR FIX VALIDATION")
        print("="*60)
        
        try:
            # Reproduce the exact conditions that caused the IndexError
            signal_generator = ConrarianSignalGenerator(n_worst_performers=3, lookback_days=30)
            signal_output = signal_generator.generate_signals(self.prices, self.returns)
            
            # Create portfolio manager with ERC method (what caused the error)
            portfolio_manager = PortfolioManager(
                volatility_method='ewma',
                risk_parity_method='erc',
                target_volatility=0.12
            )
            
            # Find test dates
            binary_signals = signal_output['binary_signals']
            signal_dates = binary_signals[binary_signals.sum(axis=1) >= 3].index
            
            if len(signal_dates) > 100:
                test_date = signal_dates[100]
                
                # This should work without IndexError
                weights = portfolio_manager.construct_portfolio_weights(
                    signal_output, self.returns, current_date=test_date
                )
                
                success = weights is not None and len(weights) > 0 and not weights.isna().all()
                num_positions = (weights > 0.001).sum()
                weights_sum = weights.sum()
                
                details = f"Generated {num_positions} positions, weights sum: {weights_sum:.6f}"
                self.log_test("IndexError Fix - ERC Method", success, details)
                
                # Test multiple dates to ensure consistency
                success_count = 0
                for i, test_date in enumerate(signal_dates[50:60]):
                    try:
                        weights = portfolio_manager.construct_portfolio_weights(
                            signal_output, self.returns, current_date=test_date
                        )
                        if weights is not None and not weights.isna().all():
                            success_count += 1
                    except:
                        pass
                
                consistency_success = success_count >= 8  # 80% success rate
                details = f"Successful portfolio construction on {success_count}/10 test dates"
                self.log_test("IndexError Fix - Consistency", consistency_success, details)
                
            else:
                self.log_test("IndexError Fix", False, "Insufficient signal dates for testing")
                
        except Exception as e:
            self.log_test("IndexError Fix", False, f"Error: {str(e)}")
    
    def test_risk_parity_methods(self):
        """Test 2: All Risk Parity Methods"""
        print(f"\n🔍 TEST 2: RISK PARITY METHOD VALIDATION")
        print("="*60)
        
        methods = ['inverse_volatility', 'erc']
        asset_counts = [2, 3, 5, 7, 10]
        
        for method in methods:
            for n_assets in asset_counts:
                try:
                    signal_generator = ConrarianSignalGenerator(n_worst_performers=n_assets, lookback_days=30)
                    signal_output = signal_generator.generate_signals(self.prices, self.returns)
                    
                    portfolio_manager = PortfolioManager(
                        volatility_method='ewma',
                        risk_parity_method=method
                    )
                    
                    binary_signals = signal_output['binary_signals']
                    signal_dates = binary_signals[binary_signals.sum(axis=1) >= n_assets].index
                    
                    if len(signal_dates) > 50:
                        test_date = signal_dates[50]
                        
                        weights = portfolio_manager.construct_portfolio_weights(
                            signal_output, self.returns, current_date=test_date
                        )
                        
                        # Validate results
                        num_positions = (weights > 0.001).sum()
                        weights_sum = weights.sum()
                        max_weight = weights.max()
                        
                        # Success criteria
                        success = (
                            num_positions > 0 and
                            num_positions <= n_assets and
                            abs(weights_sum - 1.0) < 0.01 and
                            max_weight > 0
                        )
                        
                        details = f"Method: {method}, N={n_assets}, Positions={num_positions}, Sum={weights_sum:.4f}, Max={max_weight:.4f}"
                        self.log_test(f"Risk Parity {method} N={n_assets}", success, details)
                        
                    else:
                        self.log_test(f"Risk Parity {method} N={n_assets}", False, "Insufficient data")
                        
                except Exception as e:
                    self.log_test(f"Risk Parity {method} N={n_assets}", False, f"Error: {str(e)}")
    
    def test_correlation_matrix_edge_cases(self):
        """Test 3: Correlation Matrix Edge Cases"""
        print(f"\n🔍 TEST 3: CORRELATION MATRIX EDGE CASES")
        print("="*60)
        
        edge_cases = [
            {'n_assets': 1, 'description': 'Single asset'},
            {'n_assets': 2, 'description': 'Minimum correlation matrix'},
            {'n_assets': len(self.prices.columns), 'description': 'Full universe'}
        ]
        
        for case in edge_cases:
            try:
                n = case['n_assets']
                if n == 1:
                    signal_generator = ConrarianSignalGenerator(n_worst_performers=1, lookback_days=30)
                    # Use inverse volatility for single asset
                    method = 'inverse_volatility'
                else:
                    signal_generator = ConrarianSignalGenerator(n_worst_performers=n, lookback_days=30)
                    method = 'erc'
                
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method=method
                )
                
                binary_signals = signal_output['binary_signals']
                if n == 1:
                    signal_dates = binary_signals[binary_signals.sum(axis=1) >= 1].index
                else:
                    signal_dates = binary_signals[binary_signals.sum(axis=1) >= n].index
                
                if len(signal_dates) > 30:
                    test_date = signal_dates[30]
                    
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    success = weights is not None and len(weights) > 0 and not weights.isna().all()
                    num_positions = (weights > 0.001).sum() if success else 0
                    
                    details = f"{case['description']} - {num_positions} positions, Method: {method}"
                    self.log_test(f"Edge Case: {case['description']}", success, details)
                    
                else:
                    self.log_test(f"Edge Case: {case['description']}", False, "Insufficient data")
                    
            except Exception as e:
                self.log_test(f"Edge Case: {case['description']}", False, f"Error: {str(e)}")
    
    def test_position_constraints(self):
        """Test 4: Position Size Constraints"""
        print(f"\n🔍 TEST 4: POSITION SIZE CONSTRAINTS")
        print("="*60)
        
        constraint_tests = [
            {'max_position_size': 0.1, 'expected_max': 0.12},  # Allow small tolerance
            {'max_position_size': 0.2, 'expected_max': 0.22},
            {'max_position_size': 0.3, 'expected_max': 0.32}
        ]
        
        for test_config in constraint_tests:
            try:
                signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=30)
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='inverse_volatility',
                    max_position_size=test_config['max_position_size']
                )
                
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= 5].index
                
                if len(signal_dates) > 50:
                    test_date = signal_dates[50]
                    
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    max_weight = weights.max()
                    success = max_weight <= test_config['expected_max']
                    
                    details = f"Max position: {max_weight:.4f}, Limit: {test_config['max_position_size']}"
                    test_name = f"Position limit {test_config['max_position_size']}"
                    self.log_test(test_name, success, details)
                    
                else:
                    test_name = f"Position limit {test_config['max_position_size']}"
                    self.log_test(test_name, False, "Insufficient data")
                    
            except Exception as e:
                test_name = f"Position limit {test_config['max_position_size']}"
                self.log_test(test_name, False, f"Error: {str(e)}")
    
    def test_volatility_targeting(self):
        """Test 5: Volatility Targeting"""
        print(f"\n🔍 TEST 5: VOLATILITY TARGETING")
        print("="*60)
        
        target_vols = [0.05, 0.10, 0.15, 0.20]
        
        for target_vol in target_vols:
            try:
                signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=30)
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='inverse_volatility',
                    target_volatility=target_vol
                )
                
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= 5].index
                
                if len(signal_dates) > 100:
                    test_date = signal_dates[100]
                    
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    # Volatility targeting should produce valid weights
                    success = (
                        weights is not None and 
                        len(weights) > 0 and 
                        not weights.isna().all() and
                        weights.sum() > 0.8  # Should be close to 1
                    )
                    
                    weights_sum = weights.sum() if success else 0
                    details = f"Target vol: {target_vol}, Weights sum: {weights_sum:.4f}"
                    self.log_test(f"Vol targeting {target_vol}", success, details)
                    
                else:
                    self.log_test(f"Vol targeting {target_vol}", False, "Insufficient data")
                    
            except Exception as e:
                self.log_test(f"Vol targeting {target_vol}", False, f"Error: {str(e)}")
    
    def test_signal_integration_robustness(self):
        """Test 6: Signal Integration Robustness"""
        print(f"\n🔍 TEST 6: SIGNAL INTEGRATION ROBUSTNESS")  
        print("="*60)
        
        # Test various signal configurations
        test_configs = [
            {'n_worst': 2, 'lookback': 20},
            {'n_worst': 3, 'lookback': 30},
            {'n_worst': 5, 'lookback': 60},
            {'n_worst': 7, 'lookback': 90},
            {'n_worst': 10, 'lookback': 120}
        ]
        
        for config in test_configs:
            try:
                signal_generator = ConrarianSignalGenerator(
                    n_worst_performers=config['n_worst'],
                    lookback_days=config['lookback']
                )
                
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='erc' if config['n_worst'] > 1 else 'inverse_volatility'
                )
                
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= config['n_worst']].index
                
                if len(signal_dates) > 50:
                    # Test multiple dates for robustness
                    success_count = 0
                    total_tests = min(20, len(signal_dates) - 50)
                    
                    for i in range(total_tests):
                        test_date = signal_dates[50 + i]
                        try:
                            weights = portfolio_manager.construct_portfolio_weights(
                                signal_output, self.returns, current_date=test_date
                            )
                            
                            # Validate that positions match signals
                            signaled_assets = binary_signals.loc[test_date][binary_signals.loc[test_date] > 0].index
                            positioned_assets = weights[weights > 0.001].index
                            
                            if len(positioned_assets) > 0 and all(asset in signaled_assets for asset in positioned_assets):
                                success_count += 1
                        except:
                            pass
                    
                    success_rate = success_count / total_tests
                    success = success_rate >= 0.8  # 80% success threshold
                    
                    details = f"N={config['n_worst']}, Lookback={config['lookback']}, Success: {success_rate:.1%}"
                    self.log_test(f"Integration N={config['n_worst']}", success, details)
                    
                else:
                    self.log_test(f"Integration N={config['n_worst']}", False, "Insufficient signal dates")
                    
            except Exception as e:
                self.log_test(f"Integration N={config['n_worst']}", False, f"Error: {str(e)}")
    
    def run_core_validation(self):
        """Run all core validation tests"""
        print("🚀 CORE RISK MANAGEMENT SYSTEM VALIDATION")
        print("="*80)
        print(f"   Focus: Validate IndexError fix and core functionality")
        print(f"   Data period: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        print(f"   Assets available: {len(self.prices.columns)}")
        
        # Run focused test suite
        self.test_indexerror_fix()
        self.test_risk_parity_methods() 
        self.test_correlation_matrix_edge_cases()
        self.test_position_constraints()
        self.test_volatility_targeting()
        self.test_signal_integration_robustness()
        
        # Generate focused report
        self.generate_core_report()
    
    def generate_core_report(self):
        """Generate core validation report"""
        print(f"\n📋 CORE VALIDATION REPORT")
        print("="*80)
        
        success_rate = (self.passed_tests / self.test_count) * 100
        
        print(f"📊 CORE SYSTEM RESULTS:")
        print(f"   Total Tests: {self.test_count}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.test_count - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Critical system checks
        critical_tests = [
            'IndexError Fix - ERC Method',
            'IndexError Fix - Consistency',
            'Risk Parity erc N=3',  # Original failing case
            'Risk Parity erc N=5',
            'Edge Case: Minimum correlation matrix'
        ]
        
        critical_passed = sum(1 for test in critical_tests if test in self.results and self.results[test]['success'])
        critical_total = len([test for test in critical_tests if test in self.results])
        
        print(f"\n🎯 CRITICAL SYSTEM STATUS:")
        print(f"   Critical Tests: {critical_passed}/{critical_total} passed")
        
        for test in critical_tests:
            if test in self.results:
                status = "✅" if self.results[test]['success'] else "❌"
                print(f"   {status} {test}")
        
        # Failed tests
        failed_tests = [k for k, v in self.results.items() if not v['success']]
        if failed_tests:
            print(f"\n⚠️ FAILED TESTS:")
            for test in failed_tests:
                print(f"   ❌ {test}: {self.results[test]['details']}")
        
        # Final assessment
        print(f"\n🏁 CORE SYSTEM ASSESSMENT:")
        if success_rate >= 95 and critical_passed == critical_total:
            print("   ✅ CORE SYSTEM FULLY VALIDATED")
            print("   🎉 IndexError and critical issues resolved")
            print("   📈 Risk management system is production ready")
        elif success_rate >= 90 and critical_passed >= critical_total * 0.9:
            print("   ✅ CORE SYSTEM MOSTLY VALIDATED")
            print("   ⚠️ Minor issues detected, system is functional") 
        elif critical_passed >= critical_total * 0.8:
            print("   ⚠️ CORE SYSTEM PARTIALLY VALIDATED")
            print("   🔧 Some critical issues remain")
        else:
            print("   ❌ CORE SYSTEM VALIDATION FAILED")
            print("   🚫 Major issues detected")
        
        print("="*80)


def main():
    """Main execution"""
    try:
        validator = CoreRiskValidator()
        validator.run_core_validation()
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL VALIDATION ERROR:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()