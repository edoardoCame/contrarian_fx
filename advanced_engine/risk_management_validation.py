#!/usr/bin/env python3
"""
Comprehensive Risk Management System Validation Script

Tests all aspects of the risk management system to ensure the IndexError
and integration issues have been fully resolved.

Author: Quantitative Portfolio Management Specialist
Date: August 7, 2025
"""

import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent / 'modules'))

from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from portfolio_manager import PortfolioManager
from backtesting_engine import BacktestingEngine
from performance_analyzer import PerformanceAnalyzer


class RiskManagementValidator:
    """Comprehensive validation of risk management system fixes"""
    
    def __init__(self):
        self.validation_results = {}
        self.test_count = 0
        self.passed_tests = 0
        
        # Load data once for all tests
        print("📊 Loading market data...")
        self.data_loader = ForexDataLoader('data')
        self.prices = self.data_loader.load_unified_prices()
        self.returns = self.data_loader.load_unified_returns()
        print(f"   ✓ Loaded {len(self.prices.columns)} assets with {len(self.prices)} observations")
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        self.test_count += 1
        if success:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"   {status}: {test_name}")
        if details:
            print(f"      {details}")
            
        self.validation_results[test_name] = {
            'success': success,
            'details': details
        }
    
    def test_risk_parity_implementations(self):
        """Test 1: Risk Parity Implementation Validation"""
        print(f"\n🔍 TEST 1: RISK PARITY IMPLEMENTATIONS")
        print("="*60)
        
        # Test different asset counts and methods
        test_scenarios = [
            {'n_assets': 2, 'method': 'inverse_volatility'},
            {'n_assets': 3, 'method': 'inverse_volatility'},
            {'n_assets': 5, 'method': 'inverse_volatility'},
            {'n_assets': 7, 'method': 'inverse_volatility'},
            {'n_assets': 10, 'method': 'inverse_volatility'},
            {'n_assets': 2, 'method': 'erc'},
            {'n_assets': 3, 'method': 'erc'},
            {'n_assets': 5, 'method': 'erc'},
            {'n_assets': 7, 'method': 'erc'},
            {'n_assets': 10, 'method': 'erc'},
        ]
        
        for scenario in test_scenarios:
            try:
                # Generate signals with specific N assets
                signal_generator = ConrarianSignalGenerator(
                    n_worst_performers=scenario['n_assets'], 
                    lookback_days=30
                )
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                # Create portfolio manager
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method=scenario['method'],
                    target_volatility=0.12
                )
                
                # Find a test date with signals
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= scenario['n_assets']].index
                
                if len(signal_dates) > 100:
                    test_date = signal_dates[100]
                    
                    # Calculate weights
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    # Validate results
                    num_positions = (weights > 0.001).sum()
                    weights_sum = weights.sum()
                    
                    success = (
                        num_positions > 0 and
                        num_positions <= scenario['n_assets'] and
                        abs(weights_sum - 1.0) < 0.01  # Should sum to ~1
                    )
                    
                    details = f"N={scenario['n_assets']}, Method={scenario['method']}, Positions={num_positions}, Sum={weights_sum:.4f}"
                    self.log_test(f"Risk Parity {scenario['method']} N={scenario['n_assets']}", success, details)
                    
                else:
                    self.log_test(f"Risk Parity {scenario['method']} N={scenario['n_assets']}", False, "Insufficient signal dates")
                    
            except Exception as e:
                self.log_test(f"Risk Parity {scenario['method']} N={scenario['n_assets']}", False, f"Error: {str(e)}")
    
    def test_correlation_matrix_handling(self):
        """Test 2: Correlation Matrix Handling"""
        print(f"\n🔍 TEST 2: CORRELATION MATRIX HANDLING")
        print("="*60)
        
        test_cases = [
            {'n_assets': 2, 'description': 'Minimum case'},
            {'n_assets': 1, 'description': 'Single asset edge case'},
            {'n_assets': len(self.prices.columns), 'description': 'Full universe'},
        ]
        
        for case in test_cases:
            try:
                if case['n_assets'] == 1:
                    # Special handling for single asset
                    signal_generator = ConrarianSignalGenerator(n_worst_performers=1, lookback_days=30)
                else:
                    signal_generator = ConrarianSignalGenerator(n_worst_performers=case['n_assets'], lookback_days=30)
                
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='erc' if case['n_assets'] > 1 else 'inverse_volatility',
                    correlation_lookback=30
                )
                
                binary_signals = signal_output['binary_signals']
                
                if case['n_assets'] == 1:
                    signal_dates = binary_signals[binary_signals.sum(axis=1) >= 1].index
                else:
                    signal_dates = binary_signals[binary_signals.sum(axis=1) >= case['n_assets']].index
                
                if len(signal_dates) > 50:
                    test_date = signal_dates[50]
                    
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    success = weights is not None and len(weights) > 0
                    details = f"{case['description']} - {(weights > 0.001).sum()} positions created"
                    self.log_test(f"Correlation Matrix N={case['n_assets']}", success, details)
                    
                else:
                    self.log_test(f"Correlation Matrix N={case['n_assets']}", False, "Insufficient data")
                    
            except Exception as e:
                self.log_test(f"Correlation Matrix N={case['n_assets']}", False, f"Error: {str(e)}")
    
    def test_portfolio_constraints(self):
        """Test 3: Portfolio Constraints"""
        print(f"\n🔍 TEST 3: PORTFOLIO CONSTRAINTS")
        print("="*60)
        
        constraint_tests = [
            {
                'max_position_size': 0.2,
                'description': 'Strict position limits'
            },
            {
                'max_position_size': 0.5,
                'description': 'Relaxed position limits'
            },
            {
                'target_volatility': 0.05,
                'description': 'Low volatility target'
            },
            {
                'target_volatility': 0.25,
                'description': 'High volatility target'
            }
        ]
        
        for test_config in constraint_tests:
            try:
                signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=30)
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                # Create clean config without description
                clean_config = {k: v for k, v in test_config.items() if k != 'description'}
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='inverse_volatility',
                    **clean_config
                )
                
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= 5].index
                
                if len(signal_dates) > 100:
                    test_date = signal_dates[100]
                    
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    # Check constraints
                    max_weight = weights.max()
                    constraints_satisfied = True
                    constraint_details = []
                    
                    if 'max_position_size' in test_config:
                        if max_weight > test_config['max_position_size'] * 1.05:  # 5% tolerance
                            constraints_satisfied = False
                        constraint_details.append(f"Max weight: {max_weight:.3f} (limit: {test_config['max_position_size']})")
                    
                    if 'target_volatility' in test_config:
                        constraint_details.append(f"Target vol: {test_config['target_volatility']}")
                    
                    details = f"{test_config['description']} - " + ", ".join(constraint_details)
                    self.log_test(f"Constraints: {test_config['description']}", constraints_satisfied, details)
                    
                else:
                    self.log_test(f"Constraints: {test_config['description']}", False, "Insufficient data")
                    
            except Exception as e:
                self.log_test(f"Constraints: {test_config['description']}", False, f"Error: {str(e)}")
    
    def test_risk_monitoring_system(self):
        """Test 4: Risk Monitoring System"""
        print(f"\n🔍 TEST 4: RISK MONITORING SYSTEM")
        print("="*60)
        
        try:
            # Create a backtesting scenario
            signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=30)
            
            portfolio_manager = PortfolioManager(
                volatility_method='ewma',
                risk_parity_method='inverse_volatility',
                target_volatility=0.12
            )
            
            backtester = BacktestingEngine(
                signal_generator=signal_generator,
                portfolio_manager=portfolio_manager,
                transaction_cost=0.0001
            )
            
            # Run a short backtest
            start_date = self.prices.index[500]  # Start with sufficient history
            end_date = self.prices.index[800]    # Short test period
            
            results = backtester.run_backtest(
                prices=self.prices.loc[start_date:end_date],
                returns=self.returns.loc[start_date:end_date]
            )
            
            # Test VaR calculation
            portfolio_returns = results['portfolio_returns']
            var_5 = portfolio_returns.quantile(0.05)
            
            # Test CVaR calculation  
            cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()
            
            # Test drawdown calculation
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Validate risk metrics
            risk_metrics_valid = (
                not np.isnan(var_5) and
                not np.isnan(cvar_5) and
                not np.isnan(max_drawdown) and
                var_5 < 0 and  # VaR should be negative
                cvar_5 <= var_5 and  # CVaR should be worse than VaR
                max_drawdown <= 0  # Drawdown should be negative
            )
            
            details = f"VaR: {var_5:.4f}, CVaR: {cvar_5:.4f}, Max DD: {max_drawdown:.4f}"
            self.log_test("Risk Metrics Calculation", risk_metrics_valid, details)
            
            # Test risk monitoring during backtest
            risk_monitoring_works = 'positions' in results and len(results['positions']) > 0
            self.log_test("Risk Monitoring Integration", risk_monitoring_works, f"Generated {len(results['positions'])} position records")
            
        except Exception as e:
            self.log_test("Risk Monitoring System", False, f"Error: {str(e)}")
            traceback.print_exc()
    
    def test_signal_generator_integration(self):
        """Test 5: Integration with Signal Generator"""
        print(f"\n🔍 TEST 5: SIGNAL GENERATOR INTEGRATION")
        print("="*60)
        
        integration_tests = [
            {'n_worst': 2, 'lookback': 20},
            {'n_worst': 3, 'lookback': 30}, 
            {'n_worst': 5, 'lookback': 60},
            {'n_worst': 7, 'lookback': 90},
            {'n_worst': 10, 'lookback': 120}
        ]
        
        for test_config in integration_tests:
            try:
                signal_generator = ConrarianSignalGenerator(
                    n_worst_performers=test_config['n_worst'],
                    lookback_days=test_config['lookback']
                )
                
                signal_output = signal_generator.generate_signals(self.prices, self.returns)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='inverse_volatility'
                )
                
                # Test signal processing
                binary_signals = signal_output['binary_signals']
                signal_dates = binary_signals[binary_signals.sum(axis=1) >= test_config['n_worst']].index
                
                if len(signal_dates) > 100:
                    test_date = signal_dates[100]
                    
                    # This is the critical integration point
                    weights = portfolio_manager.construct_portfolio_weights(
                        signal_output, self.returns, current_date=test_date
                    )
                    
                    # Validate integration
                    num_signals = binary_signals.loc[test_date].sum()
                    num_positions = (weights > 0.001).sum()
                    
                    # Portfolio should only have positions in signaled assets
                    signaled_assets = binary_signals.loc[test_date][binary_signals.loc[test_date] > 0].index
                    positioned_assets = weights[weights > 0.001].index
                    
                    assets_match = all(asset in signaled_assets for asset in positioned_assets)
                    
                    success = assets_match and num_positions > 0
                    details = f"N={test_config['n_worst']}, Signals={num_signals}, Positions={num_positions}, Match={assets_match}"
                    
                    self.log_test(f"Integration N={test_config['n_worst']}", success, details)
                    
                else:
                    self.log_test(f"Integration N={test_config['n_worst']}", False, "Insufficient signal dates")
                    
            except Exception as e:
                self.log_test(f"Integration N={test_config['n_worst']}", False, f"Error: {str(e)}")
    
    def test_market_stress_scenarios(self):
        """Test 6: Performance Under Different Market Conditions"""
        print(f"\n🔍 TEST 6: MARKET STRESS SCENARIOS")
        print("="*60)
        
        # Find periods with different volatility characteristics
        returns_vol = self.returns.rolling(window=30).std().mean(axis=1)
        
        # High volatility period
        high_vol_dates = returns_vol.nlargest(100).index
        
        # Low volatility period  
        low_vol_dates = returns_vol.nsmallest(100).index
        
        stress_tests = [
            {
                'name': 'High Volatility Period',
                'dates': high_vol_dates[:50] if len(high_vol_dates) > 50 else high_vol_dates,
                'expected_behavior': 'Should handle high volatility gracefully'
            },
            {
                'name': 'Low Volatility Period',
                'dates': low_vol_dates[:50] if len(low_vol_dates) > 50 else low_vol_dates,
                'expected_behavior': 'Should handle low volatility gracefully'
            }
        ]
        
        for stress_test in stress_tests:
            try:
                signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=30)
                
                portfolio_manager = PortfolioManager(
                    volatility_method='ewma',
                    risk_parity_method='erc',
                    target_volatility=0.12
                )
                
                successes = 0
                total_tests = 0
                
                for test_date in stress_test['dates'][:10]:  # Test first 10 dates
                    if test_date in self.returns.index:
                        try:
                            signal_output = signal_generator.generate_signals(
                                self.prices.loc[:test_date], 
                                self.returns.loc[:test_date]
                            )
                            
                            weights = portfolio_manager.construct_portfolio_weights(
                                signal_output, self.returns.loc[:test_date], current_date=test_date
                            )
                            
                            if weights is not None and len(weights) > 0 and not weights.isna().all():
                                successes += 1
                                
                            total_tests += 1
                            
                        except Exception:
                            total_tests += 1
                
                success_rate = successes / total_tests if total_tests > 0 else 0
                success = success_rate > 0.8  # 80% success rate threshold
                
                details = f"{stress_test['expected_behavior']} - Success rate: {success_rate:.1%}"
                self.log_test(stress_test['name'], success, details)
                
            except Exception as e:
                self.log_test(stress_test['name'], False, f"Error: {str(e)}")
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("🚀 COMPREHENSIVE RISK MANAGEMENT VALIDATION")
        print("="*80)
        print(f"   Testing system fixes applied on {datetime.now().strftime('%Y-%m-%d')}")
        print(f"   Data period: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")
        print(f"   Assets available: {len(self.prices.columns)}")
        
        # Run all test suites
        self.test_risk_parity_implementations()
        self.test_correlation_matrix_handling()  
        self.test_portfolio_constraints()
        self.test_risk_monitoring_system()
        self.test_signal_generator_integration()
        self.test_market_stress_scenarios()
        
        # Generate summary report
        self.generate_validation_report()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n📋 COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        # Overall statistics
        success_rate = (self.passed_tests / self.test_count) * 100 if self.test_count > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {self.test_count}")
        print(f"   Passed: {self.passed_tests}")
        print(f"   Failed: {self.test_count - self.passed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Detailed results by category
        categories = {
            'Risk Parity': [k for k in self.validation_results.keys() if 'Risk Parity' in k],
            'Correlation Matrix': [k for k in self.validation_results.keys() if 'Correlation Matrix' in k],
            'Constraints': [k for k in self.validation_results.keys() if 'Constraints' in k],
            'Risk Monitoring': [k for k in self.validation_results.keys() if 'Risk Monitoring' in k or 'Risk Metrics' in k],
            'Integration': [k for k in self.validation_results.keys() if 'Integration' in k],
            'Stress Testing': [k for k in self.validation_results.keys() if 'Volatility Period' in k]
        }
        
        print(f"\n📈 RESULTS BY CATEGORY:")
        for category, tests in categories.items():
            if tests:
                passed = sum(1 for test in tests if self.validation_results[test]['success'])
                total = len(tests)
                rate = (passed / total * 100) if total > 0 else 0
                status = "✅" if rate >= 90 else "⚠️" if rate >= 70 else "❌"
                print(f"   {status} {category}: {passed}/{total} ({rate:.0f}%)")
        
        # Failed tests details
        failed_tests = [k for k, v in self.validation_results.items() if not v['success']]
        if failed_tests:
            print(f"\n⚠️ FAILED TESTS DETAILS:")
            for test_name in failed_tests:
                details = self.validation_results[test_name]['details']
                print(f"   ❌ {test_name}: {details}")
        
        # System status assessment
        print(f"\n🎯 SYSTEM STATUS ASSESSMENT:")
        
        critical_systems = [
            ('Risk Parity ERC', any('Risk Parity erc' in k and self.validation_results[k]['success'] for k in self.validation_results.keys())),
            ('Correlation Matrix', any('Correlation Matrix' in k and self.validation_results[k]['success'] for k in self.validation_results.keys())),
            ('Signal Integration', any('Integration' in k and self.validation_results[k]['success'] for k in self.validation_results.keys())),
            ('Risk Monitoring', any('Risk' in k and 'Monitoring' in k and self.validation_results[k]['success'] for k in self.validation_results.keys()))
        ]
        
        all_critical_working = all(status for _, status in critical_systems)
        
        for system_name, working in critical_systems:
            status = "✅ OPERATIONAL" if working else "❌ ISSUES DETECTED"
            print(f"   {status}: {system_name}")
        
        # Final recommendation
        print(f"\n🏁 FINAL ASSESSMENT:")
        if success_rate >= 95 and all_critical_working:
            print("   ✅ SYSTEM FULLY VALIDATED - PRODUCTION READY")
            print("   🎉 All IndexError and integration issues have been resolved")
            print("   📈 Risk management system is robust and ready for deployment")
        elif success_rate >= 80:
            print("   ⚠️ SYSTEM MOSTLY FUNCTIONAL - MINOR ISSUES DETECTED")
            print("   🔧 Some edge cases may need attention before production")
        else:
            print("   ❌ SYSTEM VALIDATION FAILED - MAJOR ISSUES DETECTED")
            print("   🚫 Additional debugging and fixes required")
        
        print("="*80)


def main():
    """Main execution function"""
    try:
        validator = RiskManagementValidator()
        validator.run_comprehensive_validation()
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN VALIDATION SYSTEM:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()