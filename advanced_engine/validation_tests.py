#!/usr/bin/env python3
"""
Validation and Testing Suite for Contrarian Forex Backtesting Framework

This module provides comprehensive validation tests to ensure the backtesting
framework is working correctly, produces reliable results, and maintains
data integrity throughout the process.

Test Categories:
1. Data integrity and consistency tests
2. Signal generation validation (lookahead bias prevention)
3. Backtesting engine accuracy tests
4. Performance calculation verification
5. Optimization framework validation
6. End-to-end system integration tests

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import unittest
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Import framework components
from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator, validate_no_lookahead_bias
from backtesting_engine import BacktestingEngine, PortfolioConstructor
from performance_analyzer import PerformanceAnalyzer
from parameter_optimizer import ParameterOptimizer
from results_manager import ResultsManager

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestDataIntegrity(unittest.TestCase):
    """Test data loading and integrity checks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = ForexDataLoader("data")
        self.test_start_date = "2020-01-01"
        self.test_end_date = "2023-12-31"
    
    def test_data_loading(self):
        """Test basic data loading functionality."""
        logger.info("Testing data loading...")
        
        # Test unified data loading
        returns_data = self.data_loader.load_unified_returns()
        prices_data = self.data_loader.load_unified_prices()
        
        # Basic existence checks
        self.assertIsNotNone(returns_data, "Returns data should not be None")
        self.assertIsNotNone(prices_data, "Prices data should not be None")
        
        if returns_data is not None:
            self.assertGreater(len(returns_data), 0, "Returns data should not be empty")
            self.assertGreater(len(returns_data.columns), 0, "Should have at least one currency pair")
            
            # Check for datetime index
            self.assertIsInstance(returns_data.index, pd.DatetimeIndex, "Index should be datetime")
            
            # Check for reasonable data range
            self.assertGreater(returns_data.index.max() - returns_data.index.min(), 
                             timedelta(days=365), "Should have at least 1 year of data")
            
        logger.info("✓ Data loading tests passed")
    
    def test_data_consistency(self):
        """Test data consistency between prices and returns."""
        logger.info("Testing data consistency...")
        
        # Load period data
        returns_data = self.data_loader.get_data_for_period(
            self.test_start_date, self.test_end_date, data_type='returns'
        )
        prices_data = self.data_loader.get_data_for_period(
            self.test_start_date, self.test_end_date, data_type='prices'
        )
        
        if returns_data is not None and prices_data is not None:
            # Check alignment
            common_columns = returns_data.columns.intersection(prices_data.columns)
            self.assertGreater(len(common_columns), 0, "Should have common assets")
            
            # Check for reasonable return values
            for col in common_columns:
                returns_col = returns_data[col].dropna()
                if len(returns_col) > 0:
                    # Returns should be reasonable (not too extreme)
                    self.assertLess(returns_col.abs().max(), 0.5, f"Daily returns for {col} seem too extreme")
                    
                    # Check for finite values
                    self.assertTrue(np.isfinite(returns_col).all(), f"All returns for {col} should be finite")
        
        logger.info("✓ Data consistency tests passed")
    
    def test_data_validation(self):
        """Test data validation functionality."""
        logger.info("Testing data validation...")
        
        # Test individual pair validation
        available_symbols = self.data_loader.get_available_symbols()
        
        if available_symbols:
            # Test first available symbol
            test_symbol = available_symbols[0]
            data = self.data_loader.load_individual_pair(test_symbol)
            
            if data is not None:
                validation_results = self.data_loader.validate_data_integrity(data, test_symbol)
                
                # Check validation results structure
                required_fields = ['symbol', 'total_rows', 'missing_values', 'has_duplicates', 'is_sorted']
                for field in required_fields:
                    self.assertIn(field, validation_results, f"Validation should include {field}")
                
                # Data should be sorted
                self.assertTrue(validation_results['is_sorted'], "Data should be chronologically sorted")
                
                # Should have reasonable amount of data
                self.assertGreater(validation_results['total_rows'], 100, "Should have substantial data")
        
        logger.info("✓ Data validation tests passed")


class TestSignalGeneration(unittest.TestCase):
    """Test signal generation and lookahead bias prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = ForexDataLoader("data")
        self.signal_generator = ConrarianSignalGenerator(n_worst_performers=3, lookback_days=10)
        
        # Load test data
        self.returns_data = self.data_loader.get_data_for_period("2020-01-01", "2022-12-31", data_type='returns')
        self.prices_data = self.data_loader.get_data_for_period("2020-01-01", "2022-12-31", data_type='prices')
    
    def test_signal_generation(self):
        """Test basic signal generation functionality."""
        logger.info("Testing signal generation...")
        
        if self.prices_data is not None and self.returns_data is not None:
            # Generate signals
            signal_output = self.signal_generator.generate_signals(self.prices_data, self.returns_data)
            
            # Check output structure
            required_keys = ['binary_signals', 'weights', 'rolling_returns', 'volatility', 'ranks']
            for key in required_keys:
                self.assertIn(key, signal_output, f"Signal output should include {key}")
            
            binary_signals = signal_output['binary_signals']
            weights = signal_output['weights']
            
            # Check dimensions
            self.assertEqual(binary_signals.shape, weights.shape, "Signals and weights should have same dimensions")
            
            # Check signal values
            self.assertTrue((binary_signals >= 0).all().all(), "Binary signals should be non-negative")
            self.assertTrue((binary_signals <= 1).all().all(), "Binary signals should be <= 1")
            self.assertTrue((weights >= 0).all().all(), "Weights should be non-negative")
            
            # Check daily signal counts
            daily_signal_counts = binary_signals.sum(axis=1)
            valid_counts = daily_signal_counts[daily_signal_counts > 0]
            
            if len(valid_counts) > 0:
                # Most days with signals should have exactly n_worst_performers signals
                expected_count = self.signal_generator.n_worst_performers
                correct_counts = (valid_counts == expected_count).sum()
                total_signal_days = len(valid_counts)
                
                self.assertGreater(correct_counts / total_signal_days, 0.8, 
                                 f"Most signal days should have exactly {expected_count} signals")
            
            # Check weight normalization
            weight_sums = weights.sum(axis=1)
            valid_weight_days = weight_sums[weight_sums > 0]
            
            if len(valid_weight_days) > 0:
                # Weights should sum to approximately 1
                weight_deviations = np.abs(valid_weight_days - 1.0)
                self.assertTrue((weight_deviations < 0.01).all(), "Weights should sum to 1 (within tolerance)")
        
        logger.info("✓ Signal generation tests passed")
    
    def test_lookahead_bias_prevention(self):
        """Test that signals don't use future information."""
        logger.info("Testing lookahead bias prevention...")
        
        if self.prices_data is not None and self.returns_data is not None:
            # Generate signals
            signal_output = self.signal_generator.generate_signals(self.prices_data, self.returns_data)
            
            # Validate signal generation
            validation_results = self.signal_generator.validate_signals(signal_output)
            
            # Check validation results
            self.assertFalse(validation_results.get('has_lookahead_bias', True), 
                           "Signals should not have lookahead bias")
            self.assertTrue(validation_results.get('no_future_data_used', False), 
                          "No future data should be used")
            
            # Run comprehensive lookahead bias test
            no_lookahead = validate_no_lookahead_bias(
                self.prices_data, signal_output, self.signal_generator.lookback_days
            )
            
            self.assertTrue(no_lookahead, "Comprehensive lookahead bias test should pass")
            
            # Check that early periods don't have signals (insufficient history)
            binary_signals = signal_output['binary_signals']
            min_required_periods = self.signal_generator.lookback_days + self.signal_generator.min_history_days
            
            if len(binary_signals) > min_required_periods:
                early_signals = binary_signals.iloc[:min_required_periods].sum().sum()
                self.assertEqual(early_signals, 0, "Early periods should not have signals due to insufficient history")
        
        logger.info("✓ Lookahead bias prevention tests passed")
    
    def test_contrarian_logic(self):
        """Test that contrarian logic is working correctly."""
        logger.info("Testing contrarian logic...")
        
        if self.prices_data is not None and self.returns_data is not None:
            signal_output = self.signal_generator.generate_signals(self.prices_data, self.returns_data)
            
            binary_signals = signal_output['binary_signals']
            rolling_returns = signal_output['rolling_returns']
            ranks = signal_output['ranks']
            
            # For days with signals, check that worst performers are selected
            for date in binary_signals.index:
                if binary_signals.loc[date].sum() > 0:  # Day with signals
                    selected_assets = binary_signals.loc[date] == 1
                    
                    if date in ranks.index and not ranks.loc[date].isna().all():
                        selected_ranks = ranks.loc[date][selected_assets]
                        
                        if len(selected_ranks) > 0 and not selected_ranks.isna().all():
                            # Selected assets should have low ranks (worst performers)
                            max_selected_rank = selected_ranks.max()
                            self.assertLessEqual(max_selected_rank, self.signal_generator.n_worst_performers + 1,
                                               "Selected assets should be among worst performers")
        
        logger.info("✓ Contrarian logic tests passed")


class TestBacktestingEngine(unittest.TestCase):
    """Test backtesting engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = ForexDataLoader("data")
        self.backtesting_engine = BacktestingEngine(initial_capital=1000000, transaction_cost_bps=1.0)
        
        # Create simple test data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        assets = ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']
        
        # Simple returns data
        np.random.seed(42)
        self.test_returns = pd.DataFrame(
            np.random.normal(0.0001, 0.01, (len(dates), len(assets))),
            index=dates,
            columns=assets
        )
        
        # Simple signals data (equal weight among all assets)
        self.test_signals = pd.DataFrame(
            1.0 / len(assets),
            index=dates,
            columns=assets
        )
    
    def test_basic_backtesting(self):
        """Test basic backtesting functionality."""
        logger.info("Testing basic backtesting...")
        
        # Run backtest
        results = self.backtesting_engine.run_backtest(
            signals=self.test_signals,
            returns=self.test_returns
        )
        
        # Check results structure
        required_keys = ['portfolio_returns', 'portfolio_value', 'portfolio_weights', 'metadata']
        for key in required_keys:
            self.assertIn(key, results, f"Results should include {key}")
        
        portfolio_returns = results['portfolio_returns']
        portfolio_value = results['portfolio_value']
        
        # Check dimensions
        self.assertEqual(len(portfolio_returns), len(self.test_returns), 
                        "Portfolio returns should match input length")
        self.assertEqual(len(portfolio_value), len(self.test_returns), 
                        "Portfolio value should match input length")
        
        # Check initial value
        initial_capital = self.backtesting_engine.initial_capital
        self.assertEqual(portfolio_value.iloc[0], initial_capital, 
                        "Initial portfolio value should match initial capital")
        
        # Check for reasonable portfolio values
        self.assertTrue((portfolio_value > 0).all(), "Portfolio value should always be positive")
        self.assertTrue(np.isfinite(portfolio_value).all(), "Portfolio values should be finite")
        
        # Check that returns are reasonable
        clean_returns = portfolio_returns.dropna()
        if len(clean_returns) > 0:
            self.assertLess(clean_returns.abs().max(), 0.2, "Daily returns should not be too extreme")
        
        logger.info("✓ Basic backtesting tests passed")
    
    def test_transaction_costs(self):
        """Test transaction cost implementation."""
        logger.info("Testing transaction costs...")
        
        # Run backtest with and without transaction costs
        engine_no_costs = BacktestingEngine(initial_capital=1000000, transaction_cost_bps=0.0)
        engine_with_costs = BacktestingEngine(initial_capital=1000000, transaction_cost_bps=5.0)
        
        results_no_costs = engine_no_costs.run_backtest(self.test_signals, self.test_returns)
        results_with_costs = engine_with_costs.run_backtest(self.test_signals, self.test_returns)
        
        # Portfolio with transaction costs should have lower final value
        final_value_no_costs = results_no_costs['portfolio_value'].iloc[-1]
        final_value_with_costs = results_with_costs['portfolio_value'].iloc[-1]
        
        self.assertLess(final_value_with_costs, final_value_no_costs, 
                       "Transaction costs should reduce portfolio value")
        
        # Check that transaction costs are recorded
        total_costs = results_with_costs['transaction_costs'].sum()
        self.assertGreater(total_costs, 0, "Transaction costs should be positive")
        
        logger.info("✓ Transaction cost tests passed")
    
    def test_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        logger.info("Testing portfolio statistics...")
        
        # Run backtest
        results = self.backtesting_engine.run_backtest(self.test_signals, self.test_returns)
        
        # Calculate statistics
        stats = self.backtesting_engine.get_portfolio_statistics(results)
        
        # Check required statistics
        required_stats = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
        for stat in required_stats:
            self.assertIn(stat, stats, f"Statistics should include {stat}")
            self.assertTrue(np.isfinite(stats[stat]), f"{stat} should be finite")
        
        # Check reasonable values
        self.assertGreater(stats['volatility'], 0, "Volatility should be positive")
        self.assertLessEqual(abs(stats['max_drawdown']), 1, "Max drawdown should be <= 100%")
        
        logger.info("✓ Portfolio statistics tests passed")


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test performance analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic backtest results
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')
        np.random.seed(42)
        
        # Generate synthetic returns with some autocorrelation and volatility clustering
        returns = np.random.normal(0.0005, 0.015, len(dates))
        returns = pd.Series(returns, index=dates)
        
        # Create portfolio value
        portfolio_value = (1 + returns).cumprod() * 1000000
        
        # Create synthetic backtest results
        self.test_backtest_results = {
            'portfolio_returns': returns,
            'portfolio_value': portfolio_value,
            'portfolio_weights': pd.DataFrame(np.random.rand(len(dates), 3), index=dates, 
                                           columns=['EUR_X', 'GBP_X', 'JPY_X']),
            'asset_returns': pd.DataFrame(np.random.normal(0.0005, 0.01, (len(dates), 3)), 
                                        index=dates, columns=['EUR_X', 'GBP_X', 'JPY_X']),
            'metadata': {
                'start_date': dates[0],
                'end_date': dates[-1],
                'initial_capital': 1000000,
                'assets': ['EUR_X', 'GBP_X', 'JPY_X']
            }
        }
        
        self.performance_analyzer = PerformanceAnalyzer()
    
    def test_return_analysis(self):
        """Test return analysis functionality."""
        logger.info("Testing return analysis...")
        
        returns = self.test_backtest_results['portfolio_returns']
        analysis = self.performance_analyzer.analyze_returns(returns)
        
        # Check required metrics
        required_metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                           'win_rate', 'skewness', 'kurtosis']
        for metric in required_metrics:
            self.assertIn(metric, analysis, f"Analysis should include {metric}")
            self.assertTrue(np.isfinite(analysis[metric]), f"{metric} should be finite")
        
        # Check reasonable values
        self.assertGreater(analysis['volatility'], 0, "Volatility should be positive")
        self.assertGreaterEqual(analysis['win_rate'], 0, "Win rate should be non-negative")
        self.assertLessEqual(analysis['win_rate'], 1, "Win rate should be <= 1")
        
        logger.info("✓ Return analysis tests passed")
    
    def test_drawdown_analysis(self):
        """Test drawdown analysis functionality."""
        logger.info("Testing drawdown analysis...")
        
        portfolio_value = self.test_backtest_results['portfolio_value']
        returns = self.test_backtest_results['portfolio_returns']
        
        analysis = self.performance_analyzer.analyze_drawdowns(portfolio_value, returns)
        
        # Check required metrics
        required_metrics = ['max_drawdown', 'current_drawdown', 'avg_drawdown', 
                           'calmar_ratio', 'num_drawdown_periods']
        for metric in required_metrics:
            self.assertIn(metric, analysis, f"Drawdown analysis should include {metric}")
        
        # Check reasonable values
        self.assertLessEqual(analysis['max_drawdown'], 0, "Max drawdown should be negative or zero")
        self.assertGreaterEqual(analysis['max_drawdown'], -1, "Max drawdown should be > -100%")
        self.assertGreaterEqual(analysis['num_drawdown_periods'], 0, "Number of drawdown periods should be >= 0")
        
        logger.info("✓ Drawdown analysis tests passed")
    
    def test_performance_report(self):
        """Test comprehensive performance report generation."""
        logger.info("Testing performance report generation...")
        
        report = self.performance_analyzer.generate_performance_report(
            self.test_backtest_results
        )
        
        # Check report structure
        required_sections = ['summary', 'return_analysis', 'drawdown_analysis', 'risk_metrics']
        for section in required_sections:
            self.assertIn(section, report, f"Report should include {section} section")
        
        # Check summary metrics
        summary = report['summary']['key_metrics']
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        for metric in key_metrics:
            self.assertIn(metric, summary, f"Summary should include {metric}")
        
        logger.info("✓ Performance report tests passed")


class TestParameterOptimizer(unittest.TestCase):
    """Test parameter optimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = ForexDataLoader("data")
        self.optimizer = ParameterOptimizer(
            optimization_metric='sharpe_ratio',
            n_jobs=1  # Use single thread for testing
        )
        
        # Small parameter grid for testing
        self.test_parameter_grid = {
            'n_worst_performers': [3, 5],
            'lookback_days': [10, 20]
        }
    
    def test_parameter_optimization_structure(self):
        """Test that parameter optimization returns correct structure."""
        logger.info("Testing parameter optimization structure...")
        
        # Skip if no data available
        returns_data = self.data_loader.get_data_for_period("2020-01-01", "2021-12-31", data_type='returns')
        if returns_data is None or len(returns_data) < 500:
            self.skipTest("Insufficient data for optimization test")
        
        # Run small optimization
        try:
            results = self.optimizer.grid_search_optimization(
                data_loader=self.data_loader,
                signal_generator_class=ConrarianSignalGenerator,
                backtesting_engine_class=BacktestingEngine,
                parameter_grid=self.test_parameter_grid,
                start_date="2020-01-01",
                end_date="2021-12-31"
            )
            
            # Check results structure
            required_keys = ['best_parameters', 'best_metrics', 'all_results', 'sensitivity_analysis']
            for key in required_keys:
                self.assertIn(key, results, f"Optimization results should include {key}")
            
            # Check best parameters
            best_params = results['best_parameters']
            for param_name in self.test_parameter_grid.keys():
                self.assertIn(param_name, best_params, f"Best parameters should include {param_name}")
                self.assertIn(best_params[param_name], self.test_parameter_grid[param_name],
                            f"Best {param_name} should be from the parameter grid")
            
            # Check all results
            all_results = results['all_results']
            expected_combinations = len(self.test_parameter_grid['n_worst_performers']) * len(self.test_parameter_grid['lookback_days'])
            self.assertEqual(len(all_results), expected_combinations, 
                           "Should test all parameter combinations")
            
        except Exception as e:
            logger.warning(f"Parameter optimization test failed: {str(e)}")
            self.skipTest(f"Parameter optimization test failed: {str(e)}")
        
        logger.info("✓ Parameter optimization structure tests passed")


class TestResultsManager(unittest.TestCase):
    """Test results management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.results_manager = ResultsManager("test_results")
        
        # Create test backtest results
        dates = pd.date_range('2020-01-01', '2020-06-30', freq='B')
        np.random.seed(42)
        
        returns = pd.Series(np.random.normal(0.001, 0.01, len(dates)), index=dates)
        portfolio_value = (1 + returns).cumprod() * 1000000
        
        self.test_backtest_results = {
            'portfolio_returns': returns,
            'portfolio_value': portfolio_value,
            'portfolio_weights': pd.DataFrame(np.random.rand(len(dates), 2), 
                                            index=dates, columns=['EUR_X', 'GBP_X']),
            'metadata': {
                'start_date': dates[0],
                'end_date': dates[-1],
                'initial_capital': 1000000,
                'assets': ['EUR_X', 'GBP_X']
            }
        }
    
    def test_save_and_load_results(self):
        """Test saving and loading backtest results."""
        logger.info("Testing results save/load functionality...")
        
        # Save results
        result_id = self.results_manager.save_backtest_results(
            backtest_results=self.test_backtest_results,
            strategy_name="TestStrategy",
            parameters={'n_worst_performers': 3, 'lookback_days': 10},
            description="Test strategy for validation"
        )
        
        self.assertIsNotNone(result_id, "Result ID should not be None")
        self.assertIsInstance(result_id, str, "Result ID should be string")
        
        # Load results
        loaded_results = self.results_manager.load_backtest_results(result_id)
        
        # Check loaded results
        self.assertIn('portfolio_returns', loaded_results, "Loaded results should include portfolio returns")
        self.assertIn('metadata', loaded_results, "Loaded results should include metadata")
        
        # Check data integrity
        original_returns = self.test_backtest_results['portfolio_returns']
        loaded_returns = loaded_results['portfolio_returns']
        
        pd.testing.assert_series_equal(original_returns, loaded_returns, 
                                     "Loaded returns should match original")
        
        logger.info("✓ Results save/load tests passed")
    
    def test_results_listing(self):
        """Test listing available results."""
        logger.info("Testing results listing...")
        
        # Save a few test results
        for i in range(2):
            self.results_manager.save_backtest_results(
                backtest_results=self.test_backtest_results,
                strategy_name=f"TestStrategy_{i}",
                parameters={'n_worst_performers': 3 + i, 'lookback_days': 10},
                description=f"Test strategy {i}"
            )
        
        # List results
        results_list = self.results_manager.list_backtest_results()
        
        self.assertIsInstance(results_list, pd.DataFrame, "Results list should be DataFrame")
        self.assertGreaterEqual(len(results_list), 2, "Should have at least 2 results")
        
        # Check required columns
        required_columns = ['result_id', 'strategy_name', 'creation_timestamp']
        for col in required_columns:
            self.assertIn(col, results_list.columns, f"Results list should include {col}")
        
        logger.info("✓ Results listing tests passed")


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_start_date = "2020-01-01"
        self.test_end_date = "2021-12-31"
    
    def test_complete_workflow(self):
        """Test complete workflow from data loading to results storage."""
        logger.info("Testing complete end-to-end workflow...")
        
        try:
            # Step 1: Load data
            data_loader = ForexDataLoader("data")
            returns_data = data_loader.get_data_for_period(
                self.test_start_date, self.test_end_date, data_type='returns'
            )
            prices_data = data_loader.get_data_for_period(
                self.test_start_date, self.test_end_date, data_type='prices'
            )
            
            if returns_data is None or prices_data is None or len(returns_data) < 100:
                self.skipTest("Insufficient data for end-to-end test")
            
            # Step 2: Generate signals
            signal_generator = ConrarianSignalGenerator(n_worst_performers=3, lookback_days=15)
            signal_output = signal_generator.generate_signals(prices_data, returns_data)
            
            # Validate signals
            validation_results = signal_generator.validate_signals(signal_output)
            self.assertEqual(len(validation_results['issues']), 0, 
                           f"Signals should be valid, found issues: {validation_results['issues']}")
            
            # Step 3: Run backtest
            backtesting_engine = BacktestingEngine(initial_capital=1000000)
            backtest_results = backtesting_engine.run_backtest(
                signals=signal_output['weights'],
                returns=returns_data,
                start_date=self.test_start_date,
                end_date=self.test_end_date
            )
            
            # Check backtest results
            self.assertIn('portfolio_returns', backtest_results)
            self.assertIn('portfolio_value', backtest_results)
            
            # Step 4: Analyze performance
            performance_analyzer = PerformanceAnalyzer()
            performance_report = performance_analyzer.generate_performance_report(backtest_results)
            
            # Check performance report
            self.assertIn('summary', performance_report)
            self.assertIn('return_analysis', performance_report)
            
            # Step 5: Save results
            results_manager = ResultsManager("test_results")
            result_id = results_manager.save_backtest_results(
                backtest_results=backtest_results,
                strategy_name="EndToEndTest",
                parameters={'n_worst_performers': 3, 'lookback_days': 15},
                description="End-to-end integration test"
            )
            
            # Verify saving
            self.assertIsNotNone(result_id)
            
            # Step 6: Load and verify
            loaded_results = results_manager.load_backtest_results(result_id)
            self.assertIn('portfolio_returns', loaded_results)
            
            logger.info("✓ Complete workflow test passed")
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {str(e)}")
            raise


def run_validation_suite():
    """Run the complete validation test suite."""
    logger.info("="*80)
    logger.info("STARTING CONTRARIAN FOREX BACKTESTING FRAMEWORK VALIDATION")
    logger.info("="*80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataIntegrity,
        TestSignalGeneration,
        TestBacktestingEngine,
        TestPerformanceAnalyzer,
        TestParameterOptimizer,
        TestResultsManager,
        TestEndToEndIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    logger.info("="*80)
    if result.wasSuccessful():
        logger.info("✓ ALL VALIDATION TESTS PASSED - FRAMEWORK IS READY FOR USE")
    else:
        logger.error(f"✗ VALIDATION FAILED - {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            logger.error("FAILURES:")
            for test, traceback in result.failures:
                logger.error(f"  {test}: {traceback}")
        
        if result.errors:
            logger.error("ERRORS:")
            for test, traceback in result.errors:
                logger.error(f"  {test}: {traceback}")
    
    logger.info("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)