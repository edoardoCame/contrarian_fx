#!/usr/bin/env python3
"""
Example Backtesting Script for Contrarian Forex Strategy

This script demonstrates the complete usage of the high-performance backtesting
framework, including data loading, signal generation, backtesting, performance
analysis, parameter optimization, and results management.

Usage Examples:
1. Basic backtesting with default parameters
2. Parameter optimization across multiple dimensions
3. Walk-forward analysis for robustness testing
4. Performance comparison and analysis
5. Results storage and retrieval

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from datetime import datetime
import warnings

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

# Import framework components
from data_loader import ForexDataLoader
from signal_generator import ConrarianSignalGenerator
from backtesting_engine import BacktestingEngine, PortfolioConstructor
from performance_analyzer import PerformanceAnalyzer
from parameter_optimizer import ParameterOptimizer
from results_manager import ResultsManager

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def example_basic_backtesting():
    """
    Example 1: Basic backtesting with default parameters
    """
    logger.info("="*60)
    logger.info("EXAMPLE 1: Basic Backtesting")
    logger.info("="*60)
    
    # Initialize components
    data_loader = ForexDataLoader("data")
    signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=20)
    backtesting_engine = BacktestingEngine(initial_capital=1000000)
    performance_analyzer = PerformanceAnalyzer()
    results_manager = ResultsManager("results")
    
    # Load data
    logger.info("Loading forex data...")
    returns_data = data_loader.load_unified_returns()
    prices_data = data_loader.load_unified_prices()
    
    if returns_data is None or prices_data is None:
        logger.error("Failed to load data")
        return None
    
    logger.info(f"Loaded data: {returns_data.shape} from {returns_data.index.min()} to {returns_data.index.max()}")
    
    # Filter date range for example
    start_date = "2015-01-01"
    end_date = "2023-12-31"
    
    returns_filtered = returns_data.loc[start_date:end_date]
    prices_filtered = prices_data.loc[start_date:end_date]
    
    logger.info(f"Using data from {start_date} to {end_date}: {len(returns_filtered)} days")
    
    # Generate signals
    logger.info("Generating contrarian signals...")
    signal_output = signal_generator.generate_signals(prices_filtered, returns_filtered)
    
    # Validate signals
    validation_results = signal_generator.validate_signals(signal_output)
    logger.info(f"Signal validation: {len(validation_results['issues'])} issues found")
    
    if validation_results['issues']:
        for issue in validation_results['issues']:
            logger.warning(f"Signal validation issue: {issue}")
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_results = backtesting_engine.run_backtest(
        signals=signal_output['weights'],
        returns=returns_filtered,
        start_date=start_date,
        end_date=end_date
    )
    
    # Analyze performance
    logger.info("Analyzing performance...")
    performance_report = performance_analyzer.generate_performance_report(
        backtest_results, output_dir="results/performance_analysis"
    )
    
    # Print key metrics
    summary = performance_report['summary']['key_metrics']
    logger.info(f"Performance Summary:")
    logger.info(f"  Total Return: {summary['total_return']*100:.2f}%")
    logger.info(f"  Annualized Return: {summary['annualized_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
    logger.info(f"  Calmar Ratio: {summary['calmar_ratio']:.3f}")
    
    # Save results
    logger.info("Saving backtest results...")
    result_id = results_manager.save_backtest_results(
        backtest_results=backtest_results,
        strategy_name="ConrarianFX_Basic",
        parameters={'n_worst_performers': 5, 'lookback_days': 20},
        description="Basic contrarian forex strategy with default parameters",
        tags=['contrarian', 'forex', 'basic']
    )
    
    logger.info(f"Results saved with ID: {result_id}")
    return result_id, backtest_results, performance_report


def example_parameter_optimization():
    """
    Example 2: Parameter optimization using grid search
    """
    logger.info("="*60)
    logger.info("EXAMPLE 2: Parameter Optimization")
    logger.info("="*60)
    
    # Initialize components
    data_loader = ForexDataLoader("data")
    optimizer = ParameterOptimizer(
        optimization_metric='sharpe_ratio',
        secondary_metrics=['max_drawdown', 'calmar_ratio'],
        n_jobs=4
    )
    results_manager = ResultsManager("results")
    
    # Define parameter grid
    parameter_grid = {
        'n_worst_performers': [2, 3, 5, 7, 10],
        'lookback_days': [5, 10, 15, 20, 30]
    }
    
    logger.info(f"Parameter grid: {parameter_grid}")
    logger.info(f"Total combinations: {len(parameter_grid['n_worst_performers']) * len(parameter_grid['lookback_days'])}")
    
    # Run grid search optimization
    logger.info("Starting grid search optimization...")
    optimization_results = optimizer.grid_search_optimization(
        data_loader=data_loader,
        signal_generator_class=ConrarianSignalGenerator,
        backtesting_engine_class=BacktestingEngine,
        parameter_grid=parameter_grid,
        start_date="2010-01-01",
        end_date="2020-12-31",
        performance_analyzer_class=PerformanceAnalyzer
    )
    
    # Print optimization results
    best_params = optimization_results['best_parameters']
    best_metrics = optimization_results['best_metrics']
    
    logger.info("Optimization Results:")
    logger.info(f"  Best Parameters: {best_params}")
    logger.info(f"  Best Train Sharpe: {best_metrics.get('train_sharpe_ratio', 0):.3f}")
    logger.info(f"  Best Validation Sharpe: {best_metrics.get('val_sharpe_ratio', 0):.3f}")
    logger.info(f"  Overfitting Risk: {optimization_results['overfitting_analysis'].get('overfitting_risk', 'UNKNOWN')}")
    
    # Analyze parameter sensitivity
    sensitivity = optimization_results['sensitivity_analysis']
    logger.info("Parameter Sensitivity:")
    for param_name, param_analysis in sensitivity.items():
        logger.info(f"  {param_name}: range={param_analysis['sensitivity_range']:.4f}, optimal={param_analysis['optimal_value']}")
    
    # Save optimization results
    logger.info("Saving optimization results...")
    opt_result_id = results_manager.save_optimization_results(
        optimization_results=optimization_results,
        optimization_method="grid_search",
        parameter_space=parameter_grid,
        description="Grid search optimization for contrarian FX strategy",
        tags=['optimization', 'grid_search', 'contrarian']
    )
    
    logger.info(f"Optimization results saved with ID: {opt_result_id}")
    return opt_result_id, optimization_results


def example_walk_forward_analysis():
    """
    Example 3: Walk-forward analysis for parameter stability
    """
    logger.info("="*60)
    logger.info("EXAMPLE 3: Walk-Forward Analysis")
    logger.info("="*60)
    
    # Initialize components
    data_loader = ForexDataLoader("data")
    optimizer = ParameterOptimizer(walk_forward_periods=63)  # ~3 months
    
    # Smaller parameter grid for walk-forward (computationally intensive)
    parameter_grid = {
        'n_worst_performers': [3, 5, 7],
        'lookback_days': [10, 20, 30]
    }
    
    logger.info("Starting walk-forward analysis...")
    wf_results = optimizer.walk_forward_optimization(
        data_loader=data_loader,
        signal_generator_class=ConrarianSignalGenerator,
        backtesting_engine_class=BacktestingEngine,
        parameter_grid=parameter_grid,
        start_date="2015-01-01",
        end_date="2023-12-31"
    )
    
    # Analyze parameter stability
    stability_metrics = wf_results['parameter_stability']
    performance_stability = wf_results['performance_stability']
    
    logger.info("Parameter Stability Analysis:")
    for param_name, metrics in stability_metrics.items():
        logger.info(f"  {param_name}: CV={metrics['cv']:.3f}, mean={metrics['mean']:.2f}")
    
    logger.info("Performance Stability:")
    logger.info(f"  Mean OOS Performance: {performance_stability['mean_oos_performance']:.4f}")
    logger.info(f"  Performance CV: {performance_stability['cv_oos_performance']:.3f}")
    logger.info(f"  Positive Periods: {performance_stability['positive_periods']}/{performance_stability['total_periods']}")
    
    # Get stable parameters
    recommended_params = wf_results['recommended_parameters']
    logger.info("Recommended Stable Parameters:")
    for param_name, param_info in recommended_params.items():
        logger.info(f"  {param_name}: {param_info['recommended_value']} (stable: {param_info['is_stable']})")
    
    return wf_results


def example_multi_objective_optimization():
    """
    Example 4: Multi-objective optimization balancing return and risk
    """
    logger.info("="*60)
    logger.info("EXAMPLE 4: Multi-Objective Optimization")
    logger.info("="*60)
    
    # Initialize components
    data_loader = ForexDataLoader("data")
    optimizer = ParameterOptimizer()
    
    parameter_grid = {
        'n_worst_performers': [2, 3, 5, 7],
        'lookback_days': [10, 15, 20, 30]
    }
    
    # Define objectives and weights
    objectives = ['sharpe_ratio', 'max_drawdown', 'calmar_ratio']
    objective_weights = [0.5, 0.3, 0.2]  # Higher weight on Sharpe ratio
    
    logger.info(f"Objectives: {objectives}")
    logger.info(f"Weights: {objective_weights}")
    
    logger.info("Starting multi-objective optimization...")
    mo_results = optimizer.multi_objective_optimization(
        data_loader=data_loader,
        signal_generator_class=ConrarianSignalGenerator,
        backtesting_engine_class=BacktestingEngine,
        parameter_grid=parameter_grid,
        start_date="2012-01-01",
        end_date="2022-12-31",
        objectives=objectives,
        objective_weights=objective_weights
    )
    
    # Results
    best_params = mo_results['best_parameters']
    best_score = mo_results['best_composite_score']
    best_metrics = mo_results['best_metrics']
    
    logger.info("Multi-Objective Results:")
    logger.info(f"  Best Parameters: {best_params}")
    logger.info(f"  Best Composite Score: {best_score:.4f}")
    logger.info("  Individual Metrics:")
    for obj in objectives:
        logger.info(f"    {obj}: {best_metrics.get(f'val_{obj}', 0):.4f}")
    
    # Pareto frontier analysis
    pareto_info = mo_results['pareto_frontier']
    logger.info(f"  Pareto Optimal Solutions: {pareto_info['n_pareto_optimal']}/{len(mo_results['all_results_with_scores'])}")
    
    return mo_results


def example_results_comparison():
    """
    Example 5: Compare and analyze multiple backtest results
    """
    logger.info("="*60)
    logger.info("EXAMPLE 5: Results Comparison")
    logger.info("="*60)
    
    # Initialize results manager
    results_manager = ResultsManager("results")
    
    # List available results
    available_results = results_manager.list_backtest_results()
    logger.info(f"Found {len(available_results)} available backtest results")
    
    if len(available_results) < 2:
        logger.warning("Need at least 2 results for comparison. Running basic examples first...")
        # Run basic examples to create some results
        result_id_1, _, _ = example_basic_backtesting()
        
        # Run with different parameters
        data_loader = ForexDataLoader("data")
        signal_generator = ConrarianSignalGenerator(n_worst_performers=7, lookback_days=15)
        backtesting_engine = BacktestingEngine(initial_capital=1000000)
        
        returns_data = data_loader.get_data_for_period("2015-01-01", "2023-12-31", data_type='returns')
        prices_data = data_loader.get_data_for_period("2015-01-01", "2023-12-31", data_type='prices')
        
        signal_output = signal_generator.generate_signals(prices_data, returns_data)
        backtest_results = backtesting_engine.run_backtest(
            signals=signal_output['weights'],
            returns=returns_data
        )
        
        result_id_2 = results_manager.save_backtest_results(
            backtest_results=backtest_results,
            strategy_name="ConrarianFX_Alternative",
            parameters={'n_worst_performers': 7, 'lookback_days': 15},
            description="Alternative parameter set for comparison"
        )
        
        result_ids = [result_id_1, result_id_2]
    else:
        # Use existing results
        result_ids = available_results['result_id'].head(3).tolist()
    
    # Compare results
    logger.info(f"Comparing results: {result_ids}")
    comparison = results_manager.compare_backtest_results(
        result_ids=result_ids,
        metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
    )
    
    logger.info("Comparison Results:")
    for _, row in comparison.iterrows():
        logger.info(f"  {row['strategy_name']} ({row['result_id'][:8]}):")
        logger.info(f"    Total Return: {row.get('total_return', 0)*100:.2f}%")
        logger.info(f"    Sharpe Ratio: {row.get('sharpe_ratio', 0):.3f}")
        logger.info(f"    Max Drawdown: {row.get('max_drawdown', 0)*100:.2f}%")
    
    # Storage usage
    storage_info = results_manager.get_storage_usage()
    logger.info(f"Storage Usage: {storage_info['total_size_gb']:.2f} GB ({storage_info['usage_percentage']:.1f}% of limit)")
    
    return comparison


def example_comprehensive_workflow():
    """
    Example 6: Complete workflow from data loading to results analysis
    """
    logger.info("="*60)
    logger.info("EXAMPLE 6: Comprehensive Workflow")
    logger.info("="*60)
    
    # Step 1: Load and validate data
    logger.info("Step 1: Data Loading and Validation")
    data_loader = ForexDataLoader("data")
    
    # Get available symbols
    available_symbols = data_loader.get_available_symbols()
    logger.info(f"Available symbols: {len(available_symbols)}")
    
    # Load unified data
    returns_data = data_loader.load_unified_returns()
    prices_data = data_loader.load_unified_prices()
    
    if returns_data is None:
        logger.error("No returns data available")
        return
    
    # Data quality check
    logger.info("Performing data quality checks...")
    data_summary = {
        'n_assets': len(returns_data.columns),
        'n_days': len(returns_data),
        'date_range': f"{returns_data.index.min()} to {returns_data.index.max()}",
        'missing_data_pct': returns_data.isnull().sum().sum() / (len(returns_data) * len(returns_data.columns)) * 100
    }
    
    logger.info(f"Data Summary: {data_summary}")
    
    # Step 2: Parameter optimization
    logger.info("Step 2: Parameter Optimization")
    optimizer = ParameterOptimizer(optimization_metric='sharpe_ratio')
    
    parameter_grid = {
        'n_worst_performers': [3, 5, 7],
        'lookback_days': [15, 20, 30]
    }
    
    # Quick optimization on recent data
    opt_results = optimizer.grid_search_optimization(
        data_loader=data_loader,
        signal_generator_class=ConrarianSignalGenerator,
        backtesting_engine_class=BacktestingEngine,
        parameter_grid=parameter_grid,
        start_date="2020-01-01",
        end_date="2023-12-31"
    )
    
    optimal_params = opt_results['best_parameters']
    logger.info(f"Optimal parameters: {optimal_params}")
    
    # Step 3: Full backtest with optimal parameters
    logger.info("Step 3: Full Backtest with Optimal Parameters")
    
    signal_generator = ConrarianSignalGenerator(**optimal_params)
    backtesting_engine = BacktestingEngine(
        initial_capital=1000000,
        transaction_cost_bps=2.0,
        slippage_bps=0.5
    )
    performance_analyzer = PerformanceAnalyzer()
    
    # Use longer period for final backtest
    backtest_start = "2010-01-01"
    backtest_end = "2023-12-31"
    
    returns_full = data_loader.get_data_for_period(backtest_start, backtest_end, data_type='returns')
    prices_full = data_loader.get_data_for_period(backtest_start, backtest_end, data_type='prices')
    
    # Generate signals
    signal_output = signal_generator.generate_signals(prices_full, returns_full)
    
    # Run backtest
    final_results = backtesting_engine.run_backtest(
        signals=signal_output['weights'],
        returns=returns_full,
        start_date=backtest_start,
        end_date=backtest_end
    )
    
    # Step 4: Comprehensive performance analysis
    logger.info("Step 4: Comprehensive Performance Analysis")
    
    performance_report = performance_analyzer.generate_performance_report(
        backtest_results=final_results,
        output_dir="results/comprehensive_analysis"
    )
    
    # Print comprehensive results
    summary = performance_report['summary']['key_metrics']
    risk_metrics = performance_report['risk_metrics']
    
    logger.info("="*40)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("="*40)
    logger.info(f"Period: {backtest_start} to {backtest_end}")
    logger.info(f"Strategy Parameters: {optimal_params}")
    logger.info("Performance Metrics:")
    logger.info(f"  Total Return: {summary['total_return']*100:.2f}%")
    logger.info(f"  Annualized Return: {summary['annualized_return']*100:.2f}%")
    logger.info(f"  Volatility: {summary['volatility']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {summary['sharpe_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
    logger.info(f"  Calmar Ratio: {summary['calmar_ratio']:.3f}")
    
    logger.info("Risk Metrics:")
    logger.info(f"  VaR (5%): {risk_metrics.get('var_5_percent', 0)*100:.2f}%")
    logger.info(f"  Expected Shortfall (5%): {risk_metrics.get('expected_shortfall_5', 0)*100:.2f}%")
    logger.info(f"  Tail Ratio: {risk_metrics.get('tail_ratio', 0):.2f}")
    
    # Step 5: Save comprehensive results
    logger.info("Step 5: Saving Results")
    
    results_manager = ResultsManager("results")
    
    final_result_id = results_manager.save_backtest_results(
        backtest_results=final_results,
        strategy_name="ConrarianFX_Optimized",
        parameters=optimal_params,
        description=f"Optimized contrarian forex strategy ({backtest_start} to {backtest_end})",
        tags=['contrarian', 'forex', 'optimized', 'comprehensive']
    )
    
    # Export results
    export_file = results_manager.export_results(
        result_id=final_result_id,
        export_format='excel',
        output_path=f"results/exports/comprehensive_results_{datetime.now().strftime('%Y%m%d')}"
    )
    
    logger.info(f"Final results saved with ID: {final_result_id}")
    logger.info(f"Results exported to: {export_file}")
    logger.info("="*60)
    logger.info("COMPREHENSIVE WORKFLOW COMPLETED")
    logger.info("="*60)
    
    return final_result_id, final_results, performance_report


def main():
    """
    Main function to run all examples
    """
    logger.info("Starting Contrarian Forex Backtesting Examples")
    logger.info(f"Script started at: {datetime.now()}")
    
    try:
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Run examples based on what's most important for demonstration
        
        # Example 1: Basic backtesting
        basic_result_id, basic_results, basic_performance = example_basic_backtesting()
        
        # Example 2: Parameter optimization
        opt_result_id, opt_results = example_parameter_optimization()
        
        # Example 3: Results comparison
        comparison_results = example_results_comparison()
        
        # Example 6: Comprehensive workflow (combines everything)
        comprehensive_id, comprehensive_results, comprehensive_performance = example_comprehensive_workflow()
        
        # Note: Examples 3, 4 (walk-forward and multi-objective) are computationally intensive
        # Uncomment below to run them
        # wf_results = example_walk_forward_analysis()
        # mo_results = example_multi_objective_optimization()
        
        logger.info("="*60)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Results saved in: {Path('results').absolute()}")
        logger.info("Key result IDs for reference:")
        logger.info(f"  Basic backtest: {basic_result_id}")
        logger.info(f"  Optimization: {opt_result_id}")
        logger.info(f"  Comprehensive: {comprehensive_id}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()