#!/usr/bin/env python3
"""
Parameter Optimization Framework for Contrarian Forex Strategy

This module provides comprehensive parameter optimization capabilities including
grid search, random search, Bayesian optimization, and walk-forward analysis.
It's designed to find optimal parameters while avoiding overfitting through
robust validation techniques.

Key Features:
- Multi-objective optimization (return vs risk vs drawdown)
- Walk-forward analysis for parameter stability
- Cross-validation with time series considerations
- Parallel processing for computational efficiency
- Overfitting detection and regularization
- Out-of-sample testing capabilities
- Statistical significance testing
- Parameter sensitivity analysis

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import pickle

# Optional advanced optimization libraries
try:
    from scipy.optimize import minimize
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import make_scorer
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some optimization methods will be disabled")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """
    Comprehensive parameter optimization framework for contrarian forex strategies.
    """
    
    def __init__(self,
                 optimization_metric: str = 'sharpe_ratio',
                 secondary_metrics: List[str] = ['max_drawdown', 'calmar_ratio'],
                 min_sample_periods: int = 252,
                 validation_split: float = 0.3,
                 walk_forward_periods: int = 63,  # ~3 months
                 n_jobs: int = None,
                 random_seed: int = 42):
        """
        Initialize the parameter optimizer.
        
        Args:
            optimization_metric: Primary metric to optimize
            secondary_metrics: Additional metrics to consider
            min_sample_periods: Minimum periods required for optimization
            validation_split: Fraction of data for validation
            walk_forward_periods: Period length for walk-forward analysis
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_seed: Random seed for reproducibility
        """
        self.optimization_metric = optimization_metric
        self.secondary_metrics = secondary_metrics
        self.min_sample_periods = min_sample_periods
        self.validation_split = validation_split
        self.walk_forward_periods = walk_forward_periods
        self.n_jobs = n_jobs or max(1, mp.cpu_count() - 1)
        self.random_seed = random_seed
        
        # Results storage
        self.optimization_results = {}
        self.parameter_history = []
        self.best_parameters = {}
        
        # Set random seed
        np.random.seed(random_seed)
        
        logger.info(f"Initialized ParameterOptimizer with {optimization_metric} as primary metric")
    
    def grid_search_optimization(self,
                               data_loader,
                               signal_generator_class,
                               backtesting_engine_class,
                               parameter_grid: Dict[str, List],
                               start_date: str,
                               end_date: str,
                               performance_analyzer_class = None) -> Dict[str, Any]:
        """
        Comprehensive grid search optimization with cross-validation.
        
        Args:
            data_loader: Data loader instance
            signal_generator_class: Signal generator class
            backtesting_engine_class: Backtesting engine class
            parameter_grid: Dictionary mapping parameter names to lists of values
            start_date: Optimization start date
            end_date: Optimization end date
            performance_analyzer_class: Optional performance analyzer class
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting grid search optimization with {len(list(itertools.product(*parameter_grid.values())))} parameter combinations")
        
        # Load data
        returns_data = data_loader.get_data_for_period(start_date, end_date, data_type='returns')
        prices_data = data_loader.get_data_for_period(start_date, end_date, data_type='prices')
        
        if returns_data is None or prices_data is None:
            raise ValueError("Failed to load data for optimization")
        
        # Create parameter combinations
        param_combinations = list(ParameterGrid(parameter_grid))
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Split data for validation
        split_point = int(len(returns_data) * (1 - self.validation_split))
        
        train_returns = returns_data.iloc[:split_point]
        train_prices = prices_data.iloc[:split_point]
        val_returns = returns_data.iloc[split_point:]
        val_prices = prices_data.iloc[split_point:]
        
        logger.info(f"Training period: {train_returns.index.min()} to {train_returns.index.max()}")
        logger.info(f"Validation period: {val_returns.index.min()} to {val_returns.index.max()}")
        
        # Prepare parallel processing
        optimization_func = partial(
            self._evaluate_parameter_combination,
            data=(train_returns, train_prices, val_returns, val_prices),
            signal_generator_class=signal_generator_class,
            backtesting_engine_class=backtesting_engine_class,
            performance_analyzer_class=performance_analyzer_class
        )
        
        # Execute parallel optimization
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_params = {
                executor.submit(optimization_func, params): params 
                for params in param_combinations
            }
            
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                try:
                    result = future.result()
                    result['parameters'] = params
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(param_combinations)} combinations")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters {params}: {str(e)}")
                    continue
        
        # Process results
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            raise ValueError("No valid optimization results obtained")
        
        # Find best parameters
        best_idx = self._find_best_parameters(results_df)
        best_result = results_df.iloc[best_idx]
        
        # Calculate parameter sensitivity
        sensitivity_analysis = self._analyze_parameter_sensitivity(results_df, parameter_grid)
        
        # Overfitting analysis
        overfitting_analysis = self._detect_overfitting(results_df)
        
        optimization_results = {
            'method': 'grid_search',
            'best_parameters': best_result['parameters'],
            'best_metrics': {
                'train_' + self.optimization_metric: best_result.get(f'train_{self.optimization_metric}', 0),
                'val_' + self.optimization_metric: best_result.get(f'val_{self.optimization_metric}', 0),
                **{f'train_{metric}': best_result.get(f'train_{metric}', 0) for metric in self.secondary_metrics},
                **{f'val_{metric}': best_result.get(f'val_{metric}', 0) for metric in self.secondary_metrics}
            },
            'all_results': results_df,
            'sensitivity_analysis': sensitivity_analysis,
            'overfitting_analysis': overfitting_analysis,
            'optimization_metadata': {
                'n_combinations_tested': len(results_df),
                'training_period': f"{train_returns.index.min()} to {train_returns.index.max()}",
                'validation_period': f"{val_returns.index.min()} to {val_returns.index.max()}",
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.optimization_results['grid_search'] = optimization_results
        self.best_parameters = best_result['parameters']
        
        logger.info(f"Grid search completed. Best {self.optimization_metric}: "
                   f"train={best_result.get(f'train_{self.optimization_metric}', 0):.4f}, "
                   f"val={best_result.get(f'val_{self.optimization_metric}', 0):.4f}")
        
        return optimization_results
    
    def walk_forward_optimization(self,
                                data_loader,
                                signal_generator_class,
                                backtesting_engine_class,
                                parameter_grid: Dict[str, List],
                                start_date: str,
                                end_date: str,
                                performance_analyzer_class = None) -> Dict[str, Any]:
        """
        Walk-forward optimization for parameter stability analysis.
        
        Args:
            data_loader: Data loader instance
            signal_generator_class: Signal generator class
            backtesting_engine_class: Backtesting engine class  
            parameter_grid: Dictionary mapping parameter names to lists of values
            start_date: Optimization start date
            end_date: Optimization end date
            performance_analyzer_class: Optional performance analyzer class
            
        Returns:
            Dictionary with walk-forward optimization results
        """
        logger.info("Starting walk-forward optimization analysis")
        
        # Load data
        returns_data = data_loader.get_data_for_period(start_date, end_date, data_type='returns')
        prices_data = data_loader.get_data_for_period(start_date, end_date, data_type='prices')
        
        if returns_data is None or prices_data is None:
            raise ValueError("Failed to load data for walk-forward optimization")
        
        # Create walk-forward periods
        total_periods = len(returns_data)
        min_train_periods = self.min_sample_periods
        walk_periods = []
        
        current_start = 0
        while current_start + min_train_periods + self.walk_forward_periods < total_periods:
            train_end = current_start + min_train_periods
            test_end = min(train_end + self.walk_forward_periods, total_periods)
            
            walk_periods.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_dates': (returns_data.index[current_start], returns_data.index[train_end-1]),
                'test_dates': (returns_data.index[train_end], returns_data.index[test_end-1])
            })
            
            current_start += self.walk_forward_periods
        
        logger.info(f"Created {len(walk_periods)} walk-forward periods")
        
        # Run optimization for each period
        walk_forward_results = []
        parameter_stability = {}
        
        for i, period in enumerate(walk_periods):
            logger.info(f"Processing walk-forward period {i+1}/{len(walk_periods)}: "
                       f"{period['train_dates'][0].strftime('%Y-%m-%d')} to {period['test_dates'][1].strftime('%Y-%m-%d')}")
            
            # Extract period data
            train_returns = returns_data.iloc[period['train_start']:period['train_end']]
            train_prices = prices_data.iloc[period['train_start']:period['train_end']]
            test_returns = returns_data.iloc[period['test_start']:period['test_end']]
            test_prices = prices_data.iloc[period['test_start']:period['test_end']]
            
            # Run mini grid search for this period
            period_results = []
            param_combinations = list(ParameterGrid(parameter_grid))
            
            for params in param_combinations:
                try:
                    result = self._evaluate_parameter_combination(
                        params, 
                        (train_returns, train_prices, test_returns, test_prices),
                        signal_generator_class,
                        backtesting_engine_class,
                        performance_analyzer_class
                    )
                    result['parameters'] = params
                    result['period_index'] = i
                    period_results.append(result)
                except Exception as e:
                    logger.warning(f"Error in walk-forward period {i}, params {params}: {str(e)}")
                    continue
            
            if not period_results:
                logger.warning(f"No valid results for walk-forward period {i}")
                continue
            
            # Find best parameters for this period
            period_df = pd.DataFrame(period_results)
            best_idx = self._find_best_parameters(period_df)
            best_result = period_df.iloc[best_idx]
            
            # Store results
            walk_forward_results.append({
                'period_index': i,
                'period_info': period,
                'best_parameters': best_result['parameters'],
                'best_metrics': {
                    'train_' + self.optimization_metric: best_result.get(f'train_{self.optimization_metric}', 0),
                    'val_' + self.optimization_metric: best_result.get(f'val_{self.optimization_metric}', 0)
                },
                'all_results': period_df
            })
            
            # Track parameter stability
            for param_name, param_value in best_result['parameters'].items():
                if param_name not in parameter_stability:
                    parameter_stability[param_name] = []
                parameter_stability[param_name].append(param_value)
        
        # Analyze parameter stability
        stability_metrics = {}
        for param_name, values in parameter_stability.items():
            if len(values) > 1:
                stability_metrics[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                    'min': min(values),
                    'max': max(values),
                    'range': max(values) - min(values),
                    'values': values
                }
        
        # Calculate out-of-sample performance consistency
        oos_metrics = [result['best_metrics'].get(f'val_{self.optimization_metric}', 0) 
                      for result in walk_forward_results]
        
        performance_stability = {
            'mean_oos_performance': np.mean(oos_metrics),
            'std_oos_performance': np.std(oos_metrics),
            'cv_oos_performance': np.std(oos_metrics) / np.mean(oos_metrics) if np.mean(oos_metrics) != 0 else float('inf'),
            'min_oos_performance': min(oos_metrics) if oos_metrics else 0,
            'max_oos_performance': max(oos_metrics) if oos_metrics else 0,
            'positive_periods': sum(1 for x in oos_metrics if x > 0),
            'total_periods': len(oos_metrics)
        }
        
        walk_forward_optimization_results = {
            'method': 'walk_forward',
            'walk_forward_results': walk_forward_results,
            'parameter_stability': stability_metrics,
            'performance_stability': performance_stability,
            'recommended_parameters': self._get_stable_parameters(stability_metrics, parameter_stability),
            'metadata': {
                'n_periods': len(walk_forward_results),
                'walk_forward_length': self.walk_forward_periods,
                'min_training_periods': min_train_periods,
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.optimization_results['walk_forward'] = walk_forward_optimization_results
        
        logger.info(f"Walk-forward optimization completed. Performance stability CV: "
                   f"{performance_stability['cv_oos_performance']:.3f}")
        
        return walk_forward_optimization_results
    
    def bayesian_optimization(self,
                            data_loader,
                            signal_generator_class,
                            backtesting_engine_class,
                            parameter_space: Dict[str, Tuple],
                            start_date: str,
                            end_date: str,
                            n_calls: int = 50,
                            n_initial_points: int = 10,
                            performance_analyzer_class = None) -> Dict[str, Any]:
        """
        Bayesian optimization for efficient parameter search.
        
        Args:
            data_loader: Data loader instance
            signal_generator_class: Signal generator class
            backtesting_engine_class: Backtesting engine class
            parameter_space: Dictionary mapping parameter names to (min, max) tuples
            start_date: Optimization start date
            end_date: Optimization end date
            n_calls: Number of optimization calls
            n_initial_points: Number of initial random points
            performance_analyzer_class: Optional performance analyzer class
            
        Returns:
            Dictionary with Bayesian optimization results
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not available. Install with: pip install scikit-optimize")
        
        logger.info(f"Starting Bayesian optimization with {n_calls} calls")
        
        # Load data
        returns_data = data_loader.get_data_for_period(start_date, end_date, data_type='returns')
        prices_data = data_loader.get_data_for_period(start_date, end_date, data_type='prices')
        
        if returns_data is None or prices_data is None:
            raise ValueError("Failed to load data for Bayesian optimization")
        
        # Split data
        split_point = int(len(returns_data) * (1 - self.validation_split))
        train_returns = returns_data.iloc[:split_point]
        train_prices = prices_data.iloc[:split_point]
        val_returns = returns_data.iloc[split_point:]
        val_prices = prices_data.iloc[split_point:]
        
        # Define optimization space
        param_names = list(parameter_space.keys())
        dimensions = []
        
        for param_name, (min_val, max_val) in parameter_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                dimensions.append(Integer(min_val, max_val, name=param_name))
            else:
                dimensions.append(Real(min_val, max_val, name=param_name))
        
        # Objective function
        def objective(params):
            param_dict = {name: value for name, value in zip(param_names, params)}
            
            try:
                result = self._evaluate_parameter_combination(
                    param_dict,
                    (train_returns, train_prices, val_returns, val_prices),
                    signal_generator_class,
                    backtesting_engine_class,
                    performance_analyzer_class
                )
                
                # Return negative value for minimization (we want to maximize)
                metric_value = result.get(f'val_{self.optimization_metric}', 0)
                return -metric_value
                
            except Exception as e:
                logger.warning(f"Error evaluating Bayesian optimization point: {str(e)}")
                return 0.0  # Return neutral value on error
        
        # Run Bayesian optimization
        bayesian_result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func='EI',  # Expected Improvement
            random_state=self.random_seed
        )
        
        # Extract best parameters
        best_params = {name: value for name, value in zip(param_names, bayesian_result.x)}
        
        # Evaluate best parameters one more time for detailed metrics
        best_result = self._evaluate_parameter_combination(
            best_params,
            (train_returns, train_prices, val_returns, val_prices),
            signal_generator_class,
            backtesting_engine_class,
            performance_analyzer_class
        )
        
        # Analyze convergence
        convergence_analysis = self._analyze_bayesian_convergence(bayesian_result)
        
        bayesian_optimization_results = {
            'method': 'bayesian',
            'best_parameters': best_params,
            'best_metrics': {
                'train_' + self.optimization_metric: best_result.get(f'train_{self.optimization_metric}', 0),
                'val_' + self.optimization_metric: best_result.get(f'val_{self.optimization_metric}', 0),
                **{f'train_{metric}': best_result.get(f'train_{metric}', 0) for metric in self.secondary_metrics},
                **{f'val_{metric}': best_result.get(f'val_{metric}', 0) for metric in self.secondary_metrics}
            },
            'optimization_history': {
                'func_vals': bayesian_result.func_vals,
                'x_iters': bayesian_result.x_iters,
                'best_value_history': np.minimum.accumulate(bayesian_result.func_vals)
            },
            'convergence_analysis': convergence_analysis,
            'metadata': {
                'n_calls': n_calls,
                'n_initial_points': n_initial_points,
                'parameter_space': parameter_space,
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.optimization_results['bayesian'] = bayesian_optimization_results
        
        logger.info(f"Bayesian optimization completed. Best {self.optimization_metric}: "
                   f"val={best_result.get(f'val_{self.optimization_metric}', 0):.4f}")
        
        return bayesian_optimization_results
    
    def multi_objective_optimization(self,
                                   data_loader,
                                   signal_generator_class,
                                   backtesting_engine_class,
                                   parameter_grid: Dict[str, List],
                                   start_date: str,
                                   end_date: str,
                                   objectives: List[str] = None,
                                   objective_weights: List[float] = None,
                                   performance_analyzer_class = None) -> Dict[str, Any]:
        """
        Multi-objective optimization for balanced performance.
        
        Args:
            data_loader: Data loader instance
            signal_generator_class: Signal generator class
            backtesting_engine_class: Backtesting engine class
            parameter_grid: Dictionary mapping parameter names to lists of values
            start_date: Optimization start date
            end_date: Optimization end date
            objectives: List of objectives to optimize
            objective_weights: Weights for each objective
            performance_analyzer_class: Optional performance analyzer class
            
        Returns:
            Dictionary with multi-objective optimization results
        """
        if objectives is None:
            objectives = [self.optimization_metric] + self.secondary_metrics
        
        if objective_weights is None:
            objective_weights = [1.0] + [0.5] * len(self.secondary_metrics)
        
        if len(objectives) != len(objective_weights):
            raise ValueError("Number of objectives must match number of weights")
        
        logger.info(f"Starting multi-objective optimization with objectives: {objectives}")
        
        # Run standard grid search first
        grid_results = self.grid_search_optimization(
            data_loader, signal_generator_class, backtesting_engine_class,
            parameter_grid, start_date, end_date, performance_analyzer_class
        )
        
        results_df = grid_results['all_results']
        
        # Calculate composite scores
        composite_scores = []
        for _, row in results_df.iterrows():
            score = 0.0
            for objective, weight in zip(objectives, objective_weights):
                # Use validation metric
                val_metric = f'val_{objective}'
                if val_metric in row:
                    # Normalize and weight the objective
                    if objective == 'max_drawdown':
                        # For drawdown, lower is better (convert to positive contribution)
                        normalized_value = 1.0 - abs(row[val_metric])
                    else:
                        # For return metrics, higher is better
                        normalized_value = row[val_metric]
                    
                    score += weight * normalized_value
            
            composite_scores.append(score)
        
        results_df['composite_score'] = composite_scores
        
        # Find best parameters based on composite score
        best_idx = results_df['composite_score'].idxmax()
        best_result = results_df.iloc[best_idx]
        
        # Pareto frontier analysis
        pareto_analysis = self._analyze_pareto_frontier(results_df, objectives)
        
        multi_objective_results = {
            'method': 'multi_objective',
            'objectives': objectives,
            'objective_weights': objective_weights,
            'best_parameters': best_result['parameters'],
            'best_composite_score': best_result['composite_score'],
            'best_metrics': {
                f'val_{obj}': best_result.get(f'val_{obj}', 0) for obj in objectives
            },
            'pareto_frontier': pareto_analysis,
            'all_results_with_scores': results_df,
            'metadata': {
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.optimization_results['multi_objective'] = multi_objective_results
        
        logger.info(f"Multi-objective optimization completed. Best composite score: {best_result['composite_score']:.4f}")
        
        return multi_objective_results
    
    def cross_validation_optimization(self,
                                    data_loader,
                                    signal_generator_class,
                                    backtesting_engine_class,
                                    parameter_grid: Dict[str, List],
                                    start_date: str,
                                    end_date: str,
                                    n_splits: int = 5,
                                    performance_analyzer_class = None) -> Dict[str, Any]:
        """
        Time series cross-validation optimization.
        
        Args:
            data_loader: Data loader instance
            signal_generator_class: Signal generator class
            backtesting_engine_class: Backtesting engine class
            parameter_grid: Dictionary mapping parameter names to lists of values
            start_date: Optimization start date
            end_date: Optimization end date
            n_splits: Number of CV splits
            performance_analyzer_class: Optional performance analyzer class
            
        Returns:
            Dictionary with cross-validation optimization results
        """
        logger.info(f"Starting {n_splits}-fold cross-validation optimization")
        
        # Load data
        returns_data = data_loader.get_data_for_period(start_date, end_date, data_type='returns')
        prices_data = data_loader.get_data_for_period(start_date, end_date, data_type='prices')
        
        if returns_data is None or prices_data is None:
            raise ValueError("Failed to load data for cross-validation")
        
        # Create time series splits
        total_periods = len(returns_data)
        split_size = total_periods // n_splits
        min_train_size = self.min_sample_periods
        
        cv_splits = []
        for i in range(n_splits):
            # For time series CV, use expanding window
            train_end = min_train_size + (i + 1) * split_size
            train_end = min(train_end, total_periods)
            
            if train_end <= min_train_size:
                continue
                
            test_start = train_end
            test_end = min(test_start + split_size, total_periods)
            
            if test_end <= test_start:
                continue
            
            cv_splits.append({
                'fold': i,
                'train_start': 0,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
        
        logger.info(f"Created {len(cv_splits)} cross-validation splits")
        
        # Test each parameter combination across all folds
        param_combinations = list(ParameterGrid(parameter_grid))
        cv_results = []
        
        for params in param_combinations:
            fold_results = []
            
            for split in cv_splits:
                try:
                    # Extract fold data
                    train_returns = returns_data.iloc[split['train_start']:split['train_end']]
                    train_prices = prices_data.iloc[split['train_start']:split['train_end']]
                    test_returns = returns_data.iloc[split['test_start']:split['test_end']]
                    test_prices = prices_data.iloc[split['test_start']:split['test_end']]
                    
                    # Evaluate parameters on this fold
                    result = self._evaluate_parameter_combination(
                        params,
                        (train_returns, train_prices, test_returns, test_prices),
                        signal_generator_class,
                        backtesting_engine_class,
                        performance_analyzer_class
                    )
                    
                    result['fold'] = split['fold']
                    fold_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error in CV fold {split['fold']} for params {params}: {str(e)}")
                    continue
            
            if fold_results:
                # Calculate average performance across folds
                avg_metrics = {}
                for key in fold_results[0].keys():
                    if key != 'fold' and isinstance(fold_results[0][key], (int, float)):
                        values = [r[key] for r in fold_results if key in r and not np.isnan(r[key])]
                        if values:
                            avg_metrics[key] = np.mean(values)
                            avg_metrics[f'{key}_std'] = np.std(values)
                
                cv_result = {
                    'parameters': params,
                    'n_folds': len(fold_results),
                    'fold_results': fold_results,
                    **avg_metrics
                }
                cv_results.append(cv_result)
        
        # Find best parameters based on CV performance
        cv_results_df = pd.DataFrame([
            {**r['parameters'], 
             f'cv_{self.optimization_metric}': r.get(f'val_{self.optimization_metric}', 0),
             f'cv_{self.optimization_metric}_std': r.get(f'val_{self.optimization_metric}_std', 0)}
            for r in cv_results
        ])
        
        best_idx = cv_results_df[f'cv_{self.optimization_metric}'].idxmax()
        best_cv_result = cv_results[best_idx]
        
        cross_validation_results = {
            'method': 'cross_validation',
            'n_splits': len(cv_splits),
            'best_parameters': best_cv_result['parameters'],
            'best_cv_score': best_cv_result.get(f'val_{self.optimization_metric}', 0),
            'best_cv_std': best_cv_result.get(f'val_{self.optimization_metric}_std', 0),
            'all_cv_results': cv_results,
            'cv_summary': cv_results_df,
            'metadata': {
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        self.optimization_results['cross_validation'] = cross_validation_results
        
        logger.info(f"Cross-validation optimization completed. "
                   f"Best CV {self.optimization_metric}: {best_cv_result.get(f'val_{self.optimization_metric}', 0):.4f} "
                   f"± {best_cv_result.get(f'val_{self.optimization_metric}_std', 0):.4f}")
        
        return cross_validation_results
    
    def statistical_significance_test(self,
                                    results_df: pd.DataFrame,
                                    baseline_params: Dict[str, Any],
                                    confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Test statistical significance of parameter improvements.
        
        Args:
            results_df: DataFrame with optimization results
            baseline_params: Baseline parameter set for comparison
            confidence_level: Statistical significance level
            
        Returns:
            Dictionary with significance test results
        """
        from scipy.stats import ttest_1samp, mannwhitneyu
        
        logger.info("Performing statistical significance testing")
        
        # Find baseline performance
        baseline_mask = True
        for param, value in baseline_params.items():
            if param in results_df.columns:
                baseline_mask &= (results_df[param] == value)
        
        if not baseline_mask.any():
            logger.warning("Baseline parameters not found in results")
            return {'error': 'Baseline parameters not found'}
        
        baseline_performance = results_df.loc[baseline_mask, f'val_{self.optimization_metric}'].iloc[0]
        all_performances = results_df[f'val_{self.optimization_metric}'].values
        
        # One-sample t-test against baseline
        t_stat, t_pvalue = ttest_1samp(all_performances, baseline_performance)
        
        # Mann-Whitney U test for non-parametric comparison
        better_performances = all_performances[all_performances > baseline_performance]
        if len(better_performances) > 1:
            u_stat, u_pvalue = mannwhitneyu(better_performances, [baseline_performance] * len(better_performances))
        else:
            u_stat, u_pvalue = np.nan, 1.0
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(all_performances) - baseline_performance) / np.std(all_performances)
        
        # Practical significance
        improvement_threshold = 0.05  # 5% improvement threshold
        practical_improvement = (np.mean(all_performances) - baseline_performance) / abs(baseline_performance)
        is_practically_significant = abs(practical_improvement) > improvement_threshold
        
        significance_results = {
            'baseline_performance': baseline_performance,
            'mean_performance': np.mean(all_performances),
            'performance_improvement': np.mean(all_performances) - baseline_performance,
            'relative_improvement': practical_improvement,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue,
            'effect_size_cohens_d': effect_size,
            'is_statistically_significant': t_pvalue < confidence_level,
            'is_practically_significant': is_practically_significant,
            'confidence_level': confidence_level,
            'n_parameter_combinations': len(all_performances),
            'fraction_better_than_baseline': (all_performances > baseline_performance).mean()
        }
        
        logger.info(f"Statistical significance test completed. "
                   f"p-value: {t_pvalue:.4f}, Effect size: {effect_size:.4f}")
        
        return significance_results
    
    def save_optimization_results(self, 
                                output_dir: str,
                                prefix: str = "optimization") -> Dict[str, str]:
        """
        Save optimization results to files.
        
        Args:
            output_dir: Directory to save results
            prefix: Filename prefix
            
        Returns:
            Dictionary mapping result types to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save each optimization method's results
        for method, results in self.optimization_results.items():
            method_file = output_path / f"{prefix}_{method}_{timestamp}.json"
            
            # Prepare data for JSON serialization
            serializable_results = self._prepare_for_json(results)
            
            with open(method_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            saved_files[method] = str(method_file)
            
            # Save detailed results as parquet if available
            if 'all_results' in results and isinstance(results['all_results'], pd.DataFrame):
                parquet_file = output_path / f"{prefix}_{method}_detailed_{timestamp}.parquet"
                results['all_results'].to_parquet(parquet_file)
                saved_files[f'{method}_detailed'] = str(parquet_file)
        
        # Save best parameters summary
        summary_file = output_path / f"{prefix}_summary_{timestamp}.json"
        summary = {
            'best_parameters_by_method': {
                method: results.get('best_parameters', {})
                for method, results in self.optimization_results.items()
            },
            'optimization_summary': {
                method: {
                    'best_metric_value': results.get('best_metrics', {}).get(f'val_{self.optimization_metric}', 0),
                    'optimization_date': results.get('metadata', {}).get('optimization_date', ''),
                    'method_specific_info': self._extract_method_summary(results)
                }
                for method, results in self.optimization_results.items()
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        saved_files['summary'] = str(summary_file)
        
        logger.info(f"Optimization results saved to {output_dir}")
        return saved_files
    
    # Helper methods
    def _evaluate_parameter_combination(self,
                                      params: Dict[str, Any],
                                      data: Tuple,
                                      signal_generator_class,
                                      backtesting_engine_class,
                                      performance_analyzer_class = None) -> Dict[str, float]:
        """Evaluate a single parameter combination."""
        train_returns, train_prices, val_returns, val_prices = data
        
        try:
            # Create signal generator with parameters
            signal_generator = signal_generator_class(**params)
            
            # Generate signals on training data
            train_signals = signal_generator.generate_signals(train_prices, train_returns)
            
            # Backtest on training data
            backtester = backtesting_engine_class()
            train_results = backtester.run_backtest(
                train_signals['weights'], train_returns
            )
            
            # Calculate training metrics
            train_stats = backtester.get_portfolio_statistics(train_results)
            
            # Backtest on validation data with same signal generator
            val_signals = signal_generator.generate_signals(val_prices, val_returns)
            val_results = backtester.run_backtest(
                val_signals['weights'], val_returns
            )
            
            # Calculate validation metrics
            val_stats = backtester.get_portfolio_statistics(val_results)
            
            # Combine results
            result = {}
            for key, value in train_stats.items():
                result[f'train_{key}'] = value
            for key, value in val_stats.items():
                result[f'val_{key}'] = value
            
            return result
            
        except Exception as e:
            logger.warning(f"Error evaluating parameters {params}: {str(e)}")
            # Return zero metrics on error
            return {f'train_{self.optimization_metric}': 0.0, f'val_{self.optimization_metric}': 0.0}
    
    def _find_best_parameters(self, results_df: pd.DataFrame) -> int:
        """Find index of best parameter combination."""
        # Primary optimization on validation metric
        primary_metric = f'val_{self.optimization_metric}'
        
        if primary_metric not in results_df.columns:
            logger.warning(f"Primary metric {primary_metric} not found, using first available metric")
            primary_metric = results_df.select_dtypes(include=[np.number]).columns[0]
        
        # Handle different metric types
        if self.optimization_metric in ['max_drawdown']:
            # For drawdown, we want the smallest absolute value (closest to 0)
            best_idx = results_df[primary_metric].abs().idxmin()
        else:
            # For return metrics, we want the maximum
            best_idx = results_df[primary_metric].idxmax()
        
        return best_idx
    
    def _analyze_parameter_sensitivity(self, 
                                     results_df: pd.DataFrame, 
                                     parameter_grid: Dict[str, List]) -> Dict[str, Any]:
        """Analyze parameter sensitivity."""
        sensitivity_analysis = {}
        
        for param_name, param_values in parameter_grid.items():
            if param_name not in results_df.columns:
                continue
            
            param_performance = {}
            for value in param_values:
                mask = results_df[param_name] == value
                if mask.any():
                    performance_values = results_df.loc[mask, f'val_{self.optimization_metric}']
                    param_performance[str(value)] = {
                        'mean': performance_values.mean(),
                        'std': performance_values.std(),
                        'min': performance_values.min(),
                        'max': performance_values.max(),
                        'count': len(performance_values)
                    }
            
            # Calculate sensitivity metric (range of performance across parameter values)
            mean_performances = [stats['mean'] for stats in param_performance.values()]
            sensitivity_range = max(mean_performances) - min(mean_performances) if mean_performances else 0
            
            sensitivity_analysis[param_name] = {
                'performance_by_value': param_performance,
                'sensitivity_range': sensitivity_range,
                'optimal_value': max(param_performance.keys(), 
                                   key=lambda x: param_performance[x]['mean']) if param_performance else None
            }
        
        return sensitivity_analysis
    
    def _detect_overfitting(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential overfitting in optimization results."""
        train_metric = f'train_{self.optimization_metric}'
        val_metric = f'val_{self.optimization_metric}'
        
        if train_metric not in results_df.columns or val_metric not in results_df.columns:
            return {'error': 'Required metrics not available for overfitting analysis'}
        
        # Calculate train-validation gap
        performance_gap = results_df[train_metric] - results_df[val_metric]
        
        # Overfitting indicators
        mean_gap = performance_gap.mean()
        std_gap = performance_gap.std()
        max_gap = performance_gap.max()
        correlation = results_df[train_metric].corr(results_df[val_metric])
        
        # Classify overfitting severity
        overfitting_threshold = 0.1  # 10% performance gap threshold
        severe_overfitting = (performance_gap > overfitting_threshold).mean()
        
        overfitting_analysis = {
            'mean_train_val_gap': mean_gap,
            'std_train_val_gap': std_gap,
            'max_train_val_gap': max_gap,
            'train_val_correlation': correlation,
            'fraction_severe_overfitting': severe_overfitting,
            'overfitting_risk': 'HIGH' if severe_overfitting > 0.3 or mean_gap > overfitting_threshold 
                               else 'MODERATE' if severe_overfitting > 0.1 or mean_gap > overfitting_threshold/2
                               else 'LOW'
        }
        
        return overfitting_analysis
    
    def _analyze_bayesian_convergence(self, bayesian_result) -> Dict[str, Any]:
        """Analyze convergence of Bayesian optimization."""
        func_vals = bayesian_result.func_vals
        
        # Calculate improvement over iterations
        best_vals = np.minimum.accumulate(func_vals)
        improvements = np.diff(best_vals)
        
        # Find when optimization converged (no improvement for N iterations)
        convergence_window = 10
        converged_at = len(func_vals)
        
        for i in range(convergence_window, len(improvements)):
            if np.all(improvements[i-convergence_window:i] >= -1e-6):  # No significant improvement
                converged_at = i
                break
        
        convergence_analysis = {
            'converged_at_iteration': converged_at,
            'final_improvement_rate': abs(improvements[-5:].mean()) if len(improvements) >= 5 else 0,
            'total_improvement': abs(best_vals[-1] - best_vals[0]),
            'best_value_history': best_vals.tolist(),
            'convergence_achieved': converged_at < len(func_vals)
        }
        
        return convergence_analysis
    
    def _analyze_pareto_frontier(self, 
                               results_df: pd.DataFrame, 
                               objectives: List[str]) -> Dict[str, Any]:
        """Analyze Pareto frontier for multi-objective optimization."""
        if len(objectives) < 2:
            return {'error': 'At least 2 objectives required for Pareto analysis'}
        
        # Extract objective values (use validation metrics)
        objective_values = []
        for obj in objectives:
            val_col = f'val_{obj}'
            if val_col in results_df.columns:
                objective_values.append(results_df[val_col].values)
            else:
                return {'error': f'Objective {obj} not found in results'}
        
        objective_matrix = np.array(objective_values).T
        
        # Find Pareto optimal solutions
        pareto_mask = self._is_pareto_optimal(objective_matrix)
        pareto_solutions = results_df[pareto_mask]
        
        pareto_analysis = {
            'n_pareto_optimal': pareto_mask.sum(),
            'pareto_fraction': pareto_mask.mean(),
            'pareto_solutions': pareto_solutions.index.tolist(),
            'pareto_objectives': {
                obj: pareto_solutions[f'val_{obj}'].tolist() 
                for obj in objectives if f'val_{obj}' in pareto_solutions.columns
            }
        }
        
        return pareto_analysis
    
    def _is_pareto_optimal(self, costs: np.ndarray) -> np.ndarray:
        """Find Pareto optimal solutions."""
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                # For maximization objectives, we want to find points where no other point is better in all objectives
                dominated = np.all(costs >= c, axis=1) & np.any(costs > c, axis=1)
                is_efficient[dominated] = False
        
        return is_efficient
    
    def _get_stable_parameters(self, 
                             stability_metrics: Dict[str, Dict], 
                             parameter_history: Dict[str, List]) -> Dict[str, Any]:
        """Get recommended parameters based on stability analysis."""
        stable_params = {}
        
        for param_name, metrics in stability_metrics.items():
            # Use parameter value with lowest coefficient of variation
            if metrics['cv'] < 0.5:  # Reasonable stability threshold
                stable_params[param_name] = {
                    'recommended_value': metrics['mean'],
                    'stability_cv': metrics['cv'],
                    'value_range': (metrics['min'], metrics['max']),
                    'is_stable': True
                }
            else:
                # Use most frequent value for unstable parameters
                values = parameter_history[param_name]
                unique_values, counts = np.unique(values, return_counts=True)
                most_frequent = unique_values[np.argmax(counts)]
                
                stable_params[param_name] = {
                    'recommended_value': most_frequent,
                    'stability_cv': metrics['cv'],
                    'value_range': (metrics['min'], metrics['max']),
                    'is_stable': False,
                    'most_frequent_value': most_frequent
                }
        
        return stable_params
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj) or obj is None or obj is np.nan:
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _extract_method_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract method-specific summary information."""
        method = results.get('method', 'unknown')
        
        if method == 'grid_search':
            return {
                'n_combinations_tested': results.get('optimization_metadata', {}).get('n_combinations_tested', 0),
                'overfitting_risk': results.get('overfitting_analysis', {}).get('overfitting_risk', 'UNKNOWN')
            }
        elif method == 'walk_forward':
            return {
                'n_periods': results.get('metadata', {}).get('n_periods', 0),
                'performance_cv': results.get('performance_stability', {}).get('cv_oos_performance', 0)
            }
        elif method == 'bayesian':
            return {
                'n_calls': results.get('metadata', {}).get('n_calls', 0),
                'convergence_achieved': results.get('convergence_analysis', {}).get('convergence_achieved', False)
            }
        else:
            return {}


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("ParameterOptimizer module loaded successfully")
    print("Module ready for use. Import and use with backtesting framework.")