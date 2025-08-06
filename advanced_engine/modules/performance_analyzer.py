#!/usr/bin/env python3
"""
Comprehensive Performance Analysis Module for Contrarian Forex Strategy

This module provides advanced performance analytics, risk metrics, and detailed
reporting capabilities for backtesting results. It includes statistical analysis,
risk decomposition, and performance attribution functionality.

Key Features:
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Risk analysis with VaR, CVaR, and tail risk measures
- Drawdown analysis with duration and recovery statistics
- Performance attribution and factor analysis
- Rolling performance analysis and stability metrics
- Benchmark comparison and relative performance
- Monte Carlo simulation for robustness testing

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def calculate_rolling_sharpe_numba(returns: np.ndarray, window: int, risk_free_rate: float = 0.0) -> np.ndarray:
    """
    Numba-optimized rolling Sharpe ratio calculation.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Array of rolling Sharpe ratios
    """
    n = len(returns)
    sharpe_ratios = np.full(n, np.nan)
    daily_rf = risk_free_rate / 252.0
    
    for i in range(window-1, n):
        window_returns = returns[i-window+1:i+1]
        excess_returns = window_returns - daily_rf
        
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)
        
        if std_excess > 0:
            sharpe_ratios[i] = mean_excess / std_excess * np.sqrt(252)
        else:
            sharpe_ratios[i] = 0.0
    
    return sharpe_ratios


@nb.jit(nopython=True, cache=True)
def calculate_var_cvar_numba(returns: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """
    Numba-optimized VaR and CVaR calculation.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.05 for 5% VaR)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    if len(returns) == 0:
        return 0.0, 0.0
    
    sorted_returns = np.sort(returns)
    var_index = int(len(sorted_returns) * confidence_level)
    
    if var_index >= len(sorted_returns):
        var_index = len(sorted_returns) - 1
    elif var_index < 0:
        var_index = 0
    
    var = sorted_returns[var_index]
    
    # CVaR is the average of returns below VaR
    tail_returns = sorted_returns[:var_index+1]
    cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    return var, cvar


@nb.jit(nopython=True, cache=True)
def calculate_maximum_adverse_excursion(returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Calculate Maximum Adverse Excursion (MAE) for position analysis.
    
    Args:
        returns: Array of returns
        positions: Array of position indicators (1 for long, 0 for flat, -1 for short)
        
    Returns:
        Array of MAE values
    """
    n = len(returns)
    mae = np.zeros(n)
    
    current_position_start = 0
    current_return = 0.0
    min_return = 0.0
    
    for i in range(n):
        if positions[i] != 0:  # In position
            if current_position_start == 0 or i == 0:
                current_position_start = i
                current_return = 0.0
                min_return = 0.0
            
            current_return += returns[i] * positions[i]
            min_return = min(min_return, current_return)
            mae[i] = abs(min_return)
            
        else:  # Flat position
            current_position_start = 0
            current_return = 0.0
            min_return = 0.0
            mae[i] = 0.0
    
    return mae


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for backtesting results.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = [0.01, 0.05, 0.10],
                 benchmark_rate: float = 0.0):
        """
        Initialize the performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculations
            confidence_levels: VaR confidence levels to calculate
            benchmark_rate: Benchmark return rate for comparison
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        self.benchmark_rate = benchmark_rate
        
        # Results storage
        self.analysis_results = {}
        self.metrics_cache = {}
        
        logger.info(f"Initialized PerformanceAnalyzer with {risk_free_rate*100:.2f}% risk-free rate")
    
    def analyze_returns(self, 
                       returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive return analysis.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Optional benchmark return series
            
        Returns:
            Dictionary with comprehensive return metrics
        """
        logger.info("Analyzing portfolio returns")
        
        # Remove any NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            logger.warning("No valid returns for analysis")
            return {}
        
        # Basic statistics
        total_return = (1 + clean_returns).prod() - 1.0
        annualized_return = (1 + total_return) ** (252 / len(clean_returns)) - 1.0
        volatility = clean_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = clean_returns - self.risk_free_rate / 252.0
        sharpe_ratio = excess_returns.mean() / clean_returns.std() * np.sqrt(252) if clean_returns.std() > 0 else 0.0
        
        # Downside risk metrics
        negative_returns = clean_returns[clean_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        sortino_ratio = excess_returns.mean() / downside_vol * np.sqrt(252) if downside_vol > 0 else 0.0
        
        # Skewness and Kurtosis
        skewness = clean_returns.skew()
        kurtosis = clean_returns.kurtosis()
        
        # Win/Loss statistics
        winning_periods = (clean_returns > 0).sum()
        losing_periods = (clean_returns < 0).sum()
        flat_periods = (clean_returns == 0).sum()
        
        win_rate = winning_periods / len(clean_returns)
        avg_win = clean_returns[clean_returns > 0].mean() if winning_periods > 0 else 0.0
        avg_loss = clean_returns[clean_returns < 0].mean() if losing_periods > 0 else 0.0
        
        profit_factor = abs(avg_win * winning_periods / (avg_loss * losing_periods)) if avg_loss < 0 else float('inf')
        
        # VaR and CVaR calculations
        var_cvar_results = {}
        for conf_level in self.confidence_levels:
            var, cvar = calculate_var_cvar_numba(clean_returns.values, conf_level)
            var_cvar_results[f'VaR_{int(conf_level*100)}%'] = var
            var_cvar_results[f'CVaR_{int(conf_level*100)}%'] = cvar
        
        # Compile results
        analysis_results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_volatility': downside_vol,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winning_periods': int(winning_periods),
            'losing_periods': int(losing_periods),
            'flat_periods': int(flat_periods),
            'total_periods': len(clean_returns),
            **var_cvar_results
        }
        
        # Benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_comparison(clean_returns, benchmark_returns)
            analysis_results.update(benchmark_metrics)
        
        logger.info(f"Return analysis complete: Sharpe={sharpe_ratio:.3f}, Sortino={sortino_ratio:.3f}")
        return analysis_results
    
    def analyze_drawdowns(self, 
                         portfolio_value: pd.Series,
                         returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis.
        
        Args:
            portfolio_value: Portfolio value series
            returns: Optional return series for additional analysis
            
        Returns:
            Dictionary with drawdown metrics and statistics
        """
        logger.info("Analyzing portfolio drawdowns")
        
        # Calculate running maximum (peaks)
        running_max = portfolio_value.expanding().max()
        drawdowns = (portfolio_value - running_max) / running_max
        
        # Basic drawdown statistics
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1]
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0.0
        
        # Drawdown periods analysis
        drawdown_periods = self._identify_drawdown_periods(drawdowns)
        
        # Maximum drawdown details
        max_dd_info = self._analyze_maximum_drawdown(portfolio_value, drawdowns)
        
        # Recovery analysis
        recovery_stats = self._analyze_recovery_periods(drawdowns, drawdown_periods)
        
        # Calmar ratio (if returns provided)
        calmar_ratio = 0.0
        if returns is not None:
            clean_returns = returns.dropna()
            if len(clean_returns) > 0 and max_drawdown < 0:
                annualized_return = (1 + clean_returns).prod() ** (252 / len(clean_returns)) - 1.0
                calmar_ratio = annualized_return / abs(max_drawdown)
        
        drawdown_results = {
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'avg_drawdown': avg_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_drawdown_periods': len(drawdown_periods),
            'total_drawdown_days': sum(period['duration'] for period in drawdown_periods),
            'avg_drawdown_duration': np.mean([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0,
            'max_drawdown_duration': max([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0,
            'drawdown_periods': drawdown_periods,
            **max_dd_info,
            **recovery_stats
        }
        
        logger.info(f"Drawdown analysis complete: MaxDD={max_drawdown*100:.2f}%, Current={current_drawdown*100:.2f}%")
        return drawdown_results
    
    def rolling_performance_analysis(self, 
                                   returns: pd.Series,
                                   windows: List[int] = [30, 60, 90, 252]) -> Dict[str, pd.DataFrame]:
        """
        Rolling performance analysis across multiple time windows.
        
        Args:
            returns: Portfolio return series
            windows: List of rolling window sizes (in days)
            
        Returns:
            Dictionary mapping window sizes to rolling metrics DataFrames
        """
        logger.info(f"Calculating rolling performance for {len(windows)} window sizes")
        
        clean_returns = returns.dropna()
        rolling_results = {}
        
        for window in windows:
            if len(clean_returns) < window:
                logger.warning(f"Insufficient data for {window}-day rolling analysis")
                continue
            
            # Rolling Sharpe ratio (using numba)
            rolling_sharpe = calculate_rolling_sharpe_numba(
                clean_returns.values, window, self.risk_free_rate
            )
            
            # Rolling volatility
            rolling_vol = clean_returns.rolling(window).std() * np.sqrt(252)
            
            # Rolling returns
            rolling_return = clean_returns.rolling(window).apply(
                lambda x: (1 + x).prod() ** (252/len(x)) - 1.0
            )
            
            # Rolling maximum drawdown
            rolling_max_dd = clean_returns.rolling(window).apply(
                lambda x: self._calculate_rolling_max_dd(x)
            )
            
            # Rolling win rate
            rolling_win_rate = clean_returns.rolling(window).apply(
                lambda x: (x > 0).sum() / len(x)
            )
            
            # Rolling VaR (5%)
            rolling_var = clean_returns.rolling(window).quantile(0.05)
            
            # Compile rolling metrics
            rolling_df = pd.DataFrame({
                'rolling_sharpe': rolling_sharpe,
                'rolling_return': rolling_return,
                'rolling_volatility': rolling_vol,
                'rolling_max_drawdown': rolling_max_dd,
                'rolling_win_rate': rolling_win_rate,
                'rolling_var_5%': rolling_var,
            }, index=clean_returns.index)
            
            rolling_results[f'{window}d'] = rolling_df
        
        logger.info("Rolling performance analysis complete")
        return rolling_results
    
    def performance_attribution(self, 
                              portfolio_returns: pd.Series,
                              asset_returns: pd.DataFrame,
                              portfolio_weights: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed performance attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            asset_returns: Individual asset returns
            portfolio_weights: Portfolio weights over time
            
        Returns:
            Dictionary with attribution analysis results
        """
        logger.info("Performing detailed performance attribution")
        
        # Align all data
        common_dates = portfolio_returns.index.intersection(asset_returns.index).intersection(portfolio_weights.index)
        
        portfolio_returns_aligned = portfolio_returns.reindex(common_dates)
        asset_returns_aligned = asset_returns.reindex(common_dates)
        weights_aligned = portfolio_weights.reindex(common_dates)
        
        # Calculate asset contributions
        asset_contributions = weights_aligned.shift(1) * asset_returns_aligned
        asset_contributions = asset_contributions.fillna(0.0)
        
        # Total attribution check
        calculated_portfolio_returns = asset_contributions.sum(axis=1)
        attribution_error = (portfolio_returns_aligned - calculated_portfolio_returns).abs().mean()
        
        # Asset-level attribution statistics
        asset_attribution_stats = {}
        for asset in asset_contributions.columns:
            contributions = asset_contributions[asset]
            asset_attribution_stats[asset] = {
                'total_contribution': contributions.sum(),
                'avg_contribution': contributions.mean(),
                'contribution_volatility': contributions.std(),
                'positive_contribution_days': (contributions > 0).sum(),
                'negative_contribution_days': (contributions < 0).sum(),
                'max_single_day_contribution': contributions.max(),
                'min_single_day_contribution': contributions.min(),
            }
        
        # Time-based attribution (monthly, quarterly)
        monthly_attribution = asset_contributions.resample('M').sum()
        quarterly_attribution = asset_contributions.resample('Q').sum()
        
        # Factor analysis using PCA
        factor_analysis = self._perform_factor_analysis(asset_returns_aligned, weights_aligned)
        
        attribution_results = {
            'asset_contributions': asset_contributions,
            'asset_attribution_stats': asset_attribution_stats,
            'monthly_attribution': monthly_attribution,
            'quarterly_attribution': quarterly_attribution,
            'attribution_error': attribution_error,
            'factor_analysis': factor_analysis
        }
        
        logger.info(f"Performance attribution complete: avg error={attribution_error:.6f}")
        return attribution_results
    
    def stress_testing(self, 
                      returns: pd.Series,
                      stress_scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Comprehensive stress testing and scenario analysis.
        
        Args:
            returns: Portfolio return series
            stress_scenarios: Custom stress scenarios (optional)
            
        Returns:
            Dictionary with stress test results
        """
        logger.info("Performing stress testing and scenario analysis")
        
        clean_returns = returns.dropna()
        
        # Default stress scenarios
        if stress_scenarios is None:
            stress_scenarios = {
                '2008_crisis': {'return_shock': -0.15, 'volatility_multiplier': 2.0},
                'covid_2020': {'return_shock': -0.12, 'volatility_multiplier': 1.8},
                'high_volatility': {'return_shock': 0.0, 'volatility_multiplier': 3.0},
                'deflation': {'return_shock': -0.08, 'volatility_multiplier': 1.5},
                'currency_crisis': {'return_shock': -0.20, 'volatility_multiplier': 2.5}
            }
        
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress scenario
            stressed_returns = self._apply_stress_scenario(clean_returns, scenario_params)
            
            # Calculate stressed metrics
            stressed_metrics = self.analyze_returns(stressed_returns)
            
            # Calculate portfolio value under stress
            stressed_portfolio_value = (1 + stressed_returns).cumprod()
            stressed_drawdown_analysis = self.analyze_drawdowns(stressed_portfolio_value, stressed_returns)
            
            stress_results[scenario_name] = {
                'scenario_params': scenario_params,
                'stressed_returns': stressed_returns,
                'metrics': stressed_metrics,
                'drawdown_analysis': stressed_drawdown_analysis
            }
        
        # Monte Carlo simulation
        monte_carlo_results = self._monte_carlo_simulation(clean_returns)
        stress_results['monte_carlo'] = monte_carlo_results
        
        logger.info(f"Stress testing complete for {len(stress_scenarios)} scenarios")
        return stress_results
    
    def generate_performance_report(self, 
                                  backtest_results: Dict[str, Any],
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            backtest_results: Results from backtesting engine
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with complete performance analysis
        """
        logger.info("Generating comprehensive performance report")
        
        portfolio_returns = backtest_results['portfolio_returns']
        portfolio_value = backtest_results['portfolio_value']
        portfolio_weights = backtest_results['portfolio_weights']
        asset_returns = backtest_results['asset_returns']
        
        # Core analyses
        return_analysis = self.analyze_returns(portfolio_returns)
        drawdown_analysis = self.analyze_drawdowns(portfolio_value, portfolio_returns)
        rolling_analysis = self.rolling_performance_analysis(portfolio_returns)
        attribution_analysis = self.performance_attribution(
            portfolio_returns, asset_returns, portfolio_weights
        )
        stress_analysis = self.stress_testing(portfolio_returns)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_returns)
        
        # Trading statistics
        trading_stats = self._calculate_trading_statistics(backtest_results)
        
        # Comprehensive report
        performance_report = {
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'period': f"{portfolio_returns.index.min()} to {portfolio_returns.index.max()}",
                'total_days': len(portfolio_returns),
                'key_metrics': {
                    'total_return': return_analysis.get('total_return', 0.0),
                    'annualized_return': return_analysis.get('annualized_return', 0.0),
                    'volatility': return_analysis.get('volatility', 0.0),
                    'sharpe_ratio': return_analysis.get('sharpe_ratio', 0.0),
                    'max_drawdown': drawdown_analysis.get('max_drawdown', 0.0),
                    'calmar_ratio': drawdown_analysis.get('calmar_ratio', 0.0),
                }
            },
            'return_analysis': return_analysis,
            'drawdown_analysis': drawdown_analysis,
            'rolling_analysis': rolling_analysis,
            'attribution_analysis': attribution_analysis,
            'stress_analysis': stress_analysis,
            'risk_metrics': risk_metrics,
            'trading_statistics': trading_stats,
            'metadata': backtest_results.get('metadata', {})
        }
        
        # Save to files if output directory specified
        if output_dir:
            saved_files = self._save_performance_report(performance_report, output_dir)
            performance_report['saved_files'] = saved_files
        
        logger.info("Performance report generation complete")
        return performance_report
    
    # Helper methods
    def _calculate_benchmark_comparison(self, 
                                      returns: pd.Series, 
                                      benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate metrics relative to benchmark."""
        # Align series
        aligned_data = pd.DataFrame({'portfolio': returns, 'benchmark': benchmark_returns}).dropna()
        
        if len(aligned_data) == 0:
            return {}
        
        # Excess returns
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        
        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        
        # Beta calculation
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
        benchmark_variance = np.var(aligned_data['benchmark'])
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Alpha calculation
        portfolio_return = (1 + aligned_data['portfolio']).prod() ** (252 / len(aligned_data)) - 1.0
        benchmark_return = (1 + aligned_data['benchmark']).prod() ** (252 / len(aligned_data)) - 1.0
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return {
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation_with_benchmark': aligned_data['portfolio'].corr(aligned_data['benchmark']),
            'excess_return_annualized': excess_returns.mean() * 252,
            'tracking_error': excess_returns.std() * np.sqrt(252)
        }
    
    def _identify_drawdown_periods(self, drawdowns: pd.Series) -> List[Dict]:
        """Identify individual drawdown periods."""
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        min_value = 0
        
        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                # Start of new drawdown period
                in_drawdown = True
                start_date = date
                min_value = dd
            elif dd < 0 and in_drawdown:
                # Continuing drawdown
                min_value = min(min_value, dd)
            elif dd >= 0 and in_drawdown:
                # End of drawdown period
                in_drawdown = False
                if start_date is not None:
                    drawdown_periods.append({
                        'start_date': start_date,
                        'end_date': date,
                        'duration': (date - start_date).days,
                        'max_drawdown': min_value
                    })
        
        # Handle case where backtest ends in drawdown
        if in_drawdown and start_date is not None:
            drawdown_periods.append({
                'start_date': start_date,
                'end_date': drawdowns.index[-1],
                'duration': (drawdowns.index[-1] - start_date).days,
                'max_drawdown': min_value
            })
        
        return drawdown_periods
    
    def _analyze_maximum_drawdown(self, 
                                portfolio_value: pd.Series, 
                                drawdowns: pd.Series) -> Dict[str, Any]:
        """Detailed analysis of maximum drawdown period."""
        max_dd_idx = drawdowns.idxmin()
        max_dd_value = drawdowns.min()
        
        # Find peak before max drawdown
        pre_dd_data = portfolio_value.loc[:max_dd_idx]
        peak_idx = pre_dd_data.idxmax()
        peak_value = pre_dd_data.max()
        
        # Find recovery date (if any)
        post_dd_data = portfolio_value.loc[max_dd_idx:]
        recovery_data = post_dd_data[post_dd_data >= peak_value]
        recovery_idx = recovery_data.index[0] if len(recovery_data) > 0 else None
        
        max_dd_info = {
            'max_dd_peak_date': peak_idx,
            'max_dd_trough_date': max_dd_idx,
            'max_dd_recovery_date': recovery_idx,
            'max_dd_peak_value': peak_value,
            'max_dd_trough_value': portfolio_value.loc[max_dd_idx],
            'max_dd_decline_days': (max_dd_idx - peak_idx).days if peak_idx != max_dd_idx else 0,
            'max_dd_recovery_days': (recovery_idx - max_dd_idx).days if recovery_idx else None,
            'max_dd_total_days': (recovery_idx - peak_idx).days if recovery_idx else None
        }
        
        return max_dd_info
    
    def _analyze_recovery_periods(self, 
                                drawdowns: pd.Series, 
                                drawdown_periods: List[Dict]) -> Dict[str, Any]:
        """Analyze recovery periods after drawdowns."""
        recovery_times = []
        
        for period in drawdown_periods:
            if 'end_date' in period:
                # Find when portfolio recovered to pre-drawdown level
                post_period = drawdowns.loc[period['end_date']:]
                recovery_point = post_period[post_period >= 0].index
                if len(recovery_point) > 0:
                    recovery_days = (recovery_point[0] - period['start_date']).days
                    recovery_times.append(recovery_days)
        
        recovery_stats = {
            'avg_recovery_days': np.mean(recovery_times) if recovery_times else None,
            'median_recovery_days': np.median(recovery_times) if recovery_times else None,
            'max_recovery_days': max(recovery_times) if recovery_times else None,
            'min_recovery_days': min(recovery_times) if recovery_times else None,
            'recovery_times': recovery_times
        }
        
        return recovery_stats
    
    def _calculate_rolling_max_dd(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a rolling window."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _perform_factor_analysis(self, 
                               asset_returns: pd.DataFrame, 
                               weights: pd.DataFrame) -> Dict[str, Any]:
        """Perform PCA factor analysis on portfolio."""
        try:
            # Ensure sufficient data for PCA
            if len(asset_returns) < 50 or len(asset_returns.columns) < 3:
                return {'error': 'Insufficient data for factor analysis'}
            
            # PCA on asset returns
            pca = PCA(n_components=min(5, len(asset_returns.columns)))
            pca_result = pca.fit(asset_returns.fillna(0))
            
            # Calculate factor loadings
            factor_loadings = pd.DataFrame(
                pca_result.components_.T,
                columns=[f'Factor_{i+1}' for i in range(pca_result.n_components_)],
                index=asset_returns.columns
            )
            
            # Calculate factor contributions to portfolio
            weighted_loadings = factor_loadings.multiply(weights.mean(), axis=0)
            factor_contributions = weighted_loadings.sum()
            
            return {
                'explained_variance_ratio': pca_result.explained_variance_ratio_,
                'cumulative_explained_variance': np.cumsum(pca_result.explained_variance_ratio_),
                'factor_loadings': factor_loadings,
                'factor_contributions': factor_contributions,
                'n_components': pca_result.n_components_
            }
        except Exception as e:
            logger.warning(f"Factor analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _apply_stress_scenario(self, 
                             returns: pd.Series, 
                             scenario_params: Dict[str, float]) -> pd.Series:
        """Apply stress scenario to returns."""
        stressed_returns = returns.copy()
        
        # Apply return shock
        if 'return_shock' in scenario_params:
            stressed_returns += scenario_params['return_shock'] / len(returns)
        
        # Apply volatility multiplier
        if 'volatility_multiplier' in scenario_params:
            mean_return = returns.mean()
            excess_returns = returns - mean_return
            stressed_returns = mean_return + excess_returns * scenario_params['volatility_multiplier']
        
        return stressed_returns
    
    def _monte_carlo_simulation(self, 
                              returns: pd.Series, 
                              n_simulations: int = 1000,
                              n_periods: int = 252) -> Dict[str, Any]:
        """Monte Carlo simulation for return distribution analysis."""
        logger.info(f"Running Monte Carlo simulation with {n_simulations} paths")
        
        # Estimate parameters from historical data
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Run simulations
        simulated_paths = np.random.normal(
            mean_return, std_return, size=(n_simulations, n_periods)
        )
        
        # Calculate final returns for each path
        final_returns = (1 + simulated_paths).prod(axis=1) - 1.0
        
        # Calculate statistics
        mc_results = {
            'mean_final_return': np.mean(final_returns),
            'std_final_return': np.std(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_25': np.percentile(final_returns, 25),
            'percentile_50': np.percentile(final_returns, 50),
            'percentile_75': np.percentile(final_returns, 75),
            'percentile_95': np.percentile(final_returns, 95),
            'probability_positive': (final_returns > 0).mean(),
            'probability_loss_gt_10': (final_returns < -0.10).mean(),
            'probability_gain_gt_20': (final_returns > 0.20).mean()
        }
        
        return mc_results
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {}
        
        # Basic risk metrics
        volatility = clean_returns.std() * np.sqrt(252)
        downside_vol = clean_returns[clean_returns < 0].std() * np.sqrt(252) if (clean_returns < 0).any() else 0.0
        
        # Tail risk measures
        var_5 = np.percentile(clean_returns, 5)
        var_1 = np.percentile(clean_returns, 1)
        
        # Expected shortfall (CVaR)
        es_5 = clean_returns[clean_returns <= var_5].mean() if (clean_returns <= var_5).any() else 0.0
        es_1 = clean_returns[clean_returns <= var_1].mean() if (clean_returns <= var_1).any() else 0.0
        
        # Tail ratio
        top_10_pct = np.percentile(clean_returns, 90)
        bottom_10_pct = np.percentile(clean_returns, 10)
        tail_ratio = abs(top_10_pct / bottom_10_pct) if bottom_10_pct != 0 else float('inf')
        
        return {
            'volatility_annualized': volatility,
            'downside_volatility': downside_vol,
            'var_5_percent': var_5,
            'var_1_percent': var_1,
            'expected_shortfall_5': es_5,
            'expected_shortfall_1': es_1,
            'tail_ratio': tail_ratio,
            'semi_deviation': downside_vol / np.sqrt(252)  # Daily semi-deviation
        }
    
    def _calculate_trading_statistics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading-specific statistics."""
        portfolio_weights = backtest_results['portfolio_weights']
        transaction_costs = backtest_results['transaction_costs']
        turnover = backtest_results['turnover']
        
        # Portfolio concentration
        avg_concentration = portfolio_weights.apply(
            lambda row: (row ** 2).sum() if row.sum() > 0 else 0, axis=1
        ).mean()
        
        # Active weights statistics
        active_positions = (portfolio_weights.abs() > 0.001).sum(axis=1)
        avg_active_positions = active_positions.mean()
        
        # Cost analysis
        total_costs = transaction_costs.sum()
        avg_daily_costs = transaction_costs.mean()
        costs_as_pct_of_return = total_costs / backtest_results['portfolio_returns'].sum() if backtest_results['portfolio_returns'].sum() != 0 else 0
        
        # Turnover analysis
        avg_turnover = turnover.mean()
        max_turnover = turnover.max()
        
        trading_stats = {
            'avg_portfolio_concentration': avg_concentration,
            'avg_active_positions': avg_active_positions,
            'total_transaction_costs': total_costs,
            'avg_daily_transaction_costs': avg_daily_costs,
            'transaction_costs_pct_of_returns': costs_as_pct_of_return,
            'avg_daily_turnover': avg_turnover,
            'max_daily_turnover': max_turnover,
            'turnover_percentile_95': np.percentile(turnover, 95),
            'days_with_rebalancing': (turnover > 0).sum()
        }
        
        return trading_stats
    
    def _save_performance_report(self, 
                               performance_report: Dict[str, Any], 
                               output_dir: str) -> Dict[str, str]:
        """Save performance report to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save summary statistics
        summary_file = output_path / f"performance_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(performance_report['summary'], f, indent=2, default=str)
        saved_files['summary'] = str(summary_file)
        
        # Save detailed metrics
        metrics_file = output_path / f"performance_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            metrics_data = {
                'return_analysis': performance_report['return_analysis'],
                'risk_metrics': performance_report['risk_metrics'],
                'trading_statistics': performance_report['trading_statistics']
            }
            json.dump(metrics_data, f, indent=2, default=str)
        saved_files['metrics'] = str(metrics_file)
        
        # Save drawdown analysis
        if 'drawdown_analysis' in performance_report:
            dd_analysis = performance_report['drawdown_analysis']
            if 'drawdown_periods' in dd_analysis:
                dd_periods_df = pd.DataFrame(dd_analysis['drawdown_periods'])
                dd_file = output_path / f"drawdown_periods_{timestamp}.parquet"
                dd_periods_df.to_parquet(dd_file)
                saved_files['drawdown_periods'] = str(dd_file)
        
        logger.info(f"Performance report saved to {output_dir}")
        return saved_files


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("PerformanceAnalyzer module loaded successfully")
    print("Module ready for use. Import and use with backtesting results.")