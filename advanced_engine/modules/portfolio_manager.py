#!/usr/bin/env python3
"""
Advanced Risk Parity Portfolio Management System

This module provides comprehensive portfolio management capabilities with multiple
risk parity implementations, volatility estimation methods, risk monitoring,
and optimization algorithms. Designed for institutional-grade portfolio management
with strict lookahead bias prevention.

Key Features:
- Multiple risk parity weighting methods (ERC, inverse volatility, risk budgeting)
- Advanced volatility estimation (rolling, EWMA, GARCH, realized volatility)
- Comprehensive risk monitoring (VaR, CVaR, correlation analysis, concentration risk)
- Portfolio optimization with transaction cost awareness
- Real-time position tracking and rebalancing logic
- Integration with contrarian signal generation framework

Author: Claude Code  
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
from scipy import optimize
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
import cvxpy as cp

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@nb.jit(nopython=True, cache=True)
def calculate_ewma_volatility_numba(returns: np.ndarray, 
                                  lambda_param: float = 0.94) -> np.ndarray:
    """
    Numba-optimized EWMA volatility calculation.
    
    Args:
        returns: Array of returns
        lambda_param: EWMA decay parameter
        
    Returns:
        Array of EWMA volatilities
    """
    n = len(returns)
    ewma_var = np.zeros(n)
    
    if n == 0:
        return ewma_var
    
    # Initialize with first return squared
    ewma_var[0] = returns[0] ** 2 if not np.isnan(returns[0]) else 0.0
    
    for i in range(1, n):
        if not np.isnan(returns[i]):
            ewma_var[i] = lambda_param * ewma_var[i-1] + (1 - lambda_param) * (returns[i] ** 2)
        else:
            ewma_var[i] = ewma_var[i-1]
    
    return np.sqrt(ewma_var * 252)  # Annualize


@nb.jit(nopython=True, cache=True)
def calculate_realized_volatility_numba(returns: np.ndarray, 
                                      window: int = 22) -> np.ndarray:
    """
    Numba-optimized realized volatility calculation.
    
    Args:
        returns: Array of returns
        window: Rolling window size
        
    Returns:
        Array of realized volatilities
    """
    n = len(returns)
    realized_vol = np.full(n, np.nan)
    
    for i in range(window, n):
        window_returns = returns[i-window:i]
        valid_returns = window_returns[~np.isnan(window_returns)]
        
        if len(valid_returns) >= window // 2:
            realized_vol[i] = np.sqrt(np.sum(valid_returns ** 2) * 252)
    
    return realized_vol


@nb.jit(nopython=True, cache=True)
def calculate_portfolio_risk_numba(weights: np.ndarray, 
                                 cov_matrix: np.ndarray) -> float:
    """
    Numba-optimized portfolio risk calculation.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        
    Returns:
        Portfolio volatility
    """
    return np.sqrt(weights.T @ cov_matrix @ weights)


class VolatilityEstimator:
    """
    Advanced volatility estimation with multiple methodologies.
    """
    
    def __init__(self, method: str = 'rolling'):
        """
        Initialize volatility estimator.
        
        Args:
            method: Volatility estimation method ('rolling', 'ewma', 'garch', 'realized')
        """
        self.method = method
        self.fitted_params = {}
        logger.info(f"Initialized VolatilityEstimator with {method} method")
    
    def estimate_volatility(self, 
                          returns: pd.DataFrame, 
                          window: int = 60,
                          min_periods: Optional[int] = None) -> pd.DataFrame:
        """
        Estimate volatility using specified method with strict historical data only.
        
        Args:
            returns: Returns DataFrame (dates x assets)
            window: Lookback window for volatility estimation
            min_periods: Minimum periods required for calculation
            
        Returns:
            DataFrame with volatility estimates (lagged by 1 day)
        """
        logger.debug(f"Estimating volatility using {self.method} method")
        
        if min_periods is None:
            min_periods = max(window // 2, 10)
        
        if self.method == 'rolling':
            volatility = self._rolling_volatility(returns, window, min_periods)
        elif self.method == 'ewma':
            volatility = self._ewma_volatility(returns, window)
        elif self.method == 'garch':
            volatility = self._garch_volatility(returns, window)
        elif self.method == 'realized':
            volatility = self._realized_volatility(returns, window)
        else:
            raise ValueError(f"Unknown volatility method: {self.method}")
        
        # CRITICAL: Shift by 1 day to prevent lookahead bias
        volatility_lagged = volatility.shift(1)
        
        logger.debug(f"Calculated {self.method} volatility with 1-day lag")
        return volatility_lagged
    
    def _rolling_volatility(self, 
                           returns: pd.DataFrame, 
                           window: int, 
                           min_periods: int) -> pd.DataFrame:
        """Calculate rolling volatility."""
        rolling_vol = returns.rolling(
            window=window, 
            min_periods=min_periods
        ).std() * np.sqrt(252)
        
        return rolling_vol
    
    def _ewma_volatility(self, 
                        returns: pd.DataFrame, 
                        window: int,
                        lambda_param: float = 0.94) -> pd.DataFrame:
        """Calculate EWMA volatility using numba optimization."""
        volatility_df = pd.DataFrame(
            index=returns.index, 
            columns=returns.columns, 
            dtype=float
        )
        
        for col in returns.columns:
            returns_array = returns[col].fillna(0.0).values
            ewma_vol = calculate_ewma_volatility_numba(returns_array, lambda_param)
            volatility_df[col] = ewma_vol
        
        return volatility_df
    
    def _garch_volatility(self, 
                         returns: pd.DataFrame, 
                         window: int) -> pd.DataFrame:
        """Calculate GARCH volatility (simplified GARCH(1,1))."""
        volatility_df = pd.DataFrame(
            index=returns.index, 
            columns=returns.columns, 
            dtype=float
        )
        
        for col in returns.columns:
            col_returns = returns[col].dropna()
            if len(col_returns) < window:
                volatility_df[col] = np.nan
                continue
            
            # Simplified GARCH(1,1) implementation
            garch_vol = self._fit_simple_garch(col_returns, window)
            volatility_df[col] = garch_vol.reindex(returns.index)
        
        return volatility_df
    
    def _fit_simple_garch(self, 
                         returns: pd.Series, 
                         window: int,
                         alpha: float = 0.1, 
                         beta: float = 0.85) -> pd.Series:
        """Simplified GARCH(1,1) implementation."""
        omega = 0.000001  # Long-term variance
        
        # Initialize conditional variance
        cond_var = pd.Series(index=returns.index, dtype=float)
        cond_var.iloc[0] = returns.var()
        
        # Recursive GARCH calculation
        for i in range(1, len(returns)):
            if i < window:
                cond_var.iloc[i] = returns.iloc[:i+1].var()
            else:
                lagged_return_sq = returns.iloc[i-1] ** 2
                lagged_cond_var = cond_var.iloc[i-1]
                cond_var.iloc[i] = omega + alpha * lagged_return_sq + beta * lagged_cond_var
        
        # Convert to annualized volatility
        return np.sqrt(cond_var * 252)
    
    def _realized_volatility(self, 
                           returns: pd.DataFrame, 
                           window: int) -> pd.DataFrame:
        """Calculate realized volatility using numba optimization."""
        volatility_df = pd.DataFrame(
            index=returns.index, 
            columns=returns.columns, 
            dtype=float
        )
        
        for col in returns.columns:
            returns_array = returns[col].fillna(0.0).values
            realized_vol = calculate_realized_volatility_numba(returns_array, window)
            volatility_df[col] = realized_vol
        
        return volatility_df
    
    def forecast_volatility(self, 
                          returns: pd.DataFrame, 
                          horizon: int = 1) -> pd.DataFrame:
        """
        Forecast volatility for specified horizon.
        
        Args:
            returns: Historical returns
            horizon: Forecast horizon in days
            
        Returns:
            DataFrame with volatility forecasts
        """
        if self.method == 'garch':
            return self._garch_forecast(returns, horizon)
        else:
            # For other methods, use current volatility as forecast
            current_vol = self.estimate_volatility(returns)
            return pd.concat([current_vol] * horizon, ignore_index=True)
    
    def _garch_forecast(self, 
                       returns: pd.DataFrame, 
                       horizon: int) -> pd.DataFrame:
        """GARCH volatility forecasting."""
        # Simplified: assume volatility mean-reverts to long-term average
        current_vol = self.estimate_volatility(returns)
        long_term_vol = returns.rolling(window=252, min_periods=126).std() * np.sqrt(252)
        
        # Simple mean reversion forecast
        decay_factor = 0.95 ** np.arange(1, horizon + 1)
        forecast = pd.DataFrame(index=range(horizon), columns=returns.columns)
        
        for i in range(horizon):
            forecast.iloc[i] = (decay_factor[i] * current_vol.iloc[-1] + 
                              (1 - decay_factor[i]) * long_term_vol.iloc[-1])
        
        return forecast


class RiskParityOptimizer:
    """
    Advanced risk parity optimization with multiple methodologies.
    """
    
    def __init__(self, method: str = 'inverse_volatility'):
        """
        Initialize risk parity optimizer.
        
        Args:
            method: Risk parity method ('inverse_volatility', 'erc', 'risk_budgeting')
        """
        self.method = method
        self.optimization_results = {}
        logger.info(f"Initialized RiskParityOptimizer with {method} method")
    
    def calculate_risk_parity_weights(self, 
                                    selected_assets: pd.Series,
                                    volatility: pd.Series,
                                    correlation_matrix: Optional[np.ndarray] = None,
                                    risk_budget: Optional[np.ndarray] = None) -> pd.Series:
        """
        Calculate risk parity weights for selected assets.
        
        Args:
            selected_assets: Boolean series indicating selected assets
            volatility: Asset volatility estimates
            correlation_matrix: Asset correlation matrix (optional)
            risk_budget: Target risk contributions (optional)
            
        Returns:
            Series with risk parity weights
        """
        # Filter to selected assets only
        selected_vol = volatility[selected_assets]
        
        if len(selected_vol) == 0:
            return pd.Series(0.0, index=volatility.index)
        
        # Handle NaN values
        selected_vol = selected_vol.fillna(selected_vol.median()).clip(lower=0.001)
        
        if self.method == 'inverse_volatility':
            weights = self._inverse_volatility_weights(selected_vol)
        elif self.method == 'erc':
            weights = self._equal_risk_contribution_weights(
                selected_vol, correlation_matrix, selected_assets
            )
        elif self.method == 'risk_budgeting':
            weights = self._risk_budgeting_weights(
                selected_vol, correlation_matrix, risk_budget, selected_assets
            )
        else:
            raise ValueError(f"Unknown risk parity method: {self.method}")
        
        # Create full weight series
        full_weights = pd.Series(0.0, index=volatility.index)
        full_weights[selected_assets] = weights
        
        return full_weights
    
    def _inverse_volatility_weights(self, volatility: pd.Series) -> pd.Series:
        """Calculate inverse volatility weights."""
        inv_vol = 1.0 / volatility
        weights = inv_vol / inv_vol.sum()
        return weights
    
    def _equal_risk_contribution_weights(self, 
                                       volatility: pd.Series,
                                       correlation_matrix: Optional[np.ndarray],
                                       selected_assets: pd.Series) -> pd.Series:
        """
        Calculate Equal Risk Contribution (ERC) weights using optimization.
        """
        n = len(volatility)
        
        if correlation_matrix is None or n <= 1:
            # Fall back to inverse volatility
            return self._inverse_volatility_weights(volatility)
        
        # The correlation matrix is already calculated for selected assets only,
        # so we don't need to filter it again. Just verify dimensions match.
        if correlation_matrix.shape[0] != len(volatility) or correlation_matrix.shape[1] != len(volatility):
            # Correlation matrix size mismatch, fall back to inverse volatility
            return self._inverse_volatility_weights(volatility)
        
        filtered_corr = correlation_matrix
        vol_array = volatility.values
        
        # Create covariance matrix
        cov_matrix = np.outer(vol_array, vol_array) * filtered_corr
        
        # ERC optimization problem
        def erc_objective(weights):
            portfolio_vol = calculate_portfolio_risk_numba(weights, cov_matrix)
            
            if portfolio_vol <= 1e-8:
                return 1e6  # Large penalty for zero portfolio volatility
            
            # Risk contributions
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            
            # Objective: minimize sum of squared deviations from equal risk
            target_risk = portfolio_vol / n
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
        ]
        
        bounds = [(0.001, 1.0) for _ in range(n)]  # Long-only with minimum weight
        
        # Initial guess: inverse volatility weights
        x0 = self._inverse_volatility_weights(volatility).values
        
        try:
            result = optimize.minimize(
                erc_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=volatility.index)
                self.optimization_results['erc_success'] = True
                return weights
            else:
                logger.warning("ERC optimization failed, falling back to inverse volatility")
                self.optimization_results['erc_success'] = False
                return self._inverse_volatility_weights(volatility)
                
        except Exception as e:
            logger.warning(f"ERC optimization error: {str(e)}, falling back to inverse volatility")
            self.optimization_results['erc_success'] = False
            return self._inverse_volatility_weights(volatility)
    
    def _risk_budgeting_weights(self, 
                              volatility: pd.Series,
                              correlation_matrix: Optional[np.ndarray],
                              risk_budget: Optional[np.ndarray],
                              selected_assets: pd.Series) -> pd.Series:
        """
        Calculate risk budgeting weights with specified risk allocations.
        """
        n = len(volatility)
        
        if risk_budget is None:
            # Default to equal risk budget
            risk_budget = np.ones(n) / n
        
        if correlation_matrix is None or n <= 1:
            # Scale inverse volatility weights by risk budget
            inv_vol_weights = self._inverse_volatility_weights(volatility)
            scaled_weights = inv_vol_weights * risk_budget
            return scaled_weights / scaled_weights.sum()
        
        # Similar to ERC but with custom risk budgets
        # The correlation matrix is already calculated for selected assets only
        if correlation_matrix.shape[0] != n or correlation_matrix.shape[1] != n:
            # Correlation matrix size mismatch, fall back to scaled inverse volatility
            inv_vol_weights = self._inverse_volatility_weights(volatility)
            scaled_weights = inv_vol_weights * risk_budget
            return scaled_weights / scaled_weights.sum()
        
        filtered_corr = correlation_matrix
        vol_array = volatility.values
        cov_matrix = np.outer(vol_array, vol_array) * filtered_corr
        
        def risk_budget_objective(weights):
            portfolio_vol = calculate_portfolio_risk_numba(weights, cov_matrix)
            
            if portfolio_vol <= 1e-8:
                return 1e6
            
            marginal_risk = (cov_matrix @ weights) / portfolio_vol
            risk_contributions = weights * marginal_risk
            target_risk = risk_budget * portfolio_vol
            
            return np.sum((risk_contributions - target_risk) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.001, 1.0) for _ in range(n)]
        x0 = self._inverse_volatility_weights(volatility).values
        
        try:
            result = optimize.minimize(
                risk_budget_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.success:
                return pd.Series(result.x, index=volatility.index)
            else:
                logger.warning("Risk budgeting optimization failed, falling back to inverse volatility")
                return self._inverse_volatility_weights(volatility)
                
        except Exception as e:
            logger.warning(f"Risk budgeting error: {str(e)}, falling back to inverse volatility")
            return self._inverse_volatility_weights(volatility)


class RiskMonitor:
    """
    Comprehensive portfolio risk monitoring system.
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 lookback_window: int = 252):
        """
        Initialize risk monitor.
        
        Args:
            confidence_levels: VaR confidence levels
            lookback_window: Lookback window for risk calculations
        """
        self.confidence_levels = confidence_levels
        self.lookback_window = lookback_window
        self.risk_metrics_history = {}
        logger.info(f"Initialized RiskMonitor with {confidence_levels} confidence levels")
    
    def calculate_portfolio_risk_metrics(self, 
                                       portfolio_returns: pd.Series,
                                       portfolio_weights: pd.DataFrame,
                                       asset_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            portfolio_weights: Portfolio weights over time
            asset_returns: Individual asset returns
            
        Returns:
            Dictionary with risk metrics
        """
        risk_metrics = {}
        
        # Remove NaN values
        clean_returns = portfolio_returns.dropna()
        
        if len(clean_returns) < 10:
            logger.warning("Insufficient data for risk calculation")
            return risk_metrics
        
        # Basic risk metrics
        risk_metrics.update(self._calculate_basic_risk_metrics(clean_returns))
        
        # VaR and CVaR
        risk_metrics.update(self._calculate_var_cvar(clean_returns))
        
        # Concentration risk
        risk_metrics.update(self._calculate_concentration_risk(portfolio_weights))
        
        # Correlation-based risk
        risk_metrics.update(self._calculate_correlation_risk(
            portfolio_weights, asset_returns
        ))
        
        # Drawdown analysis
        risk_metrics.update(self._calculate_drawdown_metrics(clean_returns))
        
        # Store in history
        current_date = portfolio_returns.index[-1] if len(portfolio_returns) > 0 else datetime.now()
        self.risk_metrics_history[current_date] = risk_metrics
        
        logger.debug("Calculated comprehensive risk metrics")
        return risk_metrics
    
    def _calculate_basic_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic risk metrics."""
        metrics = {
            'volatility_annualized': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max(),
        }
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = (negative_returns.std() * np.sqrt(252) 
                                       if len(negative_returns) > 0 else 0.0)
        
        return metrics
    
    def _calculate_var_cvar(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk."""
        var_cvar = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            
            # Historical VaR
            var_historical = np.percentile(returns, alpha * 100)
            var_cvar[f'var_{int(conf_level*100)}_historical'] = var_historical
            
            # Parametric VaR (assuming normal distribution)
            var_parametric = norm.ppf(alpha, returns.mean(), returns.std())
            var_cvar[f'var_{int(conf_level*100)}_parametric'] = var_parametric
            
            # Conditional VaR (Expected Shortfall)
            cvar = returns[returns <= var_historical].mean()
            var_cvar[f'cvar_{int(conf_level*100)}'] = cvar
        
        return var_cvar
    
    def _calculate_concentration_risk(self, weights: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio concentration risk metrics."""
        if weights.empty:
            return {'concentration_hhi': 0.0, 'effective_assets': 0.0, 'max_weight': 0.0}
        
        latest_weights = weights.iloc[-1].abs()
        
        # Herfindahl-Hirschman Index
        hhi = (latest_weights ** 2).sum()
        
        # Effective number of assets
        effective_assets = 1.0 / hhi if hhi > 0 else 0.0
        
        # Maximum weight
        max_weight = latest_weights.max()
        
        return {
            'concentration_hhi': hhi,
            'effective_assets': effective_assets,
            'max_weight': max_weight,
            'weight_entropy': -np.sum(latest_weights * np.log(latest_weights + 1e-10))
        }
    
    def _calculate_correlation_risk(self, 
                                  weights: pd.DataFrame, 
                                  asset_returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation-based risk metrics."""
        if weights.empty or asset_returns.empty:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0}
        
        # Align weights and returns
        common_dates = weights.index.intersection(asset_returns.index)
        common_assets = weights.columns.intersection(asset_returns.columns)
        
        if len(common_dates) < 2 or len(common_assets) < 2:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0}
        
        aligned_returns = asset_returns.loc[common_dates, common_assets]
        aligned_weights = weights.loc[common_dates, common_assets]
        
        # Calculate correlation matrix
        correlation_matrix = aligned_returns.corr()
        
        # Weight-adjusted correlation metrics
        latest_weights = aligned_weights.iloc[-1].abs()
        active_assets = latest_weights[latest_weights > 0.001].index
        
        if len(active_assets) < 2:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0}
        
        active_corr = correlation_matrix.loc[active_assets, active_assets]
        
        # Extract upper triangle (excluding diagonal)
        upper_tri = np.triu(active_corr.values, k=1)
        correlations = upper_tri[upper_tri != 0]
        
        if len(correlations) > 0:
            avg_correlation = np.mean(correlations)
            max_correlation = np.max(np.abs(correlations))
        else:
            avg_correlation = 0.0
            max_correlation = 0.0
        
        return {
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': np.min(correlations) if len(correlations) > 0 else 0.0
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-based risk metrics."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        max_drawdown = drawdowns.min()
        
        # Drawdown duration analysis
        in_drawdown = drawdowns < -0.001  # More than 0.1% drawdown
        if in_drawdown.any():
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            avg_drawdown_duration = 0
            max_drawdown_duration = 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'current_drawdown': abs(drawdowns.iloc[-1]),
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def check_risk_limits(self, 
                         risk_metrics: Dict[str, float],
                         risk_limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if risk metrics exceed specified limits.
        
        Args:
            risk_metrics: Current risk metrics
            risk_limits: Risk limit thresholds
            
        Returns:
            Dictionary indicating which limits are breached
        """
        breaches = {}
        
        for limit_name, limit_value in risk_limits.items():
            if limit_name in risk_metrics:
                current_value = risk_metrics[limit_name]
                breaches[limit_name] = current_value > limit_value
                
                if breaches[limit_name]:
                    logger.debug(f"Risk limit breach: {limit_name} = {current_value:.4f} > {limit_value:.4f}")
        
        return breaches


class PortfolioManager:
    """
    Comprehensive risk parity portfolio management system.
    
    This class integrates volatility estimation, risk parity optimization,
    risk monitoring, and portfolio construction with strict lookahead bias prevention.
    """
    
    def __init__(self,
                 volatility_method: str = 'ewma',
                 risk_parity_method: str = 'inverse_volatility',
                 volatility_lookback: int = 60,
                 correlation_lookback: int = 126,
                 rebalancing_frequency: str = 'daily',
                 transaction_cost_bps: float = 2.0,
                 max_position_size: float = 0.25,
                 min_position_size: float = 0.01,
                 target_volatility: Optional[float] = None,
                 risk_budget: Optional[Dict[str, float]] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            volatility_method: Method for volatility estimation
            risk_parity_method: Method for risk parity calculation
            volatility_lookback: Days for volatility calculation
            correlation_lookback: Days for correlation calculation
            rebalancing_frequency: Portfolio rebalancing frequency
            transaction_cost_bps: Transaction costs in basis points
            max_position_size: Maximum position size per asset
            min_position_size: Minimum position size per asset
            target_volatility: Target portfolio volatility (optional)
            risk_budget: Custom risk budget allocation (optional)
        """
        # Initialize components
        self.volatility_estimator = VolatilityEstimator(volatility_method)
        self.risk_parity_optimizer = RiskParityOptimizer(risk_parity_method)
        self.risk_monitor = RiskMonitor()
        
        # Parameters
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.rebalancing_frequency = rebalancing_frequency
        self.transaction_cost_bps = transaction_cost_bps
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.target_volatility = target_volatility
        self.risk_budget = risk_budget
        
        # State tracking
        self.current_weights = {}
        self.portfolio_history = {}
        self.risk_metrics_history = {}
        self.rebalancing_dates = []
        
        logger.info(f"Initialized PortfolioManager: vol_method={volatility_method}, "
                   f"rp_method={risk_parity_method}, target_vol={target_volatility}")
    
    def construct_portfolio_weights(self,
                                  signal_output: Dict[str, pd.DataFrame],
                                  returns: pd.DataFrame,
                                  current_date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Construct portfolio weights for a specific date using risk parity methodology.
        
        Args:
            signal_output: Output from ConrarianSignalGenerator.generate_signals()
            returns: Asset returns DataFrame
            current_date: Date for which to construct portfolio (uses last date if None)
            
        Returns:
            Series with portfolio weights for the specified date
        """
        if current_date is None:
            current_date = signal_output['binary_signals'].index[-1]
        
        logger.debug(f"Constructing portfolio weights for {current_date}")
        
        # Get signals for the current date
        binary_signals = signal_output['binary_signals'].loc[current_date]
        selected_assets = binary_signals > 0
        
        if not selected_assets.any():
            # No assets selected, return zero weights
            return pd.Series(0.0, index=binary_signals.index)
        
        # Calculate volatility estimates (ensuring no lookahead bias)
        volatility_estimates = self.volatility_estimator.estimate_volatility(
            returns, window=self.volatility_lookback
        )
        
        if current_date not in volatility_estimates.index:
            # Use most recent available volatility
            available_dates = volatility_estimates.index[volatility_estimates.index <= current_date]
            if len(available_dates) == 0:
                logger.warning(f"No volatility data available for {current_date}")
                return pd.Series(0.0, index=binary_signals.index)
            current_date_vol = available_dates[-1]
        else:
            current_date_vol = current_date
        
        current_volatility = volatility_estimates.loc[current_date_vol]
        
        # Calculate correlation matrix if needed for advanced risk parity methods
        correlation_matrix = None
        if self.risk_parity_optimizer.method in ['erc', 'risk_budgeting']:
            correlation_matrix = self._calculate_correlation_matrix(
                returns, current_date, selected_assets
            )
        
        # Get risk budget if specified
        risk_budget = None
        if self.risk_budget and self.risk_parity_optimizer.method == 'risk_budgeting':
            risk_budget = np.array([
                self.risk_budget.get(asset, 1.0/selected_assets.sum()) 
                for asset in selected_assets.index[selected_assets]
            ])
        
        # Calculate risk parity weights
        portfolio_weights = self.risk_parity_optimizer.calculate_risk_parity_weights(
            selected_assets=selected_assets,
            volatility=current_volatility,
            correlation_matrix=correlation_matrix,
            risk_budget=risk_budget
        )
        
        # Apply position limits and constraints
        portfolio_weights = self._apply_position_constraints(portfolio_weights)
        
        # Apply volatility targeting if specified
        if self.target_volatility is not None:
            portfolio_weights = self._apply_volatility_targeting(
                portfolio_weights, current_volatility, correlation_matrix
            )
        
        logger.debug(f"Constructed portfolio with {(portfolio_weights > 0).sum()} positions")
        return portfolio_weights
    
    def _calculate_correlation_matrix(self,
                                    returns: pd.DataFrame,
                                    current_date: pd.Timestamp,
                                    selected_assets: pd.Series) -> Optional[np.ndarray]:
        """
        Calculate correlation matrix using only historical data.
        
        Args:
            returns: Asset returns DataFrame
            current_date: Current date for calculation
            selected_assets: Boolean series of selected assets
            
        Returns:
            Correlation matrix for selected assets or None
        """
        # Get historical data up to (but not including) current date
        historical_data = returns.loc[returns.index < current_date]
        
        if len(historical_data) < self.correlation_lookback:
            logger.warning(f"Insufficient data for correlation calculation: {len(historical_data)} < {self.correlation_lookback}")
            return None
        
        # Use most recent data
        recent_data = historical_data.tail(self.correlation_lookback)
        
        # Filter to selected assets
        selected_columns = selected_assets.index[selected_assets]
        correlation_data = recent_data[selected_columns].dropna()
        
        if len(correlation_data) < self.correlation_lookback // 2:
            logger.warning("Insufficient clean data for correlation calculation")
            return None
        
        # Calculate correlation matrix
        try:
            correlation_matrix = correlation_data.corr().values
            
            # Ensure matrix is positive definite
            eigenvals = np.linalg.eigvals(correlation_matrix)
            if np.any(eigenvals <= 0):
                logger.warning("Correlation matrix is not positive definite, applying regularization")
                correlation_matrix = self._regularize_correlation_matrix(correlation_matrix)
            
            return correlation_matrix
            
        except Exception as e:
            logger.warning(f"Error calculating correlation matrix: {str(e)}")
            return None
    
    def _regularize_correlation_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Regularize correlation matrix to ensure positive definiteness.
        
        Args:
            corr_matrix: Original correlation matrix
            
        Returns:
            Regularized correlation matrix
        """
        # Use Ledoit-Wolf shrinkage
        try:
            lw = LedoitWolf()
            # Dummy data for LedoitWolf (we only need the shrinkage concept)
            n = corr_matrix.shape[0]
            dummy_returns = np.random.multivariate_normal(
                np.zeros(n), corr_matrix, size=max(100, n*3)
            )
            regularized_cov = lw.fit(dummy_returns).covariance_
            
            # Convert back to correlation
            std_devs = np.sqrt(np.diag(regularized_cov))
            regularized_corr = regularized_cov / np.outer(std_devs, std_devs)
            
            return regularized_corr
            
        except Exception:
            # Fallback: simple shrinkage towards identity matrix
            alpha = 0.1
            n = corr_matrix.shape[0]
            identity = np.eye(n)
            return (1 - alpha) * corr_matrix + alpha * identity
    
    def _apply_position_constraints(self, weights: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply position size constraints and other portfolio limits.
        
        Args:
            weights: Original portfolio weights (Series or DataFrame)
            
        Returns:
            Constrained portfolio weights (same type as input)
        """
        constrained_weights = weights.copy()
        
        # Apply minimum position threshold
        constrained_weights = constrained_weights.where(
            constrained_weights.abs() >= self.min_position_size, 0.0
        )
        
        # Apply maximum position size limits
        constrained_weights = constrained_weights.clip(-self.max_position_size, self.max_position_size)
        
        if isinstance(constrained_weights, pd.Series):
            # Series case
            total_weight = constrained_weights.abs().sum()
            if total_weight > 0:
                constrained_weights = constrained_weights / total_weight
        else:
            # DataFrame case - normalize each row
            row_sums = constrained_weights.abs().sum(axis=1)
            # Avoid division by zero
            row_sums = row_sums.replace(0, 1)
            constrained_weights = constrained_weights.div(row_sums, axis=0)
        
        return constrained_weights
    
    def _apply_volatility_targeting(self,
                                  weights: pd.Series,
                                  volatility: pd.Series,
                                  correlation_matrix: Optional[np.ndarray]) -> pd.Series:
        """
        Scale portfolio to target volatility level.
        
        Args:
            weights: Portfolio weights
            volatility: Asset volatilities
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Volatility-targeted portfolio weights
        """
        if correlation_matrix is None:
            # Simple volatility scaling using weighted average volatility
            portfolio_vol = (weights.abs() * volatility).sum()
        else:
            # Use covariance matrix for accurate portfolio volatility
            # Note: correlation_matrix is only for assets with non-zero weights
            active_weights = weights[weights.abs() > 1e-6]  # Get non-zero weights
            
            if len(active_weights) == 0:
                portfolio_vol = 0.0
            elif len(active_weights) != correlation_matrix.shape[0]:
                # Correlation matrix size doesn't match active weights, fallback to simple method
                portfolio_vol = (weights.abs() * volatility).sum()
            else:
                # Extract volatilities for active assets only
                active_vol = volatility[active_weights.index].fillna(volatility.median()).values
                active_weight_array = active_weights.values
                
                # Create covariance matrix for active assets only
                cov_matrix = np.outer(active_vol, active_vol) * correlation_matrix
                portfolio_vol = calculate_portfolio_risk_numba(active_weight_array, cov_matrix)
        
        if portfolio_vol > 0 and self.target_volatility is not None:
            scaling_factor = self.target_volatility / portfolio_vol
            scaled_weights = weights * scaling_factor
            
            # Ensure we don't exceed position limits after scaling
            scaled_weights = self._apply_position_constraints(scaled_weights)
            
            logger.debug(f"Applied volatility targeting: {portfolio_vol:.4f} -> {self.target_volatility:.4f}")
            return scaled_weights
        
        return weights
    
    def _vectorized_inverse_volatility_portfolio(self,
                                               binary_signals: pd.DataFrame,
                                               volatility_data: Optional[pd.DataFrame],
                                               returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized portfolio construction for inverse volatility method.
        """
        if volatility_data is None:
            # Calculate volatility using vectorized operations
            volatility_data = returns_data.rolling(
                window=self.volatility_lookback, 
                min_periods=max(10, self.volatility_lookback // 2)
            ).std() * np.sqrt(252)
            volatility_data = volatility_data.shift(1)  # Avoid lookahead bias
        
        # Fill missing volatility with cross-sectional median
        volatility_filled = volatility_data.fillna(
            volatility_data.median(axis=1, skipna=True), axis=0
        ).fillna(0.15).clip(lower=0.01)  # 15% default volatility, minimum 1%
        
        # Calculate inverse volatility weights for selected assets
        inv_vol = 1.0 / volatility_filled
        weighted_signals = binary_signals * inv_vol
        
        # Normalize row-wise (vectorized)
        row_sums = weighted_signals.sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # Avoid division by zero
        
        portfolio_weights = weighted_signals.div(row_sums, axis=0)
        
        # Apply position constraints
        portfolio_weights = self._apply_position_constraints(portfolio_weights)
        
        return portfolio_weights
    
    def _optimized_date_by_date_portfolio(self,
                                        binary_signals: pd.DataFrame,
                                        volatility_data: Optional[pd.DataFrame],
                                        returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized date-by-date processing for ERC and risk budgeting methods.
        """
        portfolio_weights_history = pd.DataFrame(
            0.0, index=binary_signals.index, columns=binary_signals.columns
        )
        
        # Pre-calculate all volatilities to avoid repeated computation
        if volatility_data is None:
            volatility_estimates = self.volatility_estimator.estimate_volatility(
                returns_data, window=self.volatility_lookback
            )
        else:
            volatility_estimates = volatility_data
        
        # Process only rebalancing dates to reduce computation
        rebalance_schedule = self.generate_rebalancing_schedule(binary_signals.index)
        rebalancing_dates = binary_signals.index[rebalance_schedule]
        
        current_weights = pd.Series(0.0, index=binary_signals.columns)
        
        for date in rebalancing_dates:
            # Get signals and volatility for current date
            binary_signals_date = binary_signals.loc[date]
            selected_assets = binary_signals_date > 0
            
            if not selected_assets.any():
                current_weights = pd.Series(0.0, index=binary_signals.columns)
            else:
                # Get volatility estimate
                available_vol_dates = volatility_estimates.index[volatility_estimates.index <= date]
                if len(available_vol_dates) > 0:
                    vol_date = available_vol_dates[-1]
                    current_volatility = volatility_estimates.loc[vol_date]
                else:
                    # Fallback to default volatility
                    current_volatility = pd.Series(0.15, index=binary_signals.columns)
                
                # Calculate correlation matrix only if needed
                correlation_matrix = None
                if self.risk_parity_optimizer.method in ['erc', 'risk_budgeting']:
                    correlation_matrix = self._calculate_correlation_matrix(
                        returns_data, date, selected_assets
                    )
                
                # Calculate weights using risk parity optimizer
                current_weights = self.risk_parity_optimizer.calculate_risk_parity_weights(
                    selected_assets=selected_assets,
                    volatility=current_volatility,
                    correlation_matrix=correlation_matrix
                )
            
            # Forward-fill weights until next rebalancing date
            next_rebal_idx = rebalancing_dates.get_indexer([date], method='ffill')[0]
            if next_rebal_idx < len(rebalancing_dates) - 1:
                end_date = rebalancing_dates[next_rebal_idx + 1]
                date_range = binary_signals.index[(binary_signals.index >= date) & (binary_signals.index < end_date)]
            else:
                date_range = binary_signals.index[binary_signals.index >= date]
            
            for fill_date in date_range:
                portfolio_weights_history.loc[fill_date] = current_weights
        
        return portfolio_weights_history
    
    def _calculate_vectorized_transaction_costs(self,
                                              portfolio_weights: pd.DataFrame,
                                              rebalance_schedule: pd.Series) -> pd.Series:
        """
        Vectorized transaction cost calculation.
        """
        # Calculate weight changes
        weight_changes = portfolio_weights.diff().abs()
        
        # Apply base transaction cost
        base_cost = self.transaction_cost_bps / 10000.0
        
        # Sum across assets and apply only on rebalancing dates
        transaction_costs = weight_changes.sum(axis=1) * base_cost
        transaction_costs = transaction_costs * rebalance_schedule.astype(float)
        
        return transaction_costs
    
    def _calculate_strategic_risk_metrics(self,
                                        portfolio_returns: pd.Series,
                                        portfolio_weights: pd.DataFrame,
                                        returns_data: pd.DataFrame,
                                        signal_dates: pd.DatetimeIndex) -> Dict:
        """
        Calculate risk metrics at strategic points (monthly) to reduce computation.
        """
        risk_metrics_history = {}
        
        # Calculate monthly (first business day of each month) + final date
        try:
            monthly_dates = signal_dates.to_series().groupby(
                [signal_dates.year, signal_dates.month]
            ).first()
            
            strategic_dates = list(monthly_dates.values) + [signal_dates[-1]]
            strategic_dates = list(set(strategic_dates))  # Remove duplicates
            
            # Convert to timezone-naive if necessary and sort
            strategic_dates_cleaned = []
            for date in strategic_dates:
                if hasattr(date, 'tz') and date.tz is not None:
                    date = date.tz_localize(None)
                strategic_dates_cleaned.append(date)
            
            strategic_dates = strategic_dates_cleaned
            strategic_dates.sort()
        except Exception as e:
            logger.debug(f"Error processing strategic dates: {str(e)}")
            # Fallback to a simple strategy
            strategic_dates = [signal_dates[i] for i in range(0, len(signal_dates), 30)] + [signal_dates[-1]]
        
        for date in strategic_dates:
            if date in portfolio_returns.index and len(portfolio_returns.loc[:date].dropna()) > 10:
                try:
                    risk_metrics = self.risk_monitor.calculate_portfolio_risk_metrics(
                        portfolio_returns.loc[:date],
                        portfolio_weights.loc[:date],
                        returns_data.loc[:date]
                    )
                    risk_metrics_history[date] = risk_metrics
                except Exception as e:
                    logger.debug(f"Risk metrics calculation failed for {date}: {str(e)}")
                    continue
        
        return risk_metrics_history
    
    def generate_rebalancing_schedule(self,
                                    signal_dates: pd.DatetimeIndex,
                                    frequency: Optional[str] = None) -> pd.Series:
        """
        Generate rebalancing schedule based on frequency and signal availability.
        
        Args:
            signal_dates: Available signal dates
            frequency: Rebalancing frequency override
            
        Returns:
            Boolean series indicating rebalancing dates
        """
        freq = frequency or self.rebalancing_frequency
        rebalance_schedule = pd.Series(False, index=signal_dates)
        
        if freq == 'daily':
            rebalance_schedule[:] = True
        elif freq == 'weekly':
            # Rebalance on Mondays
            rebalance_schedule[signal_dates.dayofweek == 0] = True
        elif freq == 'monthly':
            # Rebalance on first available day of each month
            monthly_dates = signal_dates.to_series().groupby(
                [signal_dates.year, signal_dates.month]
            ).first()
            rebalance_schedule[monthly_dates] = True
        else:
            raise ValueError(f"Unsupported rebalancing frequency: {freq}")
        
        # Always rebalance on first available date
        if len(rebalance_schedule) > 0:
            rebalance_schedule.iloc[0] = True
        
        logger.info(f"Generated rebalancing schedule: {rebalance_schedule.sum()} dates")
        return rebalance_schedule
    
    def calculate_transaction_costs(self,
                                  current_weights: pd.Series,
                                  target_weights: pd.Series,
                                  asset_volatilities: Optional[pd.Series] = None) -> float:
        """
        Calculate transaction costs for portfolio rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            asset_volatilities: Asset volatilities for cost adjustment
            
        Returns:
            Total transaction cost as fraction of portfolio value
        """
        # Calculate weight changes
        weight_changes = (target_weights - current_weights).abs()
        
        # Base transaction cost
        base_cost = self.transaction_cost_bps / 10000.0
        
        # Adjust costs based on volatility if provided
        if asset_volatilities is not None:
            vol_adjustment = 1.0 + asset_volatilities.fillna(0.0).clip(0, 0.1)
            adjusted_costs = base_cost * vol_adjustment
        else:
            adjusted_costs = pd.Series(base_cost, index=weight_changes.index)
        
        # Total transaction cost
        total_cost = (weight_changes * adjusted_costs).sum()
        
        return total_cost
    
    def run_portfolio_management(self,
                               signal_output: Dict[str, pd.DataFrame],
                               returns: pd.DataFrame,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimized portfolio management process with vectorized operations and reduced computational overhead.
        
        Args:
            signal_output: Output from ConrarianSignalGenerator
            returns: Asset returns DataFrame
            start_date: Start date for portfolio management
            end_date: End date for portfolio management
            
        Returns:
            Dictionary with complete portfolio management results
        """
        logger.info("Starting optimized portfolio management process")
        
        # Validate and align input data
        binary_signals = signal_output['binary_signals']
        volatility_data = signal_output.get('volatility', None)
        
        # Ensure data alignment
        common_columns = binary_signals.columns.intersection(returns.columns)
        common_index = binary_signals.index.intersection(returns.index)
        
        if len(common_columns) == 0 or len(common_index) == 0:
            raise ValueError("No overlapping data between signals and returns")
        
        # Filter to common data
        binary_signals = binary_signals.loc[common_index, common_columns]
        returns_aligned = returns.loc[common_index, common_columns]
        if volatility_data is not None:
            volatility_data = volatility_data.reindex_like(binary_signals)
        
        # Apply date filtering
        signal_dates = binary_signals.index
        if start_date:
            start_dt = pd.to_datetime(start_date)
            mask = signal_dates >= start_dt
            signal_dates = signal_dates[mask]
            binary_signals = binary_signals.loc[signal_dates]
            returns_aligned = returns_aligned.loc[signal_dates]
            if volatility_data is not None:
                volatility_data = volatility_data.loc[signal_dates]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            mask = signal_dates <= end_dt
            signal_dates = signal_dates[mask]
            binary_signals = binary_signals.loc[signal_dates]
            returns_aligned = returns_aligned.loc[signal_dates]
            if volatility_data is not None:
                volatility_data = volatility_data.loc[signal_dates]
        
        logger.info(f"Processing {len(signal_dates)} dates with {len(common_columns)} assets")
        
        # Pre-calculate volatility for all dates at once (vectorized)
        if self.risk_parity_optimizer.method == 'inverse_volatility':
            # For inverse volatility, we can vectorize the entire process
            portfolio_weights_history = self._vectorized_inverse_volatility_portfolio(
                binary_signals, volatility_data, returns_aligned
            )
        else:
            # For ERC and risk budgeting, use optimized date-by-date processing
            portfolio_weights_history = self._optimized_date_by_date_portfolio(
                binary_signals, volatility_data, returns_aligned
            )
        
        # Generate rebalancing schedule
        rebalance_schedule = self.generate_rebalancing_schedule(signal_dates)
        
        # Calculate transaction costs (vectorized)
        transaction_costs_history = self._calculate_vectorized_transaction_costs(
            portfolio_weights_history, rebalance_schedule
        )
        
        # Calculate portfolio returns (vectorized)
        portfolio_returns = self._calculate_portfolio_returns_vectorized(
            portfolio_weights_history, returns_aligned
        )
        
        # Calculate risk metrics only at strategic points (monthly instead of weekly)
        risk_metrics_history = self._calculate_strategic_risk_metrics(
            portfolio_returns, portfolio_weights_history, returns_aligned, signal_dates
        )
        
        # Compile results
        results = {
            'portfolio_weights': portfolio_weights_history,
            'portfolio_returns': portfolio_returns,
            'transaction_costs': transaction_costs_history,
            'rebalancing_dates': signal_dates[rebalance_schedule],
            'risk_metrics_history': risk_metrics_history,
            'final_risk_metrics': risk_metrics_history.get(signal_dates[-1], {}) if risk_metrics_history else {},
            'metadata': {
                'volatility_method': self.volatility_estimator.method,
                'risk_parity_method': self.risk_parity_optimizer.method,
                'rebalancing_frequency': self.rebalancing_frequency,
                'volatility_lookback': self.volatility_lookback,
                'correlation_lookback': self.correlation_lookback,
                'max_position_size': self.max_position_size,
                'target_volatility': self.target_volatility,
                'total_rebalancing_dates': rebalance_schedule.sum(),
                'total_transaction_costs': transaction_costs_history.sum(),
                'date_range': f"{signal_dates.min()} to {signal_dates.max()}"
            }
        }
        
        # Store in instance
        self.portfolio_history = results
        
        logger.info(f"Optimized portfolio management completed: {len(signal_dates)} dates processed")
        return results
    
    def _calculate_portfolio_returns_vectorized(self,
                                               weights: pd.DataFrame,
                                               asset_returns: pd.DataFrame) -> pd.Series:
        """
        Vectorized portfolio returns calculation with better data handling.
        
        Args:
            weights: Portfolio weights over time
            asset_returns: Asset returns over time
            
        Returns:
            Portfolio return series
        """
        # Data should already be aligned, but ensure compatibility
        if weights.index.equals(asset_returns.index) and weights.columns.equals(asset_returns.columns):
            # Perfect alignment - use vectorized operations
            lagged_weights = weights.shift(1).fillna(0.0)
            portfolio_returns = (lagged_weights * asset_returns).sum(axis=1)
            return portfolio_returns
        else:
            # Fallback to alignment
            common_dates = weights.index.intersection(asset_returns.index)
            common_assets = weights.columns.intersection(asset_returns.columns)
            
            if len(common_dates) == 0 or len(common_assets) == 0:
                logger.warning("No common data for portfolio return calculation")
                return pd.Series(0.0, index=weights.index)
            
            aligned_weights = weights.loc[common_dates, common_assets]
            aligned_returns = asset_returns.loc[common_dates, common_assets]
            
            lagged_weights = aligned_weights.shift(1).fillna(0.0)
            portfolio_returns = (lagged_weights * aligned_returns).sum(axis=1)
            
            # Reindex to original weights index
            return portfolio_returns.reindex(weights.index, fill_value=0.0)
    
    def _calculate_portfolio_returns(self,
                                   weights: pd.DataFrame,
                                   asset_returns: pd.DataFrame) -> pd.Series:
        """
        Legacy method - redirects to vectorized version.
        """
        return self._calculate_portfolio_returns_vectorized(weights, asset_returns)
    
    def optimize_portfolio_parameters(self,
                                    signal_output: Dict[str, pd.DataFrame],
                                    returns: pd.DataFrame,
                                    optimization_metric: str = 'sharpe_ratio',
                                    parameter_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio management parameters.
        
        Args:
            signal_output: Signal generation output
            returns: Asset returns
            optimization_metric: Metric to optimize ('sharpe_ratio', 'calmar_ratio', etc.)
            parameter_grid: Parameter combinations to test
            
        Returns:
            Dictionary with optimization results
        """
        if parameter_grid is None:
            parameter_grid = {
                'volatility_method': ['ewma', 'rolling'],
                'risk_parity_method': ['inverse_volatility', 'erc'],
                'volatility_lookback': [30, 60, 90],
                'target_volatility': [None, 0.10, 0.15]
            }
        
        logger.info(f"Starting parameter optimization for {optimization_metric}")
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        optimization_results = []
        best_score = float('-inf')
        best_params = None
        
        for param_combo in product(*param_values):
            param_dict = dict(zip(param_names, param_combo))
            logger.debug(f"Testing parameters: {param_dict}")
            
            try:
                # Create temporary portfolio manager with these parameters
                temp_manager = PortfolioManager(**param_dict)
                
                # Run portfolio management
                results = temp_manager.run_portfolio_management(
                    signal_output, returns
                )
                
                # Calculate performance metric
                portfolio_returns = results['portfolio_returns'].dropna()
                if len(portfolio_returns) < 252:  # Need at least 1 year of data
                    continue
                
                if optimization_metric == 'sharpe_ratio':
                    score = (portfolio_returns.mean() / portfolio_returns.std() 
                            * np.sqrt(252) if portfolio_returns.std() > 0 else 0)
                elif optimization_metric == 'calmar_ratio':
                    total_return = (1 + portfolio_returns).prod() - 1
                    max_dd = results['final_risk_metrics'].get('max_drawdown', 1.0)
                    score = total_return / max_dd if max_dd > 0 else 0
                else:
                    score = results['final_risk_metrics'].get(optimization_metric, 0)
                
                # Store results
                result_record = param_dict.copy()
                result_record['score'] = score
                result_record['total_return'] = (1 + portfolio_returns).prod() - 1
                result_record['volatility'] = portfolio_returns.std() * np.sqrt(252)
                result_record['max_drawdown'] = results['final_risk_metrics'].get('max_drawdown', 0)
                optimization_results.append(result_record)
                
                # Track best parameters
                if score > best_score:
                    best_score = score
                    best_params = param_dict.copy()
                    
            except Exception as e:
                logger.warning(f"Parameter optimization failed for {param_dict}: {str(e)}")
                continue
        
        optimization_df = pd.DataFrame(optimization_results)
        
        logger.info(f"Parameter optimization completed: tested {len(optimization_results)} combinations")
        logger.info(f"Best parameters: {best_params} with score: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_results': optimization_df,
            'optimization_metric': optimization_metric
        }


# Integration utilities for seamless backtesting framework connection
def integrate_with_backtesting_engine(portfolio_manager: PortfolioManager,
                                    backtesting_engine,
                                    signal_output: Dict[str, pd.DataFrame],
                                    returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Integration utility to connect PortfolioManager with BacktestingEngine.
    
    Args:
        portfolio_manager: Configured PortfolioManager instance
        backtesting_engine: BacktestingEngine instance
        signal_output: Signal generation output
        returns: Asset returns DataFrame
        
    Returns:
        Combined results from portfolio management and backtesting
    """
    logger.info("Integrating PortfolioManager with BacktestingEngine")
    
    # Run portfolio management
    portfolio_results = portfolio_manager.run_portfolio_management(
        signal_output, returns
    )
    
    # Extract weights for backtesting
    portfolio_weights = portfolio_results['portfolio_weights']
    
    # Run backtesting with portfolio manager weights
    backtest_results = backtesting_engine.run_backtest(
        signals=portfolio_weights,
        returns=returns,
        volatility=signal_output.get('volatility')
    )
    
    # Combine results
    integrated_results = {
        'portfolio_management': portfolio_results,
        'backtesting': backtest_results,
        'performance_comparison': {
            'pm_total_return': (1 + portfolio_results['portfolio_returns']).prod() - 1,
            'bt_total_return': (backtest_results['portfolio_value'].iloc[-1] / 
                              backtesting_engine.initial_capital - 1),
            'pm_sharpe': (portfolio_results['portfolio_returns'].mean() / 
                         portfolio_results['portfolio_returns'].std() * np.sqrt(252)),
            'bt_sharpe': backtest_results['metadata'].get('sharpe_ratio', 0)
        }
    }
    
    logger.info("Integration completed successfully")
    return integrated_results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("PortfolioManager module loaded successfully")
    print("Advanced Risk Parity Portfolio Management System ready for use.")
    print("Key features:")
    print("- Multiple volatility estimation methods (rolling, EWMA, GARCH, realized)")
    print("- Advanced risk parity optimization (inverse volatility, ERC, risk budgeting)")
    print("- Comprehensive risk monitoring (VaR, CVaR, concentration, correlation)")
    print("- Portfolio optimization with transaction cost awareness")
    print("- Seamless integration with contrarian signal generation framework")