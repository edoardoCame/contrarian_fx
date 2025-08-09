"""
Drawdown Filter Module for Commodities Analysis

This module provides functions to apply rolling drawdown filters to commodity futures data.
Adapted from the working reference implementation to handle daily equity curves.

Key Logic:
- Calculate rolling maximum of equity curve over specified lookback period
- Compute absolute drawdown as difference between rolling max and current equity
- Allow trading when absolute drawdown is below threshold
- Apply 1-day lag to prevent lookahead bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def apply_drawdown_filter(equity_series, returns_series, lookback_days, threshold_pct):
    """
    Apply rolling drawdown filter to equity curve and returns
    
    Args:
        equity_series (pd.Series): Equity curve with datetime index
        returns_series (pd.Series): Daily returns series (same index)
        lookback_days (int): Rolling window in days for drawdown calculation
        threshold_pct (float): Drawdown threshold as percentage (e.g., 0.05 for 5%)
        
    Returns:
        pd.DataFrame: DataFrame with all filter calculations and results
    """
    # Create working DataFrame
    df = pd.DataFrame({
        'equity': equity_series,
        'returns': returns_series
    })
    
    # Calculate rolling maximum over lookback period
    df['rolling_max'] = df['equity'].rolling(window=lookback_days, min_periods=1).max()
    
    # Calculate absolute drawdown (always >= 0)
    df['rolling_dd_abs'] = df['rolling_max'] - df['equity']
    
    # Calculate percentage drawdown (always <= 0)
    df['rolling_dd_pct'] = -(df['rolling_dd_abs'] / df['rolling_max']) * 100
    
    # Convert threshold from percentage to absolute value
    df['threshold_abs'] = df['rolling_max'] * threshold_pct
    df['threshold_pct'] = threshold_pct * 100
    
    # Determine if we can trade (when drawdown is below threshold)
    df['can_trade'] = df['rolling_dd_abs'] <= df['threshold_abs']
    
    # Apply 1-day lag to prevent lookahead bias
    df['can_trade_lagged'] = df['can_trade'].shift(1).fillna(False).astype(bool)
    
    # Apply filter to returns
    df['filtered_returns'] = np.where(df['can_trade_lagged'], df['returns'], 0.0)
    
    # Calculate filtered equity curve
    df['filtered_equity'] = (1 + df['filtered_returns']).cumprod()
    
    # Normalize filtered equity to start at same level as original
    if len(df) > 0:
        df['filtered_equity'] = df['filtered_equity'] * equity_series.iloc[0]
    
    return df


def calculate_filter_performance(filter_df, original_equity, filtered_equity):
    """
    Calculate performance metrics for filtered vs original strategy
    
    Args:
        filter_df (pd.DataFrame): DataFrame from apply_drawdown_filter
        original_equity (pd.Series): Original equity curve
        filtered_equity (pd.Series): Filtered equity curve
        
    Returns:
        dict: Performance metrics comparison
    """
    # Calculate returns
    original_returns = original_equity.pct_change().dropna()
    filtered_returns = filtered_equity.pct_change().dropna()
    
    # Calculate basic metrics
    def calc_metrics(equity_curve, returns_series):
        # Total return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Annualized volatility
        annual_vol = returns_series.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        sharpe = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0
        
        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        # Calculate CAGR
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        n_years = (end_date - start_date).days / 365.25
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/n_years) - 1 if n_years > 0 else 0
        cagr_pct = cagr * 100
        
        return {
            'total_return_pct': total_return,
            'cagr_pct': cagr_pct,
            'annual_vol_pct': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd
        }
    
    original_metrics = calc_metrics(original_equity, original_returns)
    filtered_metrics = calc_metrics(filtered_equity, filtered_returns)
    
    # Calculate filter-specific metrics
    filter_active_pct = (filter_df['can_trade_lagged'].sum() / len(filter_df)) * 100
    filter_changes = filter_df['can_trade_lagged'].diff().abs().sum()
    
    return {
        'original': original_metrics,
        'filtered': filtered_metrics,
        'filter_active_time_pct': filter_active_pct,
        'filter_changes': filter_changes
    }


def plot_drawdown_filter_analysis(filter_df, title_suffix=""):
    """
    Create comprehensive plots for drawdown filter analysis
    
    Args:
        filter_df (pd.DataFrame): DataFrame from apply_drawdown_filter
        title_suffix (str): Suffix for plot titles
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Equity curves comparison
    axes[0].plot(filter_df.index, filter_df['equity'], label='Original Equity', color='blue', alpha=0.7)
    axes[0].plot(filter_df.index, filter_df['filtered_equity'], label='Filtered Equity', color='red', linewidth=2)
    axes[0].set_title(f'Equity Curves Comparison {title_suffix}')
    axes[0].set_ylabel('Equity Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rolling drawdown with filter signals
    axes[1].fill_between(filter_df.index, filter_df['rolling_dd_pct'], 0, 
                        where=~filter_df['can_trade_lagged'], 
                        color='lightcoral', alpha=0.5, label='Filter OFF (No Investment)')
    axes[1].plot(filter_df.index, filter_df['rolling_dd_pct'], color='blue', linewidth=1, label='Rolling Drawdown (%)')
    axes[1].axhline(y=-filter_df['threshold_pct'].iloc[0], color='red', linestyle='--', 
                   label=f'Threshold ({-filter_df["threshold_pct"].iloc[0]:.1f}%)')
    axes[1].set_title(f'Rolling Drawdown with Filter Signals {title_suffix}')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Filter state and equity
    ax3_equity = axes[2]
    ax3_filter = ax3_equity.twinx()
    
    # Equity on left axis
    ax3_equity.plot(filter_df.index, filter_df['filtered_equity'], color='green', linewidth=2, label='Filtered Equity')
    ax3_equity.set_ylabel('Equity Value', color='green')
    ax3_equity.tick_params(axis='y', labelcolor='green')
    
    # Filter state on right axis
    filter_state = filter_df['can_trade_lagged'].astype(int)
    ax3_filter.fill_between(filter_df.index, 0, filter_state, alpha=0.3, color='orange', label='Filter Active')
    ax3_filter.set_ylabel('Filter State', color='orange')
    ax3_filter.set_ylim(-0.1, 1.1)
    ax3_filter.tick_params(axis='y', labelcolor='orange')
    
    axes[2].set_title(f'Filtered Equity with Filter State {title_suffix}')
    axes[2].set_xlabel('Date')
    
    # Combine legends
    lines1, labels1 = ax3_equity.get_legend_handles_labels()
    lines2, labels2 = ax3_filter.get_legend_handles_labels()
    ax3_equity.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax3_equity.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def optimize_drawdown_filter(equity_series, returns_series, lookback_range, threshold_range, 
                           optimize_metric='sharpe_ratio'):
    """
    Optimize drawdown filter parameters using grid search
    
    Args:
        equity_series (pd.Series): Original equity curve
        returns_series (pd.Series): Original returns series  
        lookback_range (tuple): (start, stop, step) for lookback days
        threshold_range (tuple): (start, stop, step) for threshold percentages
        optimize_metric (str): Metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
        
    Returns:
        pd.DataFrame: Grid search results sorted by optimization metric
    """
    results = []
    
    lookback_values = list(range(lookback_range[0], lookback_range[1], lookback_range[2]))
    threshold_values = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    
    print(f"Running grid search: {len(lookback_values)} lookback × {len(threshold_values)} threshold = {len(lookback_values) * len(threshold_values)} combinations")
    
    for lookback in lookback_values:
        for threshold in threshold_values:
            try:
                # Apply filter
                filter_df = apply_drawdown_filter(equity_series, returns_series, lookback, threshold)
                
                # Calculate performance
                perf_metrics = calculate_filter_performance(filter_df, equity_series, filter_df['filtered_equity'])
                
                # Store results
                result = {
                    'lookback_days': lookback,
                    'threshold_pct': threshold * 100,
                    'sharpe_ratio': perf_metrics['filtered']['sharpe_ratio'],
                    'total_return_pct': perf_metrics['filtered']['total_return_pct'],
                    'cagr_pct': perf_metrics['filtered']['cagr_pct'],
                    'max_drawdown_pct': perf_metrics['filtered']['max_drawdown_pct'],
                    'annual_vol_pct': perf_metrics['filtered']['annual_vol_pct'],
                    'filter_active_time_pct': perf_metrics['filter_active_time_pct'],
                    'filter_changes': perf_metrics['filter_changes']
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error with lookback={lookback}, threshold={threshold:.3f}: {e}")
                continue
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Sort by optimization metric (descending for most metrics, ascending for drawdown)
        ascending = optimize_metric == 'max_drawdown_pct'
        results_df = results_df.sort_values(optimize_metric, ascending=ascending).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
    
    return results_df