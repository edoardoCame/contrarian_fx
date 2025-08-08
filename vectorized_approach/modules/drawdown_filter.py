import pandas as pd
import numpy as np

def calculate_rolling_drawdown(equity_curve, lookback_days=90):
    """
    Calculate rolling drawdown over specified lookback period using vectorized operations.
    
    Parameters:
    - equity_curve: pandas Series of equity values (cumulative returns + 1)
    - lookback_days: Number of days to look back for rolling max calculation
    
    Returns:
    - rolling_drawdown: pandas Series of rolling drawdown values (negative values)
    """
    # Calculate rolling maximum over lookback period
    rolling_max = equity_curve.rolling(window=lookback_days, min_periods=1).max()
    
    # Calculate drawdown as (current - rolling_max) / rolling_max
    rolling_drawdown = (equity_curve - rolling_max) / rolling_max
    
    return rolling_drawdown

def create_drawdown_filter(equity_curve, lookback_days=90, threshold_pct=0.15, lag_days=1):
    """
    Create a binary filter based on rolling drawdown threshold.
    
    Parameters:
    - equity_curve: pandas Series of equity values
    - lookback_days: Rolling window for drawdown calculation (default 90 days)
    - threshold_pct: Drawdown threshold as decimal (0.15 = 15% drawdown)
    - lag_days: Days to lag the filter to avoid lookahead bias (default 1)
    
    Returns:
    - filter_series: pandas Series of 1s (invest) and 0s (don't invest)
    """
    # Calculate rolling drawdown
    rolling_dd = calculate_rolling_drawdown(equity_curve, lookback_days)
    
    # Create filter: invest when drawdown is above threshold (less negative)
    # If rolling_dd > -threshold_pct, then invest (1), else don't invest (0)
    investment_filter = (rolling_dd > -threshold_pct).astype(int)
    
    # Apply lag to prevent lookahead bias - use yesterday's signal for today's decision
    lagged_filter = investment_filter.shift(lag_days).fillna(1)  # Start with investment allowed
    
    return lagged_filter

def apply_filter_to_returns(returns_series, filter_series):
    """
    Apply binary filter to return series.
    
    Parameters:
    - returns_series: pandas Series of period returns
    - filter_series: pandas Series of 1s (invest) and 0s (don't invest)
    
    Returns:
    - filtered_returns: pandas Series where returns are 0 when filter is 0
    """
    # When filter = 0, returns = 0 (no investment)
    # When filter = 1, returns = original returns (full investment)
    filtered_returns = returns_series * filter_series
    
    return filtered_returns

def apply_drawdown_filter_to_equity(equity_curve, lookback_days=90, threshold_pct=0.15, lag_days=1):
    """
    Complete workflow: apply drawdown filter to an equity curve.
    
    Parameters:
    - equity_curve: pandas Series of equity values (cumulative performance)
    - lookback_days: Rolling window for drawdown calculation
    - threshold_pct: Drawdown threshold as decimal (e.g., 0.15 for 15%)
    - lag_days: Days to lag the filter signal
    
    Returns:
    - dict with:
        - 'filtered_equity': New equity curve with filter applied
        - 'original_equity': Original equity curve  
        - 'filter_signal': Binary filter signal
        - 'rolling_drawdown': Rolling drawdown series
    """
    # Calculate period returns from equity curve
    returns = equity_curve.pct_change().fillna(0)
    
    # Create drawdown filter
    filter_signal = create_drawdown_filter(equity_curve, lookback_days, threshold_pct, lag_days)
    
    # Apply filter to returns
    filtered_returns = apply_filter_to_returns(returns, filter_signal)
    
    # Calculate new equity curve from filtered returns
    filtered_equity = (1 + filtered_returns).cumprod()
    
    # Calculate rolling drawdown for analysis
    rolling_dd = calculate_rolling_drawdown(equity_curve, lookback_days)
    
    return {
        'filtered_equity': filtered_equity,
        'original_equity': equity_curve,
        'filter_signal': filter_signal,
        'rolling_drawdown': rolling_dd,
        'original_returns': returns,
        'filtered_returns': filtered_returns
    }

def analyze_filter_performance(original_equity, filtered_equity):
    """
    Compare performance metrics between original and filtered equity curves.
    
    Parameters:
    - original_equity: pandas Series of original equity curve
    - filtered_equity: pandas Series of filtered equity curve
    
    Returns:
    - dict with performance comparison metrics
    """
    
    def calculate_metrics(equity_series):
        """Helper function to calculate standard performance metrics."""
        returns = equity_series.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], 0)
        
        if len(returns) == 0:
            return {'total_return': 0, 'annual_return': 0, 'volatility': 0, 
                   'sharpe_ratio': 0, 'max_drawdown': 0}
        
        total_return = equity_series.iloc[-1] - 1.0
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown calculation
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    original_metrics = calculate_metrics(original_equity)
    filtered_metrics = calculate_metrics(filtered_equity)
    
    return {
        'original': original_metrics,
        'filtered': filtered_metrics,
        'improvement': {
            'total_return': filtered_metrics['total_return'] - original_metrics['total_return'],
            'sharpe_ratio': filtered_metrics['sharpe_ratio'] - original_metrics['sharpe_ratio'],
            'max_drawdown': filtered_metrics['max_drawdown'] - original_metrics['max_drawdown']
        }
    }

def optimize_filter_parameters(equity_curve, lookback_range=(30, 150, 30), 
                              threshold_range=(0.05, 0.30, 0.05), optimize_metric='sharpe_ratio'):
    """
    Simple parameter optimization for drawdown filter.
    
    Parameters:
    - equity_curve: pandas Series to optimize on
    - lookback_range: tuple (start, stop, step) for lookback days
    - threshold_range: tuple (start, stop, step) for threshold percentage
    - optimize_metric: metric to optimize ('sharpe_ratio', 'total_return', 'max_drawdown')
    
    Returns:
    - dict with best parameters and results
    """
    best_metric = -np.inf if optimize_metric in ['sharpe_ratio', 'total_return'] else np.inf
    best_params = None
    results = []
    
    lookback_values = range(lookback_range[0], lookback_range[1], lookback_range[2])
    threshold_values = np.arange(threshold_range[0], threshold_range[1], threshold_range[2])
    
    for lookback in lookback_values:
        for threshold in threshold_values:
            try:
                # Apply filter with current parameters
                result = apply_drawdown_filter_to_equity(equity_curve, lookback, threshold)
                metrics = analyze_filter_performance(result['original_equity'], 
                                                   result['filtered_equity'])
                
                current_metric = metrics['filtered'][optimize_metric]
                
                # Track results
                results.append({
                    'lookback_days': lookback,
                    'threshold_pct': threshold,
                    'metric_value': current_metric,
                    **metrics['filtered']
                })
                
                # Check if this is the best so far
                if optimize_metric in ['sharpe_ratio', 'total_return']:
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_params = (lookback, threshold)
                else:  # max_drawdown (want less negative)
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_params = (lookback, threshold)
                        
            except Exception as e:
                print(f"Error with lookback={lookback}, threshold={threshold}: {e}")
                continue
    
    return {
        'best_params': best_params,
        'best_metric_value': best_metric,
        'all_results': pd.DataFrame(results),
        'optimize_metric': optimize_metric
    }