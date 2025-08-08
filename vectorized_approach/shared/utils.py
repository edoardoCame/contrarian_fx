"""
Utility functions shared between forex and commodities analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(df):
    """Pulisce completamente i dati da infiniti e NaN"""
    # Sostituisci infiniti con NaN
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    # Forward fill per riempire NaN
    df_clean = df_clean.fillna(method='ffill')
    # Se ancora ci sono NaN all'inizio, riempili con 0
    df_clean = df_clean.fillna(0)
    
    # Verifica finale e sostituisce eventuali valori non finiti
    mask = ~np.isfinite(df_clean.values)
    if mask.any():
        df_clean = df_clean.mask(mask, 0)
    
    return df_clean

def load_individual_results(results_dir):
    """Load individual results from parquet files"""
    individual_results = {}
    individual_dir = results_dir / 'individual_results'
    
    # Load summary for ticker list
    summary_df = pd.read_parquet(results_dir / 'individual_results_summary.parquet')
    
    for _, row in summary_df.iterrows():
        ticker = row['ticker']
        ticker_clean = ticker.replace('=', '_')
        
        equity_file = individual_dir / f"{ticker_clean}_equity.parquet"
        returns_file = individual_dir / f"{ticker_clean}_returns.parquet"
        
        equity = pd.read_parquet(equity_file)['equity']
        strategy_returns = pd.read_parquet(returns_file)['strategy_returns']
        
        individual_results[ticker] = {
            'equity': equity,
            'strategy_returns': strategy_returns,
            'final_return': row['final_return']
        }
    
    return individual_results

def calculate_performance_metrics(equity_series):
    """Calculate standard performance metrics for an equity series"""
    if len(equity_series) <= 1:
        return {
            'total_return': 0,
            'annual_return': 0, 
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    total_return = equity_series.iloc[-1]
    
    # Calcola ritorni periodici puliti
    returns = equity_series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)  # Sostituisci infiniti con 0
    
    if len(returns) > 0:
        annual_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown - evita divisione per zero
        rolling_max = equity_series.expanding().max()
        # Usa solo valori dove rolling_max > 1e-8 per evitare divisione per zero
        drawdown = np.where(rolling_max > 1e-8, (equity_series - rolling_max) / rolling_max, 0)
        drawdown = pd.Series(drawdown, index=equity_series.index)
        max_drawdown = drawdown.min()
        # Se max_dd è ancora infinito o NaN, impostalo a 0
        if not np.isfinite(max_drawdown):
            max_drawdown = 0
    else:
        annual_return = volatility = sharpe_ratio = max_drawdown = 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return, 
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def setup_matplotlib():
    """Setup standard matplotlib configuration"""
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.formatter.limits'] = [-3, 3]  # Evita notazione scientifica per range limitati
    return plt