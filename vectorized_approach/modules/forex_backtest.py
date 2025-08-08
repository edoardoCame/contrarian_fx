import pandas as pd
import yfinance as yf
import os
from .strategy_contrarian import strategy, rebalance_risk_parity
import warnings
warnings.filterwarnings('ignore')

def download_and_save_data(tickers, start_date='2010-01-01', end_date='2025-12-31', data_dir='../forex/data/raw'):
    """
    Download forex data for all tickers and save to disk to avoid re-downloading
    """
    print("Downloading forex data...")
    data_dict = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('=', '_')}.parquet")
        
        # Check if data already exists
        if os.path.exists(file_path):
            print(f"Loading cached data for {ticker}")
            data = pd.read_parquet(file_path)
        else:
            print(f"Downloading {ticker}...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
                if not data.empty:
                    # Save to disk
                    data.to_parquet(file_path)
                    print(f"Downloaded and saved {ticker}")
                else:
                    print(f"No data available for {ticker}")
                    continue
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        data_dict[ticker] = data
    
    return data_dict

def batch_backtest_contrarian(data_dict, results_dir='../forex/data/results'):
    """
    Run contrarian strategy on all forex pairs and save equity curves
    """
    print("Running contrarian backtests...")
    equity_curves = pd.DataFrame()
    individual_results = {}
    
    for ticker, data in data_dict.items():
        try:
            print(f"Processing {ticker}...")
            
            # Add ticker as attribute to data for transaction cost calculation
            data.ticker = ticker
            
            # Apply contrarian strategy and get full data for proper metrics calculation
            # Line 13 in strategy.py: returns['strategy_returns'] = np.where(returns.shift(1) < 0, returns['Close'], 0)
            # The shift(1) ensures we use previous day's return, preventing lookahead bias
            full_data = strategy(data, timeframe='D', return_full_data=True, apply_transaction_costs=True)
            equity = full_data['Cumulative_Returns']
            strategy_returns = full_data['strategy_returns']
            
            if not equity.empty:
                equity.name = ticker
                equity_curves = pd.concat([equity_curves, equity], axis=1)
                
                # Save individual results with both equity and periodic returns for proper metrics
                individual_results[ticker] = {
                    'equity': equity,
                    'strategy_returns': strategy_returns,
                    'final_return': equity.iloc[-1] if len(equity) > 0 else 0
                }
                
                print(f"Completed {ticker} - Final return: {equity.iloc[-1]:.4f}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Save all equity curves
    equity_curves_file = os.path.join(results_dir, 'all_equity_curves.parquet')
    equity_curves.to_parquet(equity_curves_file)
    
    # Save individual results as separate parquet files 
    results_dir_individual = os.path.join(results_dir, 'individual_results')
    os.makedirs(results_dir_individual, exist_ok=True)
    
    # Create a summary dataframe for individual results
    summary_data = []
    for ticker, result in individual_results.items():
        # Save individual equity and returns as parquet
        result['equity'].to_frame(name='equity').to_parquet(
            os.path.join(results_dir_individual, f"{ticker.replace('=', '_')}_equity.parquet")
        )
        result['strategy_returns'].to_frame(name='strategy_returns').to_parquet(
            os.path.join(results_dir_individual, f"{ticker.replace('=', '_')}_returns.parquet")
        )
        
        # Add to summary
        summary_data.append({
            'ticker': ticker,
            'final_return': result['final_return']
        })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_parquet(os.path.join(results_dir, 'individual_results_summary.parquet'), index=False)
    
    print(f"Saved equity curves for {len(equity_curves.columns)} pairs")
    return equity_curves, individual_results

def create_risk_parity_portfolio(equity_curves, n=22, threshold=-0.1, shift=1):
    """
    Apply risk parity to the CONTRARIAN STRATEGY equity curves
    The function already implements:
    - Friday rebalancing (line 51: df['is_friday'] = df.index.dayofweek == 4)  
    - Lookahead bias prevention (line 94: shift parameter)
    """
    print("Creating risk parity portfolio on contrarian strategy equity curves...")
    
    # Convert equity curves (cumulative returns starting from 0) to price-like series (starting from 1)
    # This allows the rebalance_risk_parity function to work correctly
    strategy_prices = (1 + equity_curves).fillna(1)
    
    # Ensure we have a proper datetime index for dayofweek to work
    if not isinstance(strategy_prices.index, pd.DatetimeIndex):
        strategy_prices.index = pd.to_datetime(strategy_prices.index)
    
    # Drop rows with all NaN values that might cause issues
    strategy_prices = strategy_prices.dropna(how='all')
    
    print(f"Applying risk parity to {strategy_prices.shape[1]} contrarian strategies")
    print(f"Strategy price series range: {strategy_prices.min().min():.4f} to {strategy_prices.max().max():.4f}")
    
    # The rebalance_risk_parity function will now work on the STRATEGY performance, not original forex prices
    # This creates a portfolio that rebalances weekly among the different contrarian strategies
    portfolio_df = rebalance_risk_parity(strategy_prices, n=n, threshold=threshold, shift=shift)
    
    # Save portfolio results
    portfolio_file = os.path.join('../forex/data/results', 'risk_parity_portfolio.parquet')
    portfolio_df.to_parquet(portfolio_file)
    
    print(f"Risk parity portfolio created - Final equity: {portfolio_df['equity'].iloc[-1]:.4f}")
    return portfolio_df

def run_full_backtest():
    """
    Complete workflow: download data, backtest individual pairs, create risk parity portfolio
    """
    # Forex pairs from main.ipynb
    tickers = [
        'EURUSD=X', 'GBPUSD=X', 'EURJPY=X', 'EURCHF=X', 'AUDUSD=X', 'USDJPY=X', 
        'USDCHF=X', 'USDCAD=X', 'NZDUSD=X', 'GBPJPY=X', 'EURCAD=X', 'GBPCHF=X', 
        'AUDJPY=X', 'AUDCAD=X', 'AUDCHF=X', 'NZDJPY=X', 'NZDCAD=X', 'NZDCHF=X'
    ]
    
    # Step 1: Download and cache data
    data_dict = download_and_save_data(tickers)
    
    # Step 2: Run individual backtests
    equity_curves, individual_results = batch_backtest_contrarian(data_dict)
    
    # Step 3: Create risk parity portfolio
    portfolio_df = create_risk_parity_portfolio(equity_curves)
    
    return equity_curves, individual_results, portfolio_df

if __name__ == "__main__":
    equity_curves, individual_results, portfolio_df = run_full_backtest()
    print("\n=== BACKTEST COMPLETED ===")
    print(f"Individual strategies: {len(individual_results)}")
    print(f"Risk parity final equity: {portfolio_df['equity'].iloc[-1]:.4f}")