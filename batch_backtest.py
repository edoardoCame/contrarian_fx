import pandas as pd
import yfinance as yf
import os
from strategy_contrarian import strategy, rebalance_risk_parity
import pickle
import warnings
warnings.filterwarnings('ignore')

def download_and_save_data(tickers, start_date='2010-01-01', end_date='2025-12-31', data_dir='data/raw_data'):
    """
    Download forex data for all tickers and save to disk to avoid re-downloading
    """
    print("Downloading forex data...")
    data_dict = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('=', '_')}.pkl")
        
        # Check if data already exists
        if os.path.exists(file_path):
            print(f"Loading cached data for {ticker}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Downloading {ticker}...")
            try:
                data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
                if not data.empty:
                    # Save to disk
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"Downloaded and saved {ticker}")
                else:
                    print(f"No data available for {ticker}")
                    continue
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        data_dict[ticker] = data
    
    return data_dict

def batch_backtest_contrarian(data_dict, results_dir='data/backtest_results'):
    """
    Run contrarian strategy on all forex pairs and save equity curves
    """
    print("Running contrarian backtests...")
    equity_curves = pd.DataFrame()
    individual_results = {}
    
    for ticker, data in data_dict.items():
        try:
            print(f"Processing {ticker}...")
            
            # Apply contrarian strategy and get full data for proper metrics calculation
            # Line 13 in strategy.py: returns['strategy_returns'] = np.where(returns.shift(1) < 0, returns['Close'], 0)
            # The shift(1) ensures we use previous day's return, preventing lookahead bias
            full_data = strategy(data, timeframe='D', return_full_data=True)
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
    equity_curves_file = os.path.join(results_dir, 'all_equity_curves.pkl')
    with open(equity_curves_file, 'wb') as f:
        pickle.dump(equity_curves, f)
    
    # Save individual results  
    results_file = os.path.join(results_dir, 'individual_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(individual_results, f)
    
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
    portfolio_file = os.path.join('data/backtest_results', 'risk_parity_portfolio.pkl')
    with open(portfolio_file, 'wb') as f:
        pickle.dump(portfolio_df, f)
    
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