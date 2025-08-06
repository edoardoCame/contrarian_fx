# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance project that implements and backtests a contrarian trading strategy across multiple Forex currency pairs. The project is primarily research-oriented, using Python and Jupyter notebooks for analysis and visualization.

## Key Components

### Core Strategy Implementation (`strategy_contrarian.py`)
- `strategy()`: Implements the main contrarian logic - enters long positions when previous day's return is negative
- `rebalance_risk_parity()`: Advanced portfolio management function that applies inverse volatility weighting with performance filtering

### Batch Processing (`batch_backtest.py`) 
- `download_and_save_data()`: Downloads and caches Forex data from Yahoo Finance using yfinance
- `batch_backtest_contrarian()`: Runs the contrarian strategy across all currency pairs
- `create_risk_parity_portfolio()`: Creates a meta-portfolio by applying risk parity to individual strategy equity curves
- `run_full_backtest()`: Complete workflow orchestration

### Analysis Notebooks
- `main.ipynb`: Educational notebook explaining contrarian strategy concepts with step-by-step implementation
- `lookback_analysis.ipynb`: Advanced analysis notebook (currently modified)
- `analysis_clean.ipynb`: Additional analysis notebook

## Development Commands

### Running the Complete Backtest
```bash
python batch_backtest.py
```
This downloads data for 18 Forex pairs, runs contrarian backtests, and creates a risk parity portfolio.

### Installing Dependencies
Based on README.md requirements:
```bash
pip install yfinance pandas numpy matplotlib
```

### Starting Jupyter Environment
```bash
jupyter notebook
```
Then open `main.ipynb` for the educational walkthrough.

## Data Structure

### Input Data
- Raw Forex data cached in `data/raw_data/` as parquet files (format: `EURUSD_X.parquet`)
- Covers 18 major currency pairs from 2010-2025

### Output Data  
- `data/backtest_results/all_equity_curves.parquet`: Combined equity curves for all pairs
- `data/backtest_results/individual_results/`: Directory containing individual equity and returns as separate parquet files
- `data/backtest_results/individual_results_summary.parquet`: Summary of individual results
- `data/backtest_results/risk_parity_portfolio.parquet`: Risk parity portfolio results

## Strategy Logic

### Contrarian Implementation
The core contrarian logic in `strategy_contrarian.py:13`:
```python
returns['strategy_returns'] = np.where(returns['Close'].shift(1) < 0, returns['Close'], 0)
```
Uses `shift(1)` to prevent lookahead bias - decisions based on previous day's performance.

### Risk Parity Portfolio
- Rebalances weekly among contrarian strategies based on inverse volatility weighting
- Filters out strategies with negative recent performance
- Implements proper lag to avoid lookahead bias (`shift` parameter)

## Architecture Notes

- **Caching System**: Data is automatically cached to avoid re-downloading on subsequent runs
- **Error Handling**: Robust error handling for failed downloads or missing data
- **Modular Design**: Strategy logic separated from data processing and portfolio management
- **Bias Prevention**: Careful use of `shift()` operations to prevent lookahead bias in backtests

## File Dependencies

When modifying strategy logic:
1. Update `strategy_contrarian.py` for core strategy changes
2. Modify `batch_backtest.py` for data processing or portfolio logic
3. Update notebooks for analysis and visualization changes

The notebooks depend on both Python modules, so ensure compatibility when making changes to the core strategy functions.