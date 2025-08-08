# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance project implementing contrarian trading strategies across multiple Forex currency pairs. The project consists of two main approaches:

1. **Vectorized Approach** (`vectorized_approach/`) - Educational and lightweight implementation
2. **Advanced Engine** (`advanced_engine/`) - High-performance institutional-grade backtesting framework

The codebase is primarily research-oriented, using Python with NumPy/Pandas for data processing, yfinance for data acquisition, and Jupyter notebooks for analysis.

## Development Commands

### Installing Dependencies
```bash
pip install yfinance pandas numpy matplotlib numba scipy scikit-learn cvxpy
```

### Jupyter Environment
```bash
jupyter notebook
```

### Vectorized Approach Commands
```bash
cd vectorized_approach
python batch_backtest.py    # Run complete backtest pipeline
```

### Advanced Engine Commands  
```bash
cd advanced_engine/modules
python backtesting_engine.py    # Run high-performance backtesting
python parameter_optimizer.py   # Parameter optimization with Numba
```

## Architecture Overview

### Vectorized Approach (`vectorized_approach/`)
- `strategy_contrarian.py`: Core contrarian logic and risk parity implementation
- `batch_backtest.py`: Batch processing workflow for multiple currency pairs
- `main.ipynb`: Educational notebook with step-by-step strategy explanation
- Data cached in `data/raw_data/` and results in `data/backtest_results/`

### Advanced Engine (`advanced_engine/`)
High-performance modular architecture with strict separation of concerns:

- `modules/backtesting_engine.py`: Numba-optimized vectorized backtesting with transaction costs
- `modules/portfolio_manager.py`: Advanced risk parity, volatility estimation, and risk monitoring
- `modules/signal_generator.py`: Contrarian signal generation with multiple implementations
- `modules/data_loader.py`: Robust data loading with validation and error handling
- `modules/parameter_optimizer.py`: Multi-objective parameter optimization framework
- `modules/performance_analyzer.py`: Comprehensive performance metrics and attribution
- `modules/results_manager.py`: Results persistence and metadata management

## Core Strategy Logic

### Contrarian Implementation
The fundamental contrarian logic prevents lookahead bias using proper lag:
```python
returns['strategy_returns'] = np.where(returns['Close'].shift(1) < 0, returns['Close'], 0)
```

### Risk Parity Portfolio
- Weekly rebalancing using inverse volatility weighting
- Performance filtering to exclude poorly performing strategies
- Proper lag implementation to avoid lookahead bias

## Data Architecture

### Currency Pairs Coverage
18-20 major Forex pairs: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURGBP, EURJPY, EURCHF, EURAUD, EURCAD, EURNZD, GBPJPY, GBPCHF, GBPAUD, GBPCAD, GBPNZD, AUDJPY, AUDCHF, NZDCAD, NZDCHF, NZDJPY

### Data Storage
- **Raw Data**: Parquet format in respective `data/` directories
- **Results**: Individual equity curves, portfolio results, and summary statistics
- **Caching**: Automatic caching to avoid re-downloading data

### Performance Optimization
- **Numba JIT Compilation**: Critical loops optimized with `@nb.jit(nopython=True)`
- **Vectorized Operations**: Pandas/NumPy vectorization for bulk processing
- **Memory Management**: Efficient data structures and lazy loading

## Key Development Patterns

### Lookahead Bias Prevention
Always use `.shift(1)` for signal generation:
- Signals generated from `t-1` data
- Trades executed at `t` close prices
- Returns calculated for `t` to `t+1` period

### Error Handling
- Robust error handling for data download failures
- Validation of data quality and completeness
- Graceful degradation when instruments fail

### Modular Design
- Clear separation between data, signals, portfolio construction, and analysis
- Each module has single responsibility
- Easy to extend with new strategies or risk models

## File Dependencies

### Core Module Relationships
- `signal_generator.py` ’ `backtesting_engine.py` ’ `portfolio_manager.py`
- `data_loader.py` provides data to all other modules
- `performance_analyzer.py` consumes results from backtesting and portfolio modules
- Notebooks depend on both vectorized and advanced engine modules

### Configuration Management
- Strategy parameters defined in respective modules
- Results metadata stored in SQLite database (`results/metadata.db`)
- Optimization results persisted in `results/optimizations/`

## Institutional Features (Advanced Engine)

- **Transaction Cost Modeling**: Realistic spread and slippage modeling
- **Risk Management**: VaR, CVaR, correlation monitoring, concentration limits
- **Performance Attribution**: Detailed breakdown of returns by source
- **Parallel Processing**: Multi-core parameter optimization
- **Production Readiness**: Comprehensive logging, error handling, and monitoring