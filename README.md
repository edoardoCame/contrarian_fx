# Contrarian FX Trading System

> Advanced quantitative finance research framework implementing contrarian trading strategies across Forex and commodity markets

## Abstract

This repository implements a comprehensive quantitative finance framework for researching and backtesting contrarian trading strategies across multiple asset classes. The system features dual architectures: an educational vectorized approach for learning and rapid prototyping, and a production-grade advanced engine optimized for institutional-level research and deployment.

## Theoretical Foundation

### Contrarian Strategy Methodology

The contrarian trading approach is based on the **mean reversion hypothesis** in financial markets, which posits that asset prices tend to revert to their long-term equilibrium following periods of extreme movement. This strategy capitalizes on temporary market overreactions by:

1. **Identifying Underperformers**: Systematically selecting assets with the worst N-period performance
2. **Risk Parity Allocation**: Applying inverse volatility weighting to ensure equal risk contribution across positions
3. **Temporal Lag Implementation**: Preventing lookahead bias through strict temporal separation between signal generation and execution

The mathematical foundation relies on the empirical observation that:
```
E[R(t+1) | R(t) < -threshold] > E[R(t+1)]
```
where R(t) represents period returns and the negative threshold identifies oversold conditions.

### Portfolio Construction

The system implements **Equal Risk Contribution (ERC)** portfolio construction, where asset weights w_i satisfy:
```
RC_i = w_i * ∂σ_p/∂w_i = σ_p/N ∀i
```
where RC_i is the risk contribution of asset i, σ_p is portfolio volatility, and N is the number of assets.

## System Architecture

The repository implements a **dual-architecture design** optimizing for both educational accessibility and production performance:

### 1. Vectorized Approach (`vectorized_approach/`)
**Purpose**: Educational framework and rapid prototyping  
**Design Philosophy**: Clarity and comprehensibility over optimization  
**Target Users**: Researchers, students, and strategy developers

### 2. Advanced Engine (`advanced_engine/`)
**Purpose**: Production-grade institutional backtesting framework  
**Design Philosophy**: Performance, scalability, and professional deployment  
**Target Users**: Quantitative funds, institutional traders, and production systems

## Repository Structure

```
contrarian_fx/
├── README.md                           # Main documentation (this file)
├── CLAUDE.md                          # Development instructions for AI assistance
├── FOREX_DATA_SYSTEM_README.md        # Data system documentation
├── PERFORMANCE_OPTIMIZATION_REPORT.md # Performance analysis report
├── forex_data_collection.log          # Data collection logs
│
├── vectorized_approach/               # Educational implementation
│   ├── CLAUDE.md                      # Module-specific documentation
│   ├── test_structure.py             # System integrity validation
│   ├── modules/                       # Core strategy modules
│   │   ├── strategy_contrarian.py    # Contrarian strategy implementation
│   │   ├── forex_backtest.py         # Forex backtesting engine
│   │   ├── commodities_backtest.py   # Commodities backtesting engine
│   │   └── daily_operations_analyzer.py # Daily operations analysis
│   ├── shared/                        # Shared utilities
│   │   └── utils.py                   # Common utility functions
│   ├── forex/                         # Forex analysis framework
│   │   ├── data/                      # Forex data storage
│   │   │   ├── raw/                   # Raw price data (.parquet)
│   │   │   └── results/               # Backtest results and analysis
│   │   └── notebooks/                 # Educational notebooks
│   │       ├── fx_main_educational.ipynb    # Primary educational notebook
│   │       ├── fx_analysis.ipynb            # Strategy analysis
│   │       ├── fx_lookback_analysis.ipynb   # Parameter optimization
│   │       └── fx_daily_operations.ipynb    # Operational analysis
│   └── commodities/                   # Commodities analysis framework
│       ├── data/                      # Commodities data storage
│       │   ├── raw/                   # Raw futures price data
│       │   └── results/               # Backtest results and portfolios
│       └── notebooks/                 # Analysis notebooks
│           ├── commodities_analysis.ipynb      # Primary analysis
│           └── commodities_lookback_analysis.ipynb # Parameter studies
│
└── advanced_engine/                   # Production-grade framework
    ├── README.md                      # Advanced engine documentation
    ├── OPTIMIZATION_SUMMARY.md       # Optimization results summary
    ├── data/                          # High-frequency data storage
    │   ├── *.parquet                  # Individual currency pair data
    │   ├── all_pairs_data.parquet     # Consolidated price dataset
    │   └── all_pairs_returns.parquet  # Consolidated returns dataset
    ├── modules/                       # Professional-grade modules
    │   ├── data_loader.py            # Robust data management system
    │   ├── signal_generator.py       # Advanced signal generation
    │   ├── portfolio_manager.py      # Institutional portfolio management
    │   ├── backtesting_engine.py     # High-performance backtesting
    │   ├── performance_analyzer.py   # Comprehensive metrics analysis
    │   ├── parameter_optimizer.py    # Multi-objective optimization
    │   └── results_manager.py        # Results persistence and metadata
    ├── notebooks/                     # Professional analysis notebooks
    │   ├── individual_strategy_analysis.ipynb # Single-asset analysis
    │   └── portfolio_analysis.ipynb           # Portfolio-level analysis
    └── results/                       # Generated outputs and analysis
        ├── backtests/                 # Backtest result files
        ├── optimizations/             # Parameter optimization results
        ├── exports/                   # Data exports and reports
        ├── temp/                      # Temporary processing files
        └── metadata.db                # SQLite metadata database
```

## Asset Coverage

### Forex Markets (20+ Major Pairs)
- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Cross Pairs**: EURGBP, EURJPY, EURCHF, EURAUD, EURCAD, EURNZD
- **Exotic Crosses**: GBPJPY, GBPCHF, GBPAUD, GBPCAD, GBPNZD, AUDJPY, AUDCHF
- **Additional**: NZDCAD, NZDCHF, NZDJPY

### Commodity Futures (15+ Instruments)
- **Energy**: Crude Oil (CL), Brent Oil (BZ), Natural Gas (NG), Heating Oil (HO), RBOB Gas (RB)
- **Precious Metals**: Gold (GC), Silver (SI), Palladium (PA)
- **Base Metals**: Copper (HG)
- **Agricultural**: Corn (ZC), Wheat (ZW), Soybeans (ZS), Cotton (CT), Sugar (SB), Cocoa (CC)

## Implementation Features

### Bias Prevention Architecture
- **Strict Temporal Separation**: All signals generated at T-1 for execution at T
- **Lookahead Validation**: Mathematical guarantees against future information leakage
- **Lag Implementation**: Consistent `.shift()` operations across all data transformations

### Performance Optimization
- **Vectorized Operations**: NumPy/Pandas optimizations for bulk processing
- **Numba JIT Compilation**: Critical loops optimized with `@nb.jit(nopython=True)`
- **Memory Management**: Efficient data structures and lazy loading patterns
- **Parallel Processing**: Multi-core parameter optimization and backtesting

### Risk Management Framework
- **Value at Risk (VaR)**: Historical and parametric VaR calculations
- **Conditional VaR (CVaR)**: Expected Shortfall risk metrics
- **Drawdown Analysis**: Maximum drawdown and recovery period analysis
- **Correlation Monitoring**: Dynamic correlation tracking and concentration limits
- **Volatility Targeting**: Dynamic position sizing based on realized volatility

## Quick Start Guide

### Prerequisites
```bash
# Core dependencies
pip install yfinance pandas numpy matplotlib numba scipy scikit-learn cvxpy

# Optional: Advanced optimization
pip install jupyter notebook plotly seaborn
```

### Educational Path (Vectorized Approach)
```bash
# Start with educational notebooks
cd vectorized_approach/forex/notebooks
jupyter notebook fx_main_educational.ipynb

# Run complete forex backtest
cd ../../
python -m modules.forex_backtest

# Run commodities analysis
cd commodities/notebooks
jupyter notebook commodities_analysis.ipynb
```

### Professional Path (Advanced Engine)
```bash
# Launch professional analysis
cd advanced_engine/notebooks
jupyter notebook portfolio_analysis.ipynb

# Run parameter optimization
cd ../modules
python parameter_optimizer.py

# Execute high-performance backtesting
python backtesting_engine.py
```

## Key Algorithms and Methods

### Signal Generation Algorithm
```python
# Contrarian signal generation with lookahead prevention
def generate_contrarian_signals(returns_data, n_worst=5, lookback=20):
    """
    Generate contrarian signals by identifying worst N performers
    over M-day lookback period with strict temporal lag.
    """
    # Ensure no lookahead bias: use t-1 data for t decisions
    lagged_returns = returns_data.shift(1)
    rolling_performance = lagged_returns.rolling(window=lookback).sum()
    
    # Select worst N performers
    signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    for date in rolling_performance.index:
        if pd.notna(rolling_performance.loc[date]).sum() >= n_worst:
            worst_performers = rolling_performance.loc[date].nsmallest(n_worst).index
            signals.loc[date, worst_performers] = 1
    
    return signals.fillna(0)
```

### Risk Parity Implementation
```python
# Equal Risk Contribution portfolio construction
def calculate_risk_parity_weights(returns_covariance, risk_budget=None):
    """
    Calculate risk parity weights using Equal Risk Contribution.
    Solves: min 0.5 * w'Σw subject to Σ(w_i * (Σw)_i / σ_p - 1/N)^2
    """
    n_assets = len(returns_covariance)
    if risk_budget is None:
        risk_budget = np.array([1/n_assets] * n_assets)
    
    # Optimization problem setup
    def risk_budget_objective(weights, cov_matrix, risk_budgets):
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        contrib = weights * marginal_contrib / portfolio_vol
        return np.sum((contrib - risk_budgets) ** 2)
    
    # Solve optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.array([1/n_assets] * n_assets)
    
    result = minimize(risk_budget_objective, x0, 
                     args=(returns_covariance, risk_budget),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
```

## Performance Metrics and Analysis

### Comprehensive Performance Suite
- **Return Metrics**: Total Return, Annualized Return, Excess Return
- **Risk Metrics**: Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Drawdown Analysis**: Maximum Drawdown, Average Drawdown, Recovery Time
- **Higher Moments**: Skewness, Kurtosis, Tail Risk Measures
- **Risk-Adjusted**: Information Ratio, Treynor Ratio, Jensen's Alpha
- **Benchmarking**: Against equal-weight, buy-and-hold, and market indices

### Attribution Analysis
- **Factor Decomposition**: Systematic vs. idiosyncratic risk attribution
- **Asset Contribution**: Individual asset contribution to portfolio performance
- **Time-Series Attribution**: Performance attribution across different market regimes
- **Risk Budgeting**: Actual vs. target risk contribution analysis

## Academic References

### Core Literature
1. **Mean Reversion Theory**: Fama, E. F., & French, K. R. (1988). "Permanent and temporary components of stock prices." Journal of Political Economy, 96(2), 246-273.

2. **Risk Parity Framework**: Roncalli, T. (2013). "Introduction to risk parity and budgeting." Chapman and Hall/CRC.

3. **Portfolio Construction**: Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The properties of equally weighted risk contribution portfolios." Journal of Portfolio Management, 36(4), 60-70.

4. **Backtesting Methodology**: Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." Notices of the AMS, 61(5), 458-471.

### Technical Implementation
- [Yahoo Finance API Documentation](https://github.com/ranaroussi/yfinance)
- [Pandas Financial Analysis](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [NumPy Scientific Computing](https://numpy.org/doc/stable/)
- [Numba JIT Compilation](https://numba.pydata.org/)
- [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)

## Production Deployment

This system is designed for institutional deployment with:
- **Scalable Architecture**: Handles 20+ years of daily data across 35+ instruments
- **Real-time Capability**: Sub-second signal generation and portfolio updates
- **Risk Controls**: Built-in position limits, correlation monitoring, and drawdown controls
- **Data Integrity**: Comprehensive validation and bias prevention mechanisms
- **Monitoring**: Detailed logging, performance tracking, and alert systems

## Legal Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. Past performance does not guarantee future results. All trading strategies involve substantial risk of loss. Users should:

1. Thoroughly understand the methodology before implementation
2. Conduct extensive out-of-sample testing
3. Consider transaction costs, slippage, and market impact
4. Implement appropriate risk management protocols
5. Consult with qualified financial professionals

The authors assume no responsibility for any financial losses incurred through the use of this software.

## License

MIT License - See LICENSE file for full details.

---

**Developed for academic research and institutional quantitative finance applications.**
