# Advanced Risk Parity Portfolio Management System

A comprehensive, institutional-grade portfolio management system designed for forex trading with contrarian signals and advanced risk parity weighting methodologies.

## 🎯 Overview

This system combines contrarian signal generation with sophisticated risk parity portfolio management to create a robust, production-ready trading framework. It features zero lookahead bias, comprehensive risk monitoring, and advanced optimization capabilities.

## 🚀 Key Features

### Portfolio Management
- **Multiple Volatility Estimation Methods**: Rolling, EWMA, GARCH, Realized Volatility
- **Advanced Risk Parity Algorithms**: Inverse Volatility, Equal Risk Contribution (ERC), Risk Budgeting
- **Comprehensive Risk Monitoring**: VaR, CVaR, Concentration Risk, Correlation Analysis
- **Portfolio Optimization**: Transaction cost-aware rebalancing with volatility targeting
- **Position Management**: Dynamic position sizing with correlation-based adjustments

### Risk Management
- **Real-time Risk Metrics**: Portfolio volatility, drawdown monitoring, correlation tracking
- **Risk Limit Framework**: Configurable risk limits with automatic breach detection
- **Stress Testing**: Scenario analysis and Monte Carlo simulation capabilities
- **Performance Attribution**: Factor-based return decomposition and contribution analysis

### Technical Excellence
- **Zero Lookahead Bias**: Strict temporal alignment ensuring no future data usage
- **High-Performance Computing**: Numba-optimized calculations for large portfolios
- **Modular Architecture**: Clean separation of concerns with extensible components
- **Comprehensive Testing**: Full test suite with integration and validation tests

## 📁 System Architecture

```
advanced_engine/
├── modules/
│   ├── portfolio_manager.py          # Core portfolio management system
│   ├── portfolio_validation.py       # Testing and validation suite
│   ├── signal_generator.py           # Contrarian signal generation
│   ├── backtesting_engine.py         # High-performance backtesting
│   ├── data_loader.py                # Data loading and validation
│   ├── performance_analyzer.py       # Performance analysis tools
│   └── results_manager.py            # Results storage and management
├── data/                             # Forex data storage
├── results/                          # Output directory for results
├── advanced_portfolio_example.py     # Complete workflow example
└── ADVANCED_PORTFOLIO_README.md      # This documentation
```

## 🔧 Installation and Setup

### Requirements

```python
# Core dependencies
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
numba>=0.56.0
scikit-learn>=1.1.0
cvxpy>=1.2.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Data handling
pyarrow>=9.0.0
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd advanced_engine

# Install dependencies
pip install -r requirements.txt

# Validate installation
python -c "from modules.portfolio_manager import PortfolioManager; print('✅ Installation successful')"
```

## 🏁 Quick Start

### Basic Usage

```python
from modules.portfolio_manager import PortfolioManager
from modules.signal_generator import ConrarianSignalGenerator
from modules.data_loader import ForexDataLoader

# 1. Load data
loader = ForexDataLoader("data/")
prices = loader.load_unified_prices()
returns = loader.calculate_returns_from_prices(prices)

# 2. Generate contrarian signals
signal_generator = ConrarianSignalGenerator(
    n_worst_performers=5,
    lookback_days=20,
    volatility_lookback=60
)
signals = signal_generator.generate_signals(prices, returns)

# 3. Apply portfolio management
portfolio_manager = PortfolioManager(
    volatility_method='ewma',
    risk_parity_method='inverse_volatility',
    target_volatility=0.12,
    max_position_size=0.25
)
results = portfolio_manager.run_portfolio_management(signals, returns)

# 4. Analyze results
print(f"Sharpe Ratio: {results['final_risk_metrics'].get('sharpe_ratio', 0):.2f}")
print(f"Max Drawdown: {results['final_risk_metrics'].get('max_drawdown', 0):.2%}")
```

### Complete Workflow

```python
from advanced_portfolio_example import AdvancedPortfolioWorkflow

# Run complete workflow
workflow = AdvancedPortfolioWorkflow()
results = workflow.run_complete_workflow()

# Results automatically saved to results/ directory
# Includes comprehensive reports, visualizations, and data files
```

## 📊 Core Components

### 1. VolatilityEstimator

Advanced volatility forecasting with multiple methodologies:

```python
from modules.portfolio_manager import VolatilityEstimator

# EWMA volatility (recommended for forex)
vol_estimator = VolatilityEstimator('ewma')
volatility = vol_estimator.estimate_volatility(returns, window=60)

# Available methods: 'rolling', 'ewma', 'garch', 'realized'
```

**Features:**
- Exponentially Weighted Moving Average (EWMA) with configurable decay
- GARCH(1,1) implementation for volatility clustering
- Realized volatility using high-frequency patterns
- Automatic lookahead bias prevention with 1-day lag

### 2. RiskParityOptimizer

Sophisticated risk parity weight calculation:

```python
from modules.portfolio_manager import RiskParityOptimizer

# Equal Risk Contribution (ERC) optimization
optimizer = RiskParityOptimizer('erc')
weights = optimizer.calculate_risk_parity_weights(
    selected_assets=selected_assets,
    volatility=volatility_estimates,
    correlation_matrix=correlation_matrix
)

# Available methods: 'inverse_volatility', 'erc', 'risk_budgeting'
```

**Algorithms:**
- **Inverse Volatility**: Simple 1/σ weighting for computational efficiency
- **Equal Risk Contribution**: Optimization-based approach ensuring equal risk allocation
- **Risk Budgeting**: Custom risk allocation with specified risk budgets

### 3. RiskMonitor

Comprehensive real-time risk monitoring:

```python
from modules.portfolio_manager import RiskMonitor

risk_monitor = RiskMonitor(confidence_levels=[0.95, 0.99])
risk_metrics = risk_monitor.calculate_portfolio_risk_metrics(
    portfolio_returns, portfolio_weights, asset_returns
)

# Key metrics: VaR, CVaR, concentration, correlation, drawdown
```

**Risk Metrics:**
- **Value at Risk (VaR)**: Historical and parametric methods
- **Conditional VaR**: Expected shortfall beyond VaR threshold
- **Concentration Risk**: Herfindahl-Hirschman Index and effective assets
- **Correlation Analysis**: Portfolio correlation structure monitoring

### 4. PortfolioManager

Main orchestration class integrating all components:

```python
from modules.portfolio_manager import PortfolioManager

pm = PortfolioManager(
    volatility_method='ewma',           # Volatility estimation method
    risk_parity_method='erc',           # Risk parity algorithm
    target_volatility=0.15,             # Annual volatility target
    max_position_size=0.25,             # Maximum single position
    rebalancing_frequency='weekly',      # Rebalancing schedule
    transaction_cost_bps=2.0            # Transaction costs (bps)
)
```

**Advanced Features:**
- **Volatility Targeting**: Automatic scaling to achieve target volatility
- **Position Constraints**: Maximum/minimum position sizes with concentration limits
- **Transaction Cost Optimization**: Cost-aware rebalancing decisions
- **Parameter Optimization**: Grid search for optimal parameter combinations

## 🎯 Risk Management Framework

### Risk Parity Implementation

The system implements multiple risk parity methodologies:

1. **Inverse Volatility Weighting**
   ```
   w_i = (1/σ_i) / Σ(1/σ_j)
   ```

2. **Equal Risk Contribution (ERC)**
   ```
   Minimize: Σ(RC_i - (σ_p/N))²
   where RC_i = w_i * (∂σ_p/∂w_i)
   ```

3. **Risk Budgeting**
   ```
   RC_i = b_i * σ_p
   where b_i is the target risk budget for asset i
   ```

### Risk Monitoring

Comprehensive risk metrics calculated in real-time:

- **Portfolio Volatility**: Rolling and EWMA-based estimates
- **Drawdown Analysis**: Maximum drawdown and recovery periods  
- **Value at Risk (VaR)**: Historical and parametric at 95% and 99% confidence
- **Concentration Risk**: HHI index and effective number of positions
- **Correlation Risk**: Average and maximum pairwise correlations

### Position Management

Advanced position sizing with multiple constraints:

- **Volatility-based Sizing**: Positions scaled by inverse volatility
- **Maximum Position Limits**: Configurable maximum allocation per asset
- **Correlation Adjustments**: Position scaling based on correlation structure
- **Cash Buffer Management**: Maintains liquidity reserves

## 🔬 Validation and Testing

### Comprehensive Test Suite

```bash
# Run all tests
python modules/portfolio_validation.py

# Run specific test categories
python -m unittest modules.portfolio_validation.TestVolatilityEstimator
python -m unittest modules.portfolio_validation.TestRiskParityOptimizer
python -m unittest modules.portfolio_validation.TestLookaheadBiasPrevention
```

### Test Categories

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: End-to-end workflow testing
3. **Lookahead Bias Tests**: Temporal integrity validation
4. **Performance Tests**: Computational efficiency benchmarking
5. **Statistical Validation**: Distribution and correlation testing

### Validation Checklist

- ✅ No future data usage in any calculations
- ✅ Risk parity weights sum to 1 and are non-negative
- ✅ Volatility estimates use only historical data
- ✅ Transaction costs applied correctly
- ✅ Portfolio constraints enforced
- ✅ Risk metrics within reasonable bounds

## 📈 Performance Analysis

### Key Performance Metrics

The system tracks comprehensive performance metrics:

```python
# Access performance statistics
stats = backtesting_engine.get_portfolio_statistics(results)

# Key metrics available:
# - total_return: Cumulative portfolio return
# - annualized_return: Compounded annual return
# - volatility: Annualized standard deviation
# - sharpe_ratio: Risk-adjusted return measure
# - sortino_ratio: Downside risk-adjusted return
# - max_drawdown: Maximum peak-to-trough decline
# - calmar_ratio: Return/Max Drawdown ratio
# - win_rate: Percentage of positive return periods
# - var_95/var_99: Value at Risk at different confidence levels
```

### Performance Attribution

Factor-based performance decomposition:

- **Asset Contribution**: Individual asset return contributions
- **Timing Attribution**: Performance from entry/exit timing
- **Allocation Effects**: Impact of position sizing decisions
- **Volatility Attribution**: Returns from volatility timing

### Benchmarking

Compare against standard benchmarks:

- **Equal Weight Portfolio**: Naive 1/N allocation
- **Market Cap Weighted**: Traditional market weighting
- **Minimum Variance**: Risk-minimizing allocation
- **Buy and Hold**: Passive investment strategy

## 🛠️ Configuration Guide

### Portfolio Manager Configuration

```python
portfolio_manager = PortfolioManager(
    # Volatility estimation
    volatility_method='ewma',           # 'rolling', 'ewma', 'garch', 'realized'
    volatility_lookback=60,             # Days for volatility calculation
    
    # Risk parity method
    risk_parity_method='erc',           # 'inverse_volatility', 'erc', 'risk_budgeting'
    correlation_lookback=126,           # Days for correlation calculation
    
    # Portfolio constraints  
    max_position_size=0.25,             # Maximum allocation per asset (25%)
    min_position_size=0.01,             # Minimum allocation per asset (1%)
    target_volatility=0.15,             # Target annual volatility (15%)
    
    # Rebalancing and costs
    rebalancing_frequency='weekly',      # 'daily', 'weekly', 'monthly'
    transaction_cost_bps=2.0,           # Transaction costs (2 basis points)
    
    # Risk budget (optional)
    risk_budget={'EURUSD': 0.3, 'GBPUSD': 0.2, ...}  # Custom risk allocation
)
```

### Signal Generator Configuration

```python
signal_generator = ConrarianSignalGenerator(
    n_worst_performers=5,               # Number of assets to select
    lookback_days=20,                   # Performance ranking period
    min_history_days=252,               # Minimum data before signal generation
    volatility_lookback=60              # Period for volatility calculation
)
```

### Risk Monitor Configuration

```python
risk_monitor = RiskMonitor(
    confidence_levels=[0.95, 0.99],     # VaR confidence levels
    lookback_window=252                 # Days for risk calculations
)
```

## 📊 Output and Reporting

### Generated Reports

The system automatically generates comprehensive reports:

1. **Summary Report** (`summary_report.md`)
   - Executive summary with key metrics
   - Performance analysis and attribution
   - Risk analysis and recommendations

2. **Visual Reports**
   - `equity_curve_analysis.png`: Portfolio performance over time
   - `drawdown_analysis.png`: Drawdown periods and recovery
   - `portfolio_composition.png`: Asset allocation over time
   - `risk_dashboard.png`: Risk metrics dashboard
   - `performance_attribution.png`: Return attribution analysis

3. **Data Files**
   - `portfolio_weights.parquet`: Daily portfolio weights
   - `portfolio_returns.parquet`: Daily portfolio returns
   - `portfolio_value.parquet`: Portfolio value time series
   - `backtest_statistics.json`: Comprehensive performance metrics

### Custom Reporting

```python
# Generate custom analysis
from modules.performance_analyzer import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
custom_report = analyzer.generate_custom_report(
    portfolio_results, 
    benchmark_data,
    analysis_period='2020-01-01'
)
```

## 🔧 Advanced Usage

### Parameter Optimization

Optimize portfolio parameters for maximum performance:

```python
# Define parameter grid
param_grid = {
    'volatility_method': ['ewma', 'rolling'],
    'risk_parity_method': ['inverse_volatility', 'erc'],
    'volatility_lookback': [30, 60, 90],
    'target_volatility': [0.10, 0.12, 0.15]
}

# Run optimization
optimization_results = portfolio_manager.optimize_portfolio_parameters(
    signal_output, returns, 
    optimization_metric='sharpe_ratio',
    parameter_grid=param_grid
)

best_params = optimization_results['best_parameters']
```

### Custom Risk Budgeting

Implement sector or factor-based risk budgeting:

```python
# Define custom risk budget
risk_budget = {
    'EURUSD': 0.25,    # 25% of risk budget
    'GBPUSD': 0.20,    # 20% of risk budget
    'USDJPY': 0.15,    # 15% of risk budget
    # ... remaining assets
}

portfolio_manager = PortfolioManager(
    risk_parity_method='risk_budgeting',
    risk_budget=risk_budget
)
```

### Integration with External Systems

```python
from modules.portfolio_manager import integrate_with_backtesting_engine

# Integrate with existing backtesting framework
integrated_results = integrate_with_backtesting_engine(
    portfolio_manager=portfolio_manager,
    backtesting_engine=your_backtesting_engine,
    signal_output=signals,
    returns=returns
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Usage**
   ```python
   # For large datasets, use data sampling
   sampled_data = data.iloc[::5]  # Every 5th observation
   ```

2. **Optimization Convergence**
   ```python
   # Increase iteration limits for ERC optimization
   optimizer.max_iterations = 2000
   optimizer.tolerance = 1e-10
   ```

3. **Missing Data**
   ```python
   # Handle missing data in returns
   returns_clean = returns.fillna(method='ffill').fillna(0)
   ```

### Performance Optimization

1. **Use Numba-optimized functions** for large portfolios
2. **Reduce rebalancing frequency** to minimize computation
3. **Use EWMA volatility** instead of GARCH for speed
4. **Sample data** for parameter optimization

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
results = portfolio_manager.run_portfolio_management(signals, returns)
```

## 📚 Mathematical Framework

### Risk Parity Theory

The risk parity approach aims to equalize risk contributions across portfolio assets. For a portfolio with weights **w** and covariance matrix **Σ**, the risk contribution of asset *i* is:

```
RC_i = w_i × (Σw)_i / √(w^T Σ w)
```

### Equal Risk Contribution (ERC)

The ERC optimization problem minimizes the sum of squared deviations from equal risk:

```
minimize: Σ_i (RC_i - σ_p/N)²
subject to: Σ_i w_i = 1, w_i ≥ 0
```

### Volatility Forecasting

#### EWMA Volatility
```
σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}
```

#### GARCH(1,1)
```
σ²_t = ω + αr²_{t-1} + βσ²_{t-1}
```

### Risk Metrics

#### Value at Risk (VaR)
```
VaR_α = -F^{-1}_r(α)
```
where F^{-1}_r is the inverse CDF of portfolio returns.

#### Conditional VaR (CVaR)
```
CVaR_α = E[r | r ≤ VaR_α]
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd advanced_engine

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest modules/

# Run linting
flake8 modules/
black modules/
```

## 📞 Support

For questions and support:

1. **Documentation**: Check this README and inline code documentation
2. **Issues**: Submit GitHub issues for bugs and feature requests
3. **Testing**: Run the validation suite to ensure proper installation

## 📋 Changelog

### Version 1.0.0 (2025-08-06)
- Initial release with complete portfolio management system
- Multiple volatility estimation methods
- Advanced risk parity implementations
- Comprehensive testing and validation suite
- Integration with contrarian signal generation
- Full backtesting and performance analysis capabilities

---

**Note**: This system is designed for institutional use and requires proper risk management procedures. Always validate results and test thoroughly before deploying in production environments.

*Built with ❤️ for quantitative portfolio management*