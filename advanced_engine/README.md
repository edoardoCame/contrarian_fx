# High-Performance Contrarian Forex Backtesting Framework

A comprehensive, production-ready backtesting framework for contrarian forex trading strategies with advanced optimization capabilities and strict lookahead bias prevention.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Components](#framework-components)
- [Usage Examples](#usage-examples)
- [Performance Considerations](#performance-considerations)
- [Validation & Testing](#validation--testing)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Overview

This framework implements a contrarian forex trading strategy with robust backtesting capabilities. The system identifies the worst-performing currencies over a lookback period and constructs a risk-parity weighted portfolio of these underperformers, betting on mean reversion.

**Key Philosophy**: 
- Zero lookahead bias through rigorous temporal data alignment
- Production-ready performance with numba acceleration
- Comprehensive risk management and performance analytics
- Systematic parameter optimization with overfitting prevention

## Key Features

### Core Functionality
- **Vectorized Backtesting**: High-performance vectorized calculations with numba JIT compilation
- **Zero Lookahead Bias**: Strict temporal data alignment ensuring no future information leakage
- **Risk Parity Weighting**: Automatic portfolio construction based on historical volatility
- **Transaction Cost Modeling**: Realistic bid-ask spreads, slippage, and market impact
- **Multi-Asset Support**: Handle up to 20+ forex pairs simultaneously

### Advanced Analytics
- **Comprehensive Performance Metrics**: Sharpe, Sortino, Calmar ratios, VaR, CVaR, tail risk measures
- **Drawdown Analysis**: Detailed drawdown periods, recovery times, and underwater curves
- **Rolling Performance**: Time-varying performance analysis with multiple window sizes
- **Monte Carlo Simulation**: Robustness testing with scenario analysis
- **Performance Attribution**: Asset-level contribution analysis

### Optimization Framework
- **Grid Search**: Exhaustive parameter space exploration
- **Bayesian Optimization**: Efficient parameter search using Gaussian processes
- **Walk-Forward Analysis**: Parameter stability testing across time periods
- **Multi-Objective Optimization**: Balance return, risk, and drawdown objectives
- **Cross-Validation**: Time-series aware validation with proper train/test splits
- **Overfitting Detection**: Statistical tests for parameter overfitting

### Data Management
- **Efficient Storage**: Parquet-based time series storage with compression
- **Result Management**: SQLite-based metadata database with search capabilities
- **Data Validation**: Comprehensive data integrity and quality checks
- **Export Capabilities**: Excel, CSV, JSON export formats
- **Version Control**: Result versioning and comparison utilities

## Architecture

```
advanced_engine/
├── modules/                     # Core framework components
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── signal_generator.py     # Contrarian signal generation
│   ├── backtesting_engine.py   # High-performance backtesting
│   ├── performance_analyzer.py # Performance analytics
│   ├── parameter_optimizer.py  # Parameter optimization
│   └── results_manager.py      # Results storage & management
├── data/                       # Forex data files (parquet format)
├── results/                    # Backtest results storage
├── example_backtesting.py      # Usage examples
├── validation_tests.py         # Comprehensive test suite
└── README.md                   # This file
```

## Installation

### Requirements
```bash
# Core dependencies
pandas >= 1.5.0
numpy >= 1.21.0
numba >= 0.56.0
scipy >= 1.9.0
scikit-learn >= 1.1.0

# Optional (for advanced features)
scikit-optimize  # Bayesian optimization
matplotlib       # Visualization
seaborn         # Statistical plotting
openpyxl        # Excel export
```

### Setup
```bash
# Clone or download the framework
cd advanced_engine

# Install dependencies
pip install -r requirements.txt

# Validate installation
python validation_tests.py
```

## Quick Start

### Basic Backtesting
```python
from modules.data_loader import ForexDataLoader
from modules.signal_generator import ConrarianSignalGenerator
from modules.backtesting_engine import BacktestingEngine
from modules.performance_analyzer import PerformanceAnalyzer

# Load data
data_loader = ForexDataLoader("data")
returns_data = data_loader.get_data_for_period("2015-01-01", "2023-12-31", data_type='returns')
prices_data = data_loader.get_data_for_period("2015-01-01", "2023-12-31", data_type='prices')

# Generate signals
signal_generator = ConrarianSignalGenerator(n_worst_performers=5, lookback_days=20)
signals = signal_generator.generate_signals(prices_data, returns_data)

# Run backtest
backtester = BacktestingEngine(initial_capital=1000000, transaction_cost_bps=2.0)
results = backtester.run_backtest(signals['weights'], returns_data)

# Analyze performance
analyzer = PerformanceAnalyzer()
performance_report = analyzer.generate_performance_report(results)

print(f"Sharpe Ratio: {performance_report['summary']['key_metrics']['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {performance_report['summary']['key_metrics']['max_drawdown']*100:.2f}%")
```

### Parameter Optimization
```python
from modules.parameter_optimizer import ParameterOptimizer

# Define parameter grid
parameter_grid = {
    'n_worst_performers': [2, 3, 5, 7, 10],
    'lookback_days': [5, 10, 15, 20, 30]
}

# Run optimization
optimizer = ParameterOptimizer(optimization_metric='sharpe_ratio')
opt_results = optimizer.grid_search_optimization(
    data_loader=data_loader,
    signal_generator_class=ConrarianSignalGenerator,
    backtesting_engine_class=BacktestingEngine,
    parameter_grid=parameter_grid,
    start_date="2010-01-01",
    end_date="2020-12-31"
)

best_params = opt_results['best_parameters']
print(f"Optimal parameters: {best_params}")
```

## Framework Components

### 1. Data Loader (`data_loader.py`)
- **Purpose**: Efficient forex data loading and preprocessing
- **Key Features**: Data validation, alignment, caching, quality checks
- **Supported Formats**: Parquet files with OHLCV data
- **Data Integrity**: Automatic duplicate detection, gap analysis, outlier identification

### 2. Signal Generator (`signal_generator.py`)
- **Purpose**: Contrarian signal generation with lookahead bias prevention
- **Strategy Logic**: Identify N worst performers over M-day lookback period
- **Weighting Scheme**: Risk parity based on historical volatility
- **Validation**: Comprehensive lookahead bias testing and signal quality checks

### 3. Backtesting Engine (`backtesting_engine.py`)
- **Purpose**: High-performance vectorized backtesting
- **Performance**: Numba JIT compilation for critical calculations
- **Features**: Transaction costs, slippage, rebalancing, position limits
- **Output**: Detailed portfolio performance with full attribution

### 4. Performance Analyzer (`performance_analyzer.py`)
- **Purpose**: Comprehensive performance analytics and risk assessment
- **Metrics**: 20+ performance metrics including tail risk measures
- **Analysis**: Rolling performance, drawdown analysis, Monte Carlo simulation
- **Reporting**: Automated report generation with visualizations

### 5. Parameter Optimizer (`parameter_optimizer.py`)
- **Purpose**: Systematic parameter optimization with overfitting prevention
- **Methods**: Grid search, Bayesian optimization, walk-forward analysis
- **Validation**: Cross-validation, statistical significance testing
- **Features**: Multi-objective optimization, parameter stability analysis

### 6. Results Manager (`results_manager.py`)
- **Purpose**: Results storage, retrieval, and management
- **Storage**: Efficient parquet + JSON + SQLite architecture
- **Features**: Result comparison, aggregation, export capabilities
- **Scalability**: Automatic cleanup, compression, storage monitoring

## Usage Examples

### Example 1: Walk-Forward Analysis
```python
# Test parameter stability over time
wf_results = optimizer.walk_forward_optimization(
    data_loader=data_loader,
    signal_generator_class=ConrarianSignalGenerator,
    backtesting_engine_class=BacktestingEngine,
    parameter_grid=parameter_grid,
    start_date="2015-01-01",
    end_date="2023-12-31"
)

# Analyze parameter stability
stability = wf_results['parameter_stability']
for param, metrics in stability.items():
    print(f"{param}: CV={metrics['cv']:.3f}, Range=({metrics['min']}, {metrics['max']})")
```

### Example 2: Multi-Objective Optimization
```python
# Balance return, risk, and drawdown
mo_results = optimizer.multi_objective_optimization(
    data_loader=data_loader,
    signal_generator_class=ConrarianSignalGenerator,
    backtesting_engine_class=BacktestingEngine,
    parameter_grid=parameter_grid,
    objectives=['sharpe_ratio', 'max_drawdown', 'calmar_ratio'],
    objective_weights=[0.5, 0.3, 0.2],
    start_date="2012-01-01",
    end_date="2022-12-31"
)
```

### Example 3: Results Management
```python
from modules.results_manager import ResultsManager

# Save results
results_manager = ResultsManager("results")
result_id = results_manager.save_backtest_results(
    backtest_results=results,
    strategy_name="ConrarianFX_Optimized",
    parameters={'n_worst_performers': 5, 'lookback_days': 20},
    description="Optimized contrarian strategy",
    tags=['contrarian', 'forex', 'optimized']
)

# Compare multiple results
available_results = results_manager.list_backtest_results()
comparison = results_manager.compare_backtest_results(
    result_ids=available_results['result_id'].head(3).tolist(),
    metrics=['sharpe_ratio', 'max_drawdown', 'calmar_ratio']
)
```

## Performance Considerations

### Computational Efficiency
- **Numba JIT**: Critical loops compiled for near-native performance
- **Vectorization**: Pandas/NumPy operations for bulk calculations
- **Memory Management**: Efficient data structures and caching
- **Parallel Processing**: Multi-core parameter optimization

### Scalability Benchmarks
- **Single Backtest**: ~1-2 seconds for 10 years, 20 assets
- **Parameter Grid (25 combinations)**: ~30-60 seconds
- **Walk-Forward Analysis**: ~5-10 minutes for 5 years
- **Memory Usage**: ~100-500MB for typical datasets

### Optimization Tips
```python
# Use appropriate number of parallel jobs
optimizer = ParameterOptimizer(n_jobs=4)  # Adjust based on CPU cores

# Cache data for repeated use
data_loader = ForexDataLoader("data")
data_loader.load_unified_returns()  # Cached automatically

# Limit parameter grid size for initial exploration
parameter_grid = {
    'n_worst_performers': [3, 5, 7],    # Start with fewer values
    'lookback_days': [10, 20, 30]
}
```

## Validation & Testing

The framework includes comprehensive validation tests:

```bash
# Run full validation suite
python validation_tests.py

# Run specific test categories
python -m unittest validation_tests.TestSignalGeneration
python -m unittest validation_tests.TestBacktestingEngine
```

### Test Coverage
- **Data Integrity**: Loading, validation, consistency checks
- **Signal Generation**: Lookahead bias prevention, contrarian logic
- **Backtesting**: Transaction costs, portfolio construction, statistics
- **Performance Analysis**: Metrics calculation, risk measures
- **Optimization**: Parameter search, overfitting detection
- **Results Management**: Storage, retrieval, data integrity
- **End-to-End**: Complete workflow validation

## API Reference

### Signal Generator API
```python
ConrarianSignalGenerator(
    n_worst_performers: int = 5,        # Number of worst performers to select
    lookback_days: int = 20,            # Lookback period for performance ranking
    min_history_days: int = 252,        # Minimum history before generating signals
    volatility_lookback: int = 60       # Volatility calculation window
)

# Generate signals with validation
signals = generator.generate_signals(prices_df, returns_df)
validation = generator.validate_signals(signals)
```

### Backtesting Engine API
```python
BacktestingEngine(
    initial_capital: float = 1000000,    # Starting portfolio value
    transaction_cost_bps: float = 2.0,   # Transaction costs (basis points)
    slippage_bps: float = 0.5,          # Market impact/slippage
    max_position_size: float = 0.3,      # Maximum position size per asset
    rebalance_frequency: str = 'daily'   # Rebalancing frequency
)

# Run backtest with full results
results = engine.run_backtest(signals, returns, start_date, end_date)
statistics = engine.get_portfolio_statistics(results)
```

### Performance Analyzer API
```python
PerformanceAnalyzer(
    risk_free_rate: float = 0.02,        # Annual risk-free rate
    confidence_levels: List[float] = [0.01, 0.05, 0.10]  # VaR confidence levels
)

# Generate comprehensive analysis
report = analyzer.generate_performance_report(backtest_results)
return_analysis = analyzer.analyze_returns(portfolio_returns)
drawdown_analysis = analyzer.analyze_drawdowns(portfolio_value)
```

## Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 with 88-character line limit
2. **Documentation**: Comprehensive docstrings for all public methods
3. **Testing**: Unit tests for all new functionality
4. **Performance**: Benchmark critical paths with profiling
5. **Validation**: Run full validation suite before commits

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-optimization-method`
2. Implement with tests: Add to appropriate module with comprehensive tests
3. Update documentation: Include usage examples and API documentation
4. Validate performance: Ensure no significant performance regression
5. Submit pull request: Include benchmark results and test coverage

### Performance Optimization
- Profile with `cProfile` and `line_profiler`
- Use `@numba.jit` for computational bottlenecks
- Vectorize operations with pandas/numpy
- Cache expensive calculations
- Monitor memory usage with `memory_profiler`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Numba Team**: For JIT compilation capabilities
- **Pandas/NumPy**: For efficient data manipulation
- **Scikit-Learn**: For optimization utilities
- **QuantLib**: For financial mathematics inspiration

---

**Framework Version**: 1.0.0  
**Last Updated**: 2025-08-06  
**Python Compatibility**: 3.8+  
**Status**: Production Ready  

For questions, issues, or contributions, please refer to the project documentation or contact the development team.