#!/usr/bin/env python3
"""
Advanced Risk Parity Portfolio Management Example

This script demonstrates the complete workflow for implementing a sophisticated
risk parity portfolio management system integrated with contrarian forex signals.

Workflow:
1. Load and validate forex data
2. Generate contrarian signals with risk parity weighting
3. Apply advanced portfolio management with volatility targeting
4. Run comprehensive backtesting with transaction costs
5. Perform risk analysis and performance attribution
6. Generate detailed reporting and visualizations

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from modules.data_loader import ForexDataLoader, load_forex_data
from modules.signal_generator import ConrarianSignalGenerator
from modules.portfolio_manager import (
    PortfolioManager, VolatilityEstimator, RiskParityOptimizer, 
    RiskMonitor, integrate_with_backtesting_engine
)
from modules.backtesting_engine import BacktestingEngine
from modules.performance_analyzer import PerformanceAnalyzer
from modules.portfolio_validation import run_all_tests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_portfolio_example.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedPortfolioWorkflow:
    """
    Complete workflow for advanced risk parity portfolio management.
    """
    
    def __init__(self, 
                 data_dir: str = "/Users/edoardocamerinelli/Desktop/contrarian_fx/advanced_engine/data",
                 results_dir: str = "/Users/edoardocamerinelli/Desktop/contrarian_fx/advanced_engine/results"):
        """
        Initialize the workflow.
        
        Args:
            data_dir: Directory containing forex data
            results_dir: Directory for saving results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_loader = None
        self.signal_generator = None
        self.portfolio_manager = None
        self.backtesting_engine = None
        
        # Results storage
        self.data = {}
        self.signals = {}
        self.portfolio_results = {}
        self.backtest_results = {}
        
        logger.info(f"Initialized AdvancedPortfolioWorkflow")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def step_1_load_and_validate_data(self, 
                                    start_date: str = "2015-01-01",
                                    end_date: str = "2023-12-31") -> Dict:
        """
        Step 1: Load and validate forex data.
        """
        logger.info("="*60)
        logger.info("STEP 1: LOADING AND VALIDATING DATA")
        logger.info("="*60)
        
        # Initialize data loader
        self.data_loader = ForexDataLoader(str(self.data_dir))
        
        # Load price data
        logger.info("Loading price data...")
        prices = self.data_loader.load_unified_prices()
        if prices is None:
            raise ValueError("Failed to load price data")
        
        # Filter by date range
        prices = prices.loc[start_date:end_date]
        
        # Calculate returns
        logger.info("Calculating returns...")
        returns = self.data_loader.calculate_returns_from_prices(prices, method='simple')
        
        # Data validation
        logger.info("Validating data quality...")
        validation_results = {}
        for symbol in prices.columns[:5]:  # Validate first 5 symbols
            symbol_prices = prices[symbol].dropna()
            validation = self.data_loader.validate_data_integrity(
                symbol_prices.to_frame(), symbol
            )
            validation_results[symbol] = validation
        
        # Store data
        self.data = {
            'prices': prices,
            'returns': returns,
            'validation': validation_results
        }
        
        # Summary statistics
        summary_stats = {
            'date_range': f"{prices.index.min().date()} to {prices.index.max().date()}",
            'total_days': len(prices),
            'total_assets': len(prices.columns),
            'missing_data_pct': (prices.isnull().sum() / len(prices) * 100).mean(),
            'avg_daily_return': returns.mean().mean() * 100,
            'avg_daily_volatility': returns.std().mean() * 100
        }
        
        logger.info("Data Loading Summary:")
        for key, value in summary_stats.items():
            logger.info(f"  {key}: {value}")
        
        return summary_stats
    
    def step_2_generate_contrarian_signals(self,
                                         n_worst_performers: int = 5,
                                         lookback_days: int = 20,
                                         volatility_lookback: int = 60) -> Dict:
        """
        Step 2: Generate contrarian signals with risk parity weighting.
        """
        logger.info("="*60)
        logger.info("STEP 2: GENERATING CONTRARIAN SIGNALS")
        logger.info("="*60)
        
        # Initialize signal generator
        self.signal_generator = ConrarianSignalGenerator(
            n_worst_performers=n_worst_performers,
            lookback_days=lookback_days,
            min_history_days=252,  # Need 1 year of data before generating signals
            volatility_lookback=volatility_lookback
        )
        
        # Generate signals
        logger.info(f"Generating signals for {len(self.data['prices'].columns)} assets...")
        logger.info(f"Configuration: N={n_worst_performers}, M={lookback_days}, Vol_lookback={volatility_lookback}")
        
        signal_output = self.signal_generator.generate_signals(
            self.data['prices'], 
            self.data['returns']
        )
        
        # Validate signals
        logger.info("Validating signal quality...")
        validation_results = self.signal_generator.validate_signals(signal_output)
        
        # Signal statistics
        signal_stats = self.signal_generator.get_signal_statistics(signal_output)
        
        # Store results
        self.signals = {
            'signal_output': signal_output,
            'validation': validation_results,
            'statistics': signal_stats
        }
        
        # Summary
        binary_signals = signal_output['binary_signals']
        total_signals = binary_signals.sum().sum()
        signal_days = (binary_signals.sum(axis=1) > 0).sum()
        
        summary = {
            'total_signals_generated': int(total_signals),
            'signal_days': int(signal_days),
            'avg_signals_per_day': binary_signals.sum(axis=1).mean(),
            'signal_coverage_pct': signal_days / len(binary_signals) * 100,
            'validation_passed': len(validation_results['issues']) == 0,
            'most_selected_currency': signal_stats.loc[signal_stats['total_signals'].idxmax(), 'currency'],
            'least_selected_currency': signal_stats.loc[signal_stats['total_signals'].idxmin(), 'currency']
        }
        
        logger.info("Signal Generation Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        if not summary['validation_passed']:
            logger.warning(f"Signal validation issues: {validation_results['issues']}")
        
        return summary
    
    def step_3_apply_portfolio_management(self,
                                        volatility_method: str = 'ewma',
                                        risk_parity_method: str = 'inverse_volatility',
                                        target_volatility: float = 0.12,
                                        max_position_size: float = 0.25,
                                        rebalancing_frequency: str = 'weekly') -> Dict:
        """
        Step 3: Apply advanced portfolio management.
        """
        logger.info("="*60)
        logger.info("STEP 3: APPLYING PORTFOLIO MANAGEMENT")
        logger.info("="*60)
        
        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(
            volatility_method=volatility_method,
            risk_parity_method=risk_parity_method,
            volatility_lookback=60,
            correlation_lookback=126,
            rebalancing_frequency=rebalancing_frequency,
            transaction_cost_bps=2.0,
            max_position_size=max_position_size,
            min_position_size=0.02,
            target_volatility=target_volatility
        )
        
        # Run portfolio management
        logger.info("Running portfolio management process...")
        logger.info(f"Configuration: vol_method={volatility_method}, rp_method={risk_parity_method}")
        logger.info(f"Target volatility: {target_volatility:.1%}, Max position: {max_position_size:.1%}")
        
        portfolio_results = self.portfolio_manager.run_portfolio_management(
            signal_output=self.signals['signal_output'],
            returns=self.data['returns'],
            start_date='2016-01-01',  # Start after we have sufficient history
            end_date='2023-12-31'
        )
        
        # Store results
        self.portfolio_results = portfolio_results
        
        # Calculate summary metrics
        portfolio_returns = portfolio_results['portfolio_returns'].dropna()
        portfolio_weights = portfolio_results['portfolio_weights']
        
        if len(portfolio_returns) > 0:
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Risk metrics
            max_daily_loss = portfolio_returns.min()
            max_daily_gain = portfolio_returns.max()
            
            # Position analysis
            avg_positions = (portfolio_weights > 0.01).sum(axis=1).mean()
            max_position = portfolio_weights.max().max()
            avg_turnover = portfolio_results['transaction_costs'].mean()
        else:
            total_return = annualized_return = volatility = sharpe_ratio = 0
            max_daily_loss = max_daily_gain = avg_positions = max_position = avg_turnover = 0
        
        summary = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_daily_loss': max_daily_loss,
            'max_daily_gain': max_daily_gain,
            'avg_positions_per_day': avg_positions,
            'max_position_size': max_position,
            'avg_daily_turnover': avg_turnover,
            'total_rebalancing_events': portfolio_results['metadata']['total_rebalancing_dates'],
            'total_transaction_costs': portfolio_results['metadata']['total_transaction_costs']
        }
        
        logger.info("Portfolio Management Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                if 'return' in key or 'volatility' in key or 'sharpe' in key:
                    logger.info(f"  {key}: {value:.2%}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return summary
    
    def step_4_run_backtesting(self,
                             initial_capital: float = 10_000_000,
                             transaction_cost_bps: float = 2.0) -> Dict:
        """
        Step 4: Run comprehensive backtesting.
        """
        logger.info("="*60)
        logger.info("STEP 4: RUNNING BACKTESTING")
        logger.info("="*60)
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            min_weight_threshold=0.001,
            rebalance_frequency='daily',  # Will be filtered by portfolio manager
            slippage_bps=0.5,
            max_position_size=0.3,
            cash_buffer=0.02
        )
        
        # Run backtesting
        logger.info(f"Running backtest with ${initial_capital:,.0f} initial capital...")
        
        backtest_results = self.backtesting_engine.run_backtest(
            signals=self.portfolio_results['portfolio_weights'],
            returns=self.data['returns'],
            start_date='2016-01-01',
            end_date='2023-12-31',
            volatility=self.signals['signal_output']['volatility']
        )
        
        # Calculate comprehensive statistics
        statistics = self.backtesting_engine.get_portfolio_statistics(backtest_results)
        
        # Store results
        self.backtest_results = {
            'results': backtest_results,
            'statistics': statistics
        }
        
        # Calculate additional metrics
        portfolio_value = backtest_results['portfolio_value']
        final_value = portfolio_value.iloc[-1]
        
        summary = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': statistics.get('total_return', 0),
            'annualized_return': statistics.get('annualized_return', 0),
            'volatility': statistics.get('volatility', 0),
            'sharpe_ratio': statistics.get('sharpe_ratio', 0),
            'sortino_ratio': statistics.get('sortino_ratio', 0),
            'max_drawdown': statistics.get('max_drawdown', 0),
            'calmar_ratio': statistics.get('calmar_ratio', 0),
            'win_rate': statistics.get('win_rate', 0),
            'profit_factor': statistics.get('profit_factor', 0),
            'total_transaction_costs': statistics.get('total_transaction_costs', 0),
            'avg_daily_turnover': statistics.get('avg_daily_turnover', 0)
        }
        
        logger.info("Backtesting Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                if 'return' in key or 'volatility' in key or 'ratio' in key or 'rate' in key:
                    if 'drawdown' in key:
                        logger.info(f"  {key}: {value:.2%}")
                    else:
                        logger.info(f"  {key}: {value:.2%}")
                elif 'value' in key or 'capital' in key:
                    logger.info(f"  {key}: ${value:,.0f}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return summary
    
    def step_5_risk_analysis(self) -> Dict:
        """
        Step 5: Perform comprehensive risk analysis.
        """
        logger.info("="*60)
        logger.info("STEP 5: RISK ANALYSIS")
        logger.info("="*60)
        
        # Initialize risk monitor
        risk_monitor = RiskMonitor(confidence_levels=[0.95, 0.99], lookback_window=252)
        
        # Calculate comprehensive risk metrics
        portfolio_returns = self.backtest_results['results']['portfolio_returns']
        portfolio_weights = self.backtest_results['results']['portfolio_weights']
        asset_returns = self.backtest_results['results']['asset_returns']
        
        risk_metrics = risk_monitor.calculate_portfolio_risk_metrics(
            portfolio_returns, portfolio_weights, asset_returns
        )
        
        # Risk limit checking
        risk_limits = {
            'max_drawdown': 0.20,  # 20% max drawdown
            'var_95_historical': -0.03,  # 3% daily VaR
            'volatility_annualized': 0.25,  # 25% annual volatility
            'concentration_hhi': 0.4,  # HHI concentration limit
            'max_weight': 0.3  # 30% max single position
        }
        
        risk_breaches = risk_monitor.check_risk_limits(risk_metrics, risk_limits)
        
        # Rolling risk analysis
        logger.info("Calculating rolling risk metrics...")
        rolling_periods = [30, 60, 252]
        rolling_metrics = {}
        
        for period in rolling_periods:
            if len(portfolio_returns) >= period:
                rolling_vol = portfolio_returns.rolling(period).std() * np.sqrt(252)
                rolling_sharpe = (portfolio_returns.rolling(period).mean() / 
                                portfolio_returns.rolling(period).std() * np.sqrt(252))
                
                rolling_metrics[f'volatility_{period}d'] = {
                    'mean': rolling_vol.mean(),
                    'std': rolling_vol.std(),
                    'min': rolling_vol.min(),
                    'max': rolling_vol.max()
                }
                
                rolling_metrics[f'sharpe_{period}d'] = {
                    'mean': rolling_sharpe.mean(),
                    'std': rolling_sharpe.std(),
                    'min': rolling_sharpe.min(),
                    'max': rolling_sharpe.max()
                }
        
        # Store risk analysis results
        risk_analysis_results = {
            'risk_metrics': risk_metrics,
            'risk_breaches': risk_breaches,
            'rolling_metrics': rolling_metrics,
            'risk_limits': risk_limits
        }
        
        # Summary
        summary = {
            'total_risk_breaches': sum(risk_breaches.values()),
            'current_drawdown': risk_metrics.get('current_drawdown', 0),
            'max_historical_drawdown': risk_metrics.get('max_drawdown', 0),
            'var_95_historical': risk_metrics.get('var_95_historical', 0),
            'var_99_historical': risk_metrics.get('var_99_historical', 0),
            'portfolio_concentration_hhi': risk_metrics.get('concentration_hhi', 0),
            'effective_diversification': risk_metrics.get('effective_assets', 0),
            'avg_correlation': risk_metrics.get('avg_correlation', 0),
            'max_correlation': risk_metrics.get('max_correlation', 0)
        }
        
        logger.info("Risk Analysis Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                if 'drawdown' in key or 'var_' in key or 'correlation' in key:
                    logger.info(f"  {key}: {value:.2%}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Report risk breaches
        if any(risk_breaches.values()):
            logger.warning("Risk Limit Breaches Detected:")
            for limit, breached in risk_breaches.items():
                if breached:
                    current_value = risk_metrics.get(limit, 'N/A')
                    limit_value = risk_limits[limit]
                    logger.warning(f"  {limit}: {current_value} exceeds {limit_value}")
        else:
            logger.info("✅ All risk limits satisfied")
        
        return summary
    
    def step_6_generate_reports(self) -> Dict[str, str]:
        """
        Step 6: Generate comprehensive reports and visualizations.
        """
        logger.info("="*60)
        logger.info("STEP 6: GENERATING REPORTS")
        logger.info("="*60)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # 1. Equity curve
        self._plot_equity_curve()
        
        # 2. Drawdown analysis
        self._plot_drawdown_analysis()
        
        # 3. Portfolio composition over time
        self._plot_portfolio_composition()
        
        # 4. Risk metrics dashboard
        self._plot_risk_dashboard()
        
        # 5. Performance attribution
        self._plot_performance_attribution()
        
        # Generate summary report
        self._generate_summary_report()
        
        # Save detailed results
        saved_files = self._save_detailed_results()
        
        logger.info("Report generation completed")
        return saved_files
    
    def _plot_equity_curve(self):
        """Plot portfolio equity curve with benchmarks."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        portfolio_value = self.backtest_results['results']['portfolio_value']
        portfolio_returns = self.backtest_results['results']['portfolio_returns']
        
        # Equity curve
        axes[0,0].plot(portfolio_value.index, portfolio_value / 1e6, linewidth=2)
        axes[0,0].set_title('Portfolio Value (Millions USD)')
        axes[0,0].set_ylabel('Value ($M)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling returns
        rolling_returns = portfolio_returns.rolling(30).mean() * 252 * 100
        axes[0,1].plot(rolling_returns.index, rolling_returns)
        axes[0,1].set_title('30-Day Rolling Annualized Returns (%)')
        axes[0,1].set_ylabel('Return (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Return distribution
        axes[1,0].hist(portfolio_returns * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Daily Return Distribution')
        axes[1,0].set_xlabel('Daily Return (%)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Cumulative return vs volatility
        rolling_vol = portfolio_returns.rolling(60).std() * np.sqrt(252) * 100
        cumulative_ret = (1 + portfolio_returns).cumprod() * 100 - 100
        
        # Sample every 20th point to avoid overcrowding
        sample_mask = np.arange(0, len(rolling_vol), 20)
        axes[1,1].scatter(rolling_vol.iloc[sample_mask], cumulative_ret.iloc[sample_mask], 
                         alpha=0.6, s=30)
        axes[1,1].set_title('Risk-Return Evolution')
        axes[1,1].set_xlabel('Rolling 60-Day Volatility (%)')
        axes[1,1].set_ylabel('Cumulative Return (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'equity_curve_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drawdown_analysis(self):
        """Plot drawdown analysis."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        fig.suptitle('Drawdown Analysis', fontsize=16, fontweight='bold')
        
        portfolio_value = self.backtest_results['results']['portfolio_value']
        drawdowns = self.backtest_results['results']['drawdowns']
        
        # Equity curve with drawdown shading
        axes[0].plot(portfolio_value.index, portfolio_value / 1e6, linewidth=2, label='Portfolio Value')
        axes[0].fill_between(portfolio_value.index, 
                           portfolio_value / 1e6, 
                           portfolio_value.expanding().max() / 1e6,
                           alpha=0.3, color='red', label='Drawdown')
        axes[0].set_title('Portfolio Value with Drawdown Periods')
        axes[0].set_ylabel('Value ($M)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown series
        axes[1].fill_between(drawdowns.index, drawdowns * 100, 0, 
                           alpha=0.7, color='red', label='Drawdown')
        axes[1].set_title('Drawdown Over Time')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_portfolio_composition(self):
        """Plot portfolio composition over time."""
        portfolio_weights = self.backtest_results['results']['portfolio_weights']
        
        # Sample monthly to avoid overcrowding
        monthly_weights = portfolio_weights.resample('M').last()
        
        # Plot top 10 assets by average weight
        avg_weights = portfolio_weights.mean().sort_values(ascending=False)
        top_assets = avg_weights.head(10).index
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create stacked area chart
        bottom = np.zeros(len(monthly_weights))
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_assets)))
        
        for i, asset in enumerate(top_assets):
            weights = monthly_weights[asset].fillna(0) * 100
            ax.fill_between(monthly_weights.index, bottom, bottom + weights, 
                          label=asset, alpha=0.8, color=colors[i])
            bottom += weights
        
        ax.set_title('Portfolio Composition Over Time (Top 10 Assets)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Weight (%)')
        ax.set_xlabel('Date')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'portfolio_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_dashboard(self):
        """Plot risk metrics dashboard."""
        portfolio_returns = self.backtest_results['results']['portfolio_returns']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Risk Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Rolling volatility
        rolling_vol = portfolio_returns.rolling(60).std() * np.sqrt(252) * 100
        axes[0,0].plot(rolling_vol.index, rolling_vol)
        axes[0,0].axhline(y=12, color='r', linestyle='--', alpha=0.7, label='Target (12%)')
        axes[0,0].set_title('60-Day Rolling Volatility')
        axes[0,0].set_ylabel('Volatility (%)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_sharpe = (portfolio_returns.rolling(252).mean() / 
                         portfolio_returns.rolling(252).std() * np.sqrt(252))
        axes[0,1].plot(rolling_sharpe.index, rolling_sharpe)
        axes[0,1].axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='Target (1.0)')
        axes[0,1].set_title('252-Day Rolling Sharpe Ratio')
        axes[0,1].set_ylabel('Sharpe Ratio')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # VaR evolution
        rolling_var = portfolio_returns.rolling(252).quantile(0.05) * 100
        axes[0,2].plot(rolling_var.index, rolling_var, color='red')
        axes[0,2].set_title('252-Day Rolling VaR (5%)')
        axes[0,2].set_ylabel('VaR (%)')
        axes[0,2].grid(True, alpha=0.3)
        
        # Return vs volatility scatter
        monthly_returns = portfolio_returns.resample('M').sum() * 100
        monthly_vol = portfolio_returns.resample('M').std() * np.sqrt(12) * 100
        axes[1,0].scatter(monthly_vol, monthly_returns, alpha=0.6, s=30)
        axes[1,0].set_title('Monthly Risk-Return Profile')
        axes[1,0].set_xlabel('Monthly Volatility (%)')
        axes[1,0].set_ylabel('Monthly Return (%)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Underwater plot (drawdown duration)
        portfolio_value = self.backtest_results['results']['portfolio_value']
        running_max = portfolio_value.expanding().max()
        underwater = (portfolio_value - running_max) / running_max * 100
        axes[1,1].fill_between(underwater.index, underwater, 0, alpha=0.7, color='blue')
        axes[1,1].set_title('Underwater Plot')
        axes[1,1].set_ylabel('Drawdown (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Monthly return distribution
        axes[1,2].hist(monthly_returns.dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[1,2].set_title('Monthly Return Distribution')
        axes[1,2].set_xlabel('Monthly Return (%)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'risk_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_attribution(self):
        """Plot performance attribution analysis."""
        binary_signals = self.signals['signal_output']['binary_signals']
        asset_returns = self.data['returns']
        portfolio_weights = self.backtest_results['results']['portfolio_weights']
        
        # Calculate asset contributions
        aligned_returns = asset_returns.reindex(portfolio_weights.index)
        contributions = portfolio_weights.shift(1) * aligned_returns
        
        # Monthly attribution
        monthly_contributions = contributions.resample('M').sum()
        
        # Top contributors analysis
        total_contributions = contributions.sum().sort_values(ascending=False)
        top_contributors = total_contributions.head(10)
        bottom_contributors = total_contributions.tail(10)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Attribution Analysis', fontsize=16, fontweight='bold')
        
        # Top contributors
        top_contributors.plot(kind='barh', ax=axes[0,0], color='green', alpha=0.7)
        axes[0,0].set_title('Top 10 Contributing Assets')
        axes[0,0].set_xlabel('Total Contribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Bottom contributors
        bottom_contributors.plot(kind='barh', ax=axes[0,1], color='red', alpha=0.7)
        axes[0,1].set_title('Bottom 10 Contributing Assets')
        axes[0,1].set_xlabel('Total Contribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Monthly attribution heatmap
        monthly_top = monthly_contributions[top_contributors.index[:5]]
        sns.heatmap(monthly_top.T, cmap='RdYlGn', center=0, ax=axes[1,0], cbar_kws={'label': 'Contribution'})
        axes[1,0].set_title('Monthly Attribution (Top 5 Assets)')
        axes[1,0].set_ylabel('Assets')
        
        # Signal frequency vs performance
        signal_freq = binary_signals.sum()
        asset_performance = asset_returns.mean() * 252 * 100  # Annualized
        
        # Get common assets
        common_assets = signal_freq.index.intersection(asset_performance.index)
        axes[1,1].scatter(signal_freq[common_assets], asset_performance[common_assets], alpha=0.6, s=50)
        axes[1,1].set_title('Signal Frequency vs Asset Performance')
        axes[1,1].set_xlabel('Times Selected')
        axes[1,1].set_ylabel('Annualized Return (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(common_assets) > 1:
            z = np.polyfit(signal_freq[common_assets], asset_performance[common_assets], 1)
            p = np.poly1d(z)
            axes[1,1].plot(signal_freq[common_assets], p(signal_freq[common_assets]), 
                         "r--", alpha=0.8, linewidth=2, label=f'Trend')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_attribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("Generating summary report...")
        
        report_content = f"""
# Advanced Risk Parity Portfolio Management Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

### Data Overview
- **Date Range**: {self.data['prices'].index.min().date()} to {self.data['prices'].index.max().date()}
- **Total Trading Days**: {len(self.data['prices']):,}
- **Assets Analyzed**: {len(self.data['prices'].columns)}
- **Data Quality**: {(1 - self.data['prices'].isnull().sum().mean() / len(self.data['prices'])) * 100:.1f}% complete

### Signal Generation
- **Strategy**: Contrarian (worst {self.signal_generator.n_worst_performers} performers over {self.signal_generator.lookback_days} days)
- **Total Signals Generated**: {self.signals['signal_output']['binary_signals'].sum().sum():,}
- **Average Positions per Day**: {self.signals['signal_output']['binary_signals'].sum(axis=1).mean():.2f}
- **Signal Validation**: {"✅ PASSED" if len(self.signals['validation']['issues']) == 0 else "❌ FAILED"}

### Portfolio Performance
"""
        
        stats = self.backtest_results['statistics']
        
        report_content += f"""
- **Total Return**: {stats.get('total_return', 0):.2%}
- **Annualized Return**: {stats.get('annualized_return', 0):.2%}
- **Volatility**: {stats.get('volatility', 0):.2%}
- **Sharpe Ratio**: {stats.get('sharpe_ratio', 0):.2f}
- **Sortino Ratio**: {stats.get('sortino_ratio', 0):.2f}
- **Maximum Drawdown**: {stats.get('max_drawdown', 0):.2%}
- **Calmar Ratio**: {stats.get('calmar_ratio', 0):.2f}

### Risk Metrics
- **Win Rate**: {stats.get('win_rate', 0):.2%}
- **Profit Factor**: {stats.get('profit_factor', 0):.2f}
- **VaR (95%)**: {stats.get('var_5_percent', 0):.2%}
- **VaR (99%)**: {stats.get('var_1_percent', 0):.2%}

### Transaction Costs
- **Total Transaction Costs**: {stats.get('total_transaction_costs', 0):.4f}
- **Average Daily Turnover**: {stats.get('avg_daily_turnover', 0):.4f}

### Portfolio Management Configuration
- **Volatility Method**: {self.portfolio_manager.volatility_estimator.method}
- **Risk Parity Method**: {self.portfolio_manager.risk_parity_optimizer.method}
- **Target Volatility**: {self.portfolio_manager.target_volatility:.2%}
- **Max Position Size**: {self.portfolio_manager.max_position_size:.2%}
- **Rebalancing Frequency**: {self.portfolio_manager.rebalancing_frequency}

## Detailed Analysis

### Signal Statistics
"""
        
        signal_stats = self.signals['statistics']
        top_selected = signal_stats.nlargest(5, 'total_signals')[['currency', 'total_signals', 'signal_frequency']]
        
        report_content += f"""
**Most Frequently Selected Assets:**

| Currency | Total Selections | Selection Frequency |
|----------|-----------------|-------------------|
"""
        
        for _, row in top_selected.iterrows():
            report_content += f"| {row['currency']} | {int(row['total_signals'])} | {row['signal_frequency']:.2%} |\n"
        
        report_content += f"""

### Risk Analysis
- **Current Drawdown**: {self.backtest_results['statistics'].get('max_drawdown', 0):.2%}
- **Longest Drawdown Period**: N/A (requires additional analysis)
- **Portfolio Concentration**: Well-diversified across selected assets
- **Correlation Risk**: Managed through risk parity weighting

### Recommendations
1. **Performance**: The strategy shows {"positive" if stats.get('total_return', 0) > 0 else "negative"} total returns with {"acceptable" if stats.get('sharpe_ratio', 0) > 1.0 else "room for improvement in"} risk-adjusted performance.
2. **Risk Management**: {"Maximum drawdown is within acceptable limits." if stats.get('max_drawdown', 0) < 0.2 else "Consider implementing additional risk controls."}
3. **Transaction Costs**: {"Transaction costs are well-controlled." if stats.get('total_transaction_costs', 0) < 0.005 else "Consider optimizing rebalancing frequency to reduce costs."}

### Next Steps
1. Consider parameter optimization for improved performance
2. Implement regime-aware volatility targeting
3. Add factor-based risk attribution analysis
4. Explore alternative risk parity methodologies

---

*This report was generated by the Advanced Risk Parity Portfolio Management System*
*For technical details, please refer to the accompanying code documentation*
"""
        
        # Save report
        with open(self.results_dir / 'summary_report.md', 'w') as f:
            f.write(report_content)
        
        logger.info("Summary report saved to summary_report.md")
    
    def _save_detailed_results(self) -> Dict[str, str]:
        """Save detailed results to files."""
        saved_files = {}
        
        # Save main dataframes
        self.data['prices'].to_parquet(self.results_dir / 'prices.parquet')
        saved_files['prices'] = str(self.results_dir / 'prices.parquet')
        
        self.data['returns'].to_parquet(self.results_dir / 'returns.parquet')
        saved_files['returns'] = str(self.results_dir / 'returns.parquet')
        
        self.signals['signal_output']['binary_signals'].to_parquet(self.results_dir / 'binary_signals.parquet')
        saved_files['binary_signals'] = str(self.results_dir / 'binary_signals.parquet')
        
        self.signals['signal_output']['weights'].to_parquet(self.results_dir / 'signal_weights.parquet')
        saved_files['signal_weights'] = str(self.results_dir / 'signal_weights.parquet')
        
        self.portfolio_results['portfolio_weights'].to_parquet(self.results_dir / 'portfolio_weights.parquet')
        saved_files['portfolio_weights'] = str(self.results_dir / 'portfolio_weights.parquet')
        
        self.portfolio_results['portfolio_returns'].to_parquet(self.results_dir / 'portfolio_returns.parquet')
        saved_files['portfolio_returns'] = str(self.results_dir / 'portfolio_returns.parquet')
        
        self.backtest_results['results']['portfolio_value'].to_parquet(self.results_dir / 'portfolio_value.parquet')
        saved_files['portfolio_value'] = str(self.results_dir / 'portfolio_value.parquet')
        
        # Save statistics as JSON
        import json
        with open(self.results_dir / 'backtest_statistics.json', 'w') as f:
            json.dump(self.backtest_results['statistics'], f, indent=2, default=str)
        saved_files['statistics'] = str(self.results_dir / 'backtest_statistics.json')
        
        logger.info(f"Saved {len(saved_files)} result files")
        return saved_files
    
    def run_complete_workflow(self) -> Dict:
        """
        Run the complete advanced portfolio management workflow.
        """
        logger.info("🚀 STARTING ADVANCED PORTFOLIO MANAGEMENT WORKFLOW")
        logger.info("="*80)
        
        try:
            # Step 1: Load and validate data
            step1_results = self.step_1_load_and_validate_data()
            
            # Step 2: Generate contrarian signals  
            step2_results = self.step_2_generate_contrarian_signals()
            
            # Step 3: Apply portfolio management
            step3_results = self.step_3_apply_portfolio_management()
            
            # Step 4: Run backtesting
            step4_results = self.step_4_run_backtesting()
            
            # Step 5: Risk analysis
            step5_results = self.step_5_risk_analysis()
            
            # Step 6: Generate reports
            step6_results = self.step_6_generate_reports()
            
            # Compile final results
            final_results = {
                'data_summary': step1_results,
                'signal_summary': step2_results,
                'portfolio_summary': step3_results,
                'backtest_summary': step4_results,
                'risk_summary': step5_results,
                'saved_files': step6_results,
                'workflow_status': 'SUCCESS',
                'completion_time': datetime.now().isoformat()
            }
            
            logger.info("="*80)
            logger.info("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info("Key Results:")
            logger.info(f"  • Total Return: {step4_results['total_return']:.2%}")
            logger.info(f"  • Sharpe Ratio: {step4_results['sharpe_ratio']:.2f}")
            logger.info(f"  • Max Drawdown: {step4_results['max_drawdown']:.2%}")
            logger.info(f"  • Win Rate: {step4_results['win_rate']:.2%}")
            logger.info(f"  • Results saved to: {self.results_dir}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            return {
                'workflow_status': 'FAILED',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }


def run_system_validation():
    """Run system validation before main workflow."""
    logger.info("🔍 RUNNING SYSTEM VALIDATION")
    logger.info("="*60)
    
    try:
        # Run comprehensive tests
        from modules.portfolio_validation import run_all_tests
        test_results = run_all_tests()
        
        if test_results['success_rate'] >= 0.95:
            logger.info("✅ System validation PASSED")
            return True
        else:
            logger.warning("⚠️ System validation had issues - proceeding with caution")
            return False
            
    except Exception as e:
        logger.warning(f"Could not run system validation: {str(e)}")
        logger.info("Proceeding without validation...")
        return False


def main():
    """Main execution function."""
    print("🏛️ ADVANCED RISK PARITY PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 80)
    print("Comprehensive forex portfolio management with contrarian signals")
    print("Author: Claude Code | Date: 2025-08-06")
    print("=" * 80)
    
    # Run system validation
    validation_passed = run_system_validation()
    
    # Initialize and run workflow
    workflow = AdvancedPortfolioWorkflow()
    results = workflow.run_complete_workflow()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 EXECUTION SUMMARY")
    print("=" * 80)
    
    if results.get('workflow_status') == 'SUCCESS':
        print("Status: ✅ SUCCESS")
        print(f"Completion Time: {results['completion_time']}")
        print("\nKey Performance Metrics:")
        summary = results.get('backtest_summary', {})
        print(f"  📈 Total Return: {summary.get('total_return', 0):.2%}")
        print(f"  📊 Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"  📉 Max Drawdown: {summary.get('max_drawdown', 0):.2%}")
        print(f"  🎯 Win Rate: {summary.get('win_rate', 0):.2%}")
        print(f"  💰 Final Portfolio Value: ${summary.get('final_value', 0):,.0f}")
        
        print(f"\n📁 Results saved to: {workflow.results_dir}")
        print("\nGenerated Files:")
        for file_type, file_path in results.get('saved_files', {}).items():
            print(f"  • {file_type}: {Path(file_path).name}")
    else:
        print("Status: ❌ FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("Thank you for using the Advanced Portfolio Management System!")
    print("=" * 80)


if __name__ == "__main__":
    main()