#!/usr/bin/env python3
"""
Results Management and Storage System for Contrarian Forex Strategy

This module provides comprehensive results storage, loading, and management
capabilities for the backtesting framework. It handles result serialization,
database storage, performance comparison, and portfolio analysis.

Key Features:
- Efficient parquet-based storage for time series data
- JSON storage for metadata and configurations
- Result comparison and benchmarking utilities
- Portfolio performance tracking and analysis
- Data validation and integrity checks
- Compression and optimization for large datasets
- Result aggregation and summarization
- Export capabilities for reporting

Author: Claude Code
Date: 2025-08-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json
import pickle
import sqlite3
import hashlib
import gzip
import shutil
from dataclasses import dataclass, asdict
from collections import defaultdict

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class BacktestMetadata:
    """Structured metadata for backtest results."""
    strategy_name: str
    parameters: Dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float
    assets: List[str]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    creation_timestamp: str
    data_hash: str
    version: str = "1.0"


@dataclass
class OptimizationMetadata:
    """Structured metadata for optimization results."""
    optimization_method: str
    parameter_space: Dict[str, Any]
    best_parameters: Dict[str, Any]
    optimization_metric: str
    best_metric_value: float
    n_trials: int
    optimization_duration: float
    creation_timestamp: str
    version: str = "1.0"


class ResultsManager:
    """
    Comprehensive results management system for backtesting and optimization.
    """
    
    def __init__(self,
                 base_directory: str = "results",
                 compression_level: int = 3,
                 auto_cleanup: bool = True,
                 max_storage_gb: float = 10.0):
        """
        Initialize the results manager.
        
        Args:
            base_directory: Base directory for storing results
            compression_level: Compression level for parquet files (0-9)
            auto_cleanup: Whether to automatically clean old results
            max_storage_gb: Maximum storage space in GB
        """
        self.base_directory = Path(base_directory)
        self.compression_level = compression_level
        self.auto_cleanup = auto_cleanup
        self.max_storage_gb = max_storage_gb
        
        # Create directory structure
        self._initialize_directory_structure()
        
        # Initialize database
        self._initialize_metadata_database()
        
        logger.info(f"Initialized ResultsManager with base directory: {self.base_directory}")
    
    def save_backtest_results(self,
                            backtest_results: Dict[str, Any],
                            strategy_name: str,
                            parameters: Dict[str, Any],
                            description: str = "",
                            tags: List[str] = None) -> str:
        """
        Save complete backtest results with metadata.
        
        Args:
            backtest_results: Results from backtesting engine
            strategy_name: Name of the strategy
            parameters: Strategy parameters used
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Unique result ID
        """
        logger.info(f"Saving backtest results for strategy: {strategy_name}")
        
        # Generate unique ID
        result_id = self._generate_result_id(strategy_name, parameters)
        
        # Create result directory
        result_dir = self.base_directory / "backtests" / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main time series data
            self._save_time_series_data(backtest_results, result_dir)
            
            # Create and save metadata
            metadata = BacktestMetadata(
                strategy_name=strategy_name,
                parameters=parameters,
                start_date=str(backtest_results['metadata']['start_date']),
                end_date=str(backtest_results['metadata']['end_date']),
                initial_capital=backtest_results['metadata']['initial_capital'],
                assets=backtest_results['metadata']['assets'],
                total_return=self._calculate_total_return(backtest_results),
                sharpe_ratio=self._calculate_sharpe_ratio(backtest_results),
                max_drawdown=backtest_results['metadata'].get('max_drawdown', 0),
                creation_timestamp=datetime.now().isoformat(),
                data_hash=self._calculate_data_hash(backtest_results),
                version="1.0"
            )
            
            # Save metadata
            self._save_metadata(metadata, result_dir, description, tags)
            
            # Update database
            self._update_metadata_database(result_id, metadata, description, tags)
            
            # Cleanup if needed
            if self.auto_cleanup:
                self._cleanup_old_results()
            
            logger.info(f"Backtest results saved with ID: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            # Cleanup partial save
            if result_dir.exists():
                shutil.rmtree(result_dir)
            raise
    
    def save_optimization_results(self,
                                optimization_results: Dict[str, Any],
                                optimization_method: str,
                                parameter_space: Dict[str, Any],
                                description: str = "",
                                tags: List[str] = None) -> str:
        """
        Save optimization results with metadata.
        
        Args:
            optimization_results: Results from parameter optimizer
            optimization_method: Optimization method used
            parameter_space: Parameter space searched
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Unique result ID
        """
        logger.info(f"Saving optimization results for method: {optimization_method}")
        
        # Generate unique ID
        result_id = self._generate_optimization_id(optimization_method, parameter_space)
        
        # Create result directory
        result_dir = self.base_directory / "optimizations" / result_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save optimization data
            self._save_optimization_data(optimization_results, result_dir)
            
            # Create metadata
            metadata = OptimizationMetadata(
                optimization_method=optimization_method,
                parameter_space=parameter_space,
                best_parameters=optimization_results.get('best_parameters', {}),
                optimization_metric=optimization_results.get('optimization_metric', 'unknown'),
                best_metric_value=self._extract_best_metric_value(optimization_results),
                n_trials=self._extract_n_trials(optimization_results),
                optimization_duration=optimization_results.get('optimization_duration', 0),
                creation_timestamp=datetime.now().isoformat(),
                version="1.0"
            )
            
            # Save metadata
            optimization_file = result_dir / "optimization_metadata.json"
            with open(optimization_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Save additional info
            info_file = result_dir / "optimization_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'description': description,
                    'tags': tags or [],
                    'creation_date': datetime.now().isoformat()
                }, f, indent=2)
            
            # Update database
            self._update_optimization_database(result_id, metadata, description, tags)
            
            logger.info(f"Optimization results saved with ID: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Error saving optimization results: {str(e)}")
            if result_dir.exists():
                shutil.rmtree(result_dir)
            raise
    
    def load_backtest_results(self, result_id: str) -> Dict[str, Any]:
        """
        Load complete backtest results.
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            Dictionary with complete backtest results
        """
        logger.info(f"Loading backtest results: {result_id}")
        
        result_dir = self.base_directory / "backtests" / result_id
        
        if not result_dir.exists():
            raise FileNotFoundError(f"Backtest results not found: {result_id}")
        
        try:
            # Load metadata
            metadata_file = result_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load time series data
            results = self._load_time_series_data(result_dir)
            
            # Add metadata
            results['metadata'] = metadata
            results['result_id'] = result_id
            
            logger.info(f"Successfully loaded backtest results: {result_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading backtest results {result_id}: {str(e)}")
            raise
    
    def load_optimization_results(self, result_id: str) -> Dict[str, Any]:
        """
        Load optimization results.
        
        Args:
            result_id: Unique result identifier
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Loading optimization results: {result_id}")
        
        result_dir = self.base_directory / "optimizations" / result_id
        
        if not result_dir.exists():
            raise FileNotFoundError(f"Optimization results not found: {result_id}")
        
        try:
            # Load metadata
            metadata_file = result_dir / "optimization_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load optimization data
            results = self._load_optimization_data(result_dir)
            
            # Add metadata
            results['metadata'] = metadata
            results['result_id'] = result_id
            
            logger.info(f"Successfully loaded optimization results: {result_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading optimization results {result_id}: {str(e)}")
            raise
    
    def list_backtest_results(self,
                            strategy_name: Optional[str] = None,
                            date_range: Optional[Tuple[str, str]] = None,
                            tags: Optional[List[str]] = None,
                            min_sharpe: Optional[float] = None,
                            sort_by: str = 'creation_timestamp',
                            ascending: bool = False) -> pd.DataFrame:
        """
        List available backtest results with filtering.
        
        Args:
            strategy_name: Filter by strategy name
            date_range: Filter by date range (start, end)
            tags: Filter by tags
            min_sharpe: Minimum Sharpe ratio filter
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            DataFrame with backtest result summaries
        """
        logger.info("Listing backtest results")
        
        try:
            # Query database
            conn = sqlite3.connect(self.base_directory / "metadata.db")
            
            # Build query
            query = "SELECT * FROM backtest_metadata WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if date_range:
                query += " AND start_date >= ? AND end_date <= ?"
                params.extend(date_range)
            
            if min_sharpe is not None:
                query += " AND sharpe_ratio >= ?"
                params.append(min_sharpe)
            
            if tags:
                # Handle tags filtering (simplified)
                tag_conditions = " AND (".join([f"tags LIKE ?" for _ in tags])
                if tag_conditions:
                    query += " AND (" + tag_conditions + ")"
                    params.extend([f"%{tag}%" for tag in tags])
            
            # Add sorting
            query += f" ORDER BY {sort_by}"
            if not ascending:
                query += " DESC"
            
            results_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            logger.info(f"Found {len(results_df)} backtest results")
            return results_df
            
        except Exception as e:
            logger.error(f"Error listing backtest results: {str(e)}")
            return pd.DataFrame()
    
    def list_optimization_results(self,
                                optimization_method: Optional[str] = None,
                                min_metric_value: Optional[float] = None,
                                sort_by: str = 'creation_timestamp',
                                ascending: bool = False) -> pd.DataFrame:
        """
        List available optimization results with filtering.
        
        Args:
            optimization_method: Filter by optimization method
            min_metric_value: Minimum metric value filter
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            DataFrame with optimization result summaries
        """
        logger.info("Listing optimization results")
        
        try:
            conn = sqlite3.connect(self.base_directory / "metadata.db")
            
            query = "SELECT * FROM optimization_metadata WHERE 1=1"
            params = []
            
            if optimization_method:
                query += " AND optimization_method = ?"
                params.append(optimization_method)
            
            if min_metric_value is not None:
                query += " AND best_metric_value >= ?"
                params.append(min_metric_value)
            
            query += f" ORDER BY {sort_by}"
            if not ascending:
                query += " DESC"
            
            results_df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            logger.info(f"Found {len(results_df)} optimization results")
            return results_df
            
        except Exception as e:
            logger.error(f"Error listing optimization results: {str(e)}")
            return pd.DataFrame()
    
    def compare_backtest_results(self, 
                               result_ids: List[str],
                               metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            result_ids: List of result IDs to compare
            metrics: List of metrics to include in comparison
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(result_ids)} backtest results")
        
        if metrics is None:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        
        comparison_data = []
        
        for result_id in result_ids:
            try:
                results = self.load_backtest_results(result_id)
                metadata = results['metadata']
                
                # Extract metrics
                row = {
                    'result_id': result_id,
                    'strategy_name': metadata['strategy_name'],
                    'start_date': metadata['start_date'],
                    'end_date': metadata['end_date']
                }
                
                # Add requested metrics
                for metric in metrics:
                    row[metric] = metadata.get(metric, np.nan)
                
                # Add parameter information
                params = metadata.get('parameters', {})
                for param_name, param_value in params.items():
                    row[f'param_{param_name}'] = param_value
                
                comparison_data.append(row)
                
            except Exception as e:
                logger.warning(f"Error loading results for comparison {result_id}: {str(e)}")
                continue
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info(f"Successfully compared {len(comparison_df)} results")
        return comparison_df
    
    def aggregate_portfolio_performance(self,
                                      result_ids: List[str],
                                      weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Aggregate performance across multiple backtest results.
        
        Args:
            result_ids: List of result IDs to aggregate
            weights: Optional weights for each result
            
        Returns:
            Dictionary with aggregated performance metrics
        """
        logger.info(f"Aggregating performance across {len(result_ids)} results")
        
        if weights is None:
            weights = [1.0 / len(result_ids)] * len(result_ids)
        elif len(weights) != len(result_ids):
            raise ValueError("Number of weights must match number of result IDs")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Load all results
        portfolio_returns = []
        metadata_list = []
        
        for result_id in result_ids:
            try:
                results = self.load_backtest_results(result_id)
                portfolio_returns.append(results['portfolio_returns'])
                metadata_list.append(results['metadata'])
            except Exception as e:
                logger.warning(f"Error loading results {result_id}: {str(e)}")
                continue
        
        if not portfolio_returns:
            raise ValueError("No valid results loaded for aggregation")
        
        # Align time series
        aligned_returns = pd.concat(portfolio_returns, axis=1, join='inner')
        
        if len(aligned_returns) == 0:
            raise ValueError("No overlapping dates found across results")
        
        # Calculate weighted portfolio returns
        weighted_returns = (aligned_returns * weights).sum(axis=1)
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_portfolio_metrics(weighted_returns)
        
        aggregation_results = {
            'aggregated_returns': weighted_returns,
            'component_weights': dict(zip(result_ids, weights)),
            'metrics': aggregated_metrics,
            'component_metadata': dict(zip(result_ids, metadata_list)),
            'aggregation_date': datetime.now().isoformat()
        }
        
        logger.info("Portfolio aggregation completed")
        return aggregation_results
    
    def export_results(self,
                      result_id: str,
                      export_format: str = 'excel',
                      output_path: Optional[str] = None) -> str:
        """
        Export backtest results to various formats.
        
        Args:
            result_id: Result ID to export
            export_format: Export format ('excel', 'csv', 'json', 'pdf')
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        logger.info(f"Exporting results {result_id} in {export_format} format")
        
        # Load results
        results = self.load_backtest_results(result_id)
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{result_id}_{export_format}_{timestamp}"
        
        output_file = Path(output_path)
        
        try:
            if export_format.lower() == 'excel':
                output_file = output_file.with_suffix('.xlsx')
                self._export_to_excel(results, output_file)
                
            elif export_format.lower() == 'csv':
                output_file = output_file.with_suffix('.csv')
                self._export_to_csv(results, output_file)
                
            elif export_format.lower() == 'json':
                output_file = output_file.with_suffix('.json')
                self._export_to_json(results, output_file)
                
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Results exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            raise
    
    def cleanup_results(self,
                       older_than_days: int = 30,
                       keep_best_n: int = 10,
                       dry_run: bool = True) -> Dict[str, int]:
        """
        Clean up old results based on criteria.
        
        Args:
            older_than_days: Remove results older than N days
            keep_best_n: Keep N best results regardless of age
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with cleanup statistics
        """
        logger.info(f"Starting cleanup - older than {older_than_days} days, keep best {keep_best_n}")
        
        # Get all backtest results
        all_results = self.list_backtest_results(sort_by='sharpe_ratio', ascending=False)
        
        if len(all_results) == 0:
            return {'deleted': 0, 'kept': 0}
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        cutoff_str = cutoff_date.isoformat()
        
        # Identify results to keep
        keep_results = set()
        
        # Keep best N results
        if keep_best_n > 0:
            best_results = all_results.head(keep_best_n)
            keep_results.update(best_results['result_id'].tolist())
        
        # Keep recent results
        recent_results = all_results[all_results['creation_timestamp'] > cutoff_str]
        keep_results.update(recent_results['result_id'].tolist())
        
        # Identify results to delete
        all_result_ids = set(all_results['result_id'].tolist())
        delete_results = all_result_ids - keep_results
        
        logger.info(f"Found {len(delete_results)} results to delete, {len(keep_results)} to keep")
        
        if not dry_run:
            # Actually delete results
            for result_id in delete_results:
                try:
                    self._delete_backtest_result(result_id)
                except Exception as e:
                    logger.warning(f"Error deleting result {result_id}: {str(e)}")
        
        cleanup_stats = {
            'deleted': len(delete_results),
            'kept': len(keep_results),
            'dry_run': dry_run
        }
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary with storage information
        """
        def get_dir_size(path):
            total = 0
            for entry in Path(path).rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
            return total
        
        backtests_size = get_dir_size(self.base_directory / "backtests")
        optimizations_size = get_dir_size(self.base_directory / "optimizations")
        total_size = backtests_size + optimizations_size
        
        # Count files
        n_backtest_results = len(list((self.base_directory / "backtests").glob("*")))
        n_optimization_results = len(list((self.base_directory / "optimizations").glob("*")))
        
        storage_info = {
            'total_size_gb': total_size / (1024**3),
            'backtests_size_gb': backtests_size / (1024**3),
            'optimizations_size_gb': optimizations_size / (1024**3),
            'n_backtest_results': n_backtest_results,
            'n_optimization_results': n_optimization_results,
            'max_storage_gb': self.max_storage_gb,
            'usage_percentage': (total_size / (1024**3)) / self.max_storage_gb * 100
        }
        
        return storage_info
    
    # Helper methods
    def _initialize_directory_structure(self):
        """Initialize directory structure."""
        directories = [
            self.base_directory,
            self.base_directory / "backtests",
            self.base_directory / "optimizations",
            self.base_directory / "exports",
            self.base_directory / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _initialize_metadata_database(self):
        """Initialize SQLite database for metadata."""
        db_path = self.base_directory / "metadata.db"
        conn = sqlite3.connect(db_path)
        
        # Create backtest metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_metadata (
                result_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                start_date TEXT,
                end_date TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                creation_timestamp TEXT,
                description TEXT,
                tags TEXT
            )
        """)
        
        # Create optimization metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS optimization_metadata (
                result_id TEXT PRIMARY KEY,
                optimization_method TEXT,
                optimization_metric TEXT,
                best_metric_value REAL,
                n_trials INTEGER,
                creation_timestamp TEXT,
                description TEXT,
                tags TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _generate_result_id(self, strategy_name: str, parameters: Dict[str, Any]) -> str:
        """Generate unique result ID."""
        # Create hash from strategy name, parameters, and timestamp
        content = f"{strategy_name}_{parameters}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"{strategy_name}_{hash_obj.hexdigest()[:8]}"
    
    def _generate_optimization_id(self, method: str, parameter_space: Dict[str, Any]) -> str:
        """Generate unique optimization ID."""
        content = f"{method}_{parameter_space}_{datetime.now().isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"{method}_{hash_obj.hexdigest()[:8]}"
    
    def _save_time_series_data(self, backtest_results: Dict[str, Any], result_dir: Path):
        """Save time series data to parquet files."""
        # Portfolio returns and values
        main_data = pd.DataFrame({
            'portfolio_returns': backtest_results['portfolio_returns'],
            'portfolio_value': backtest_results['portfolio_value'],
            'transaction_costs': backtest_results.get('transaction_costs', 0),
            'turnover': backtest_results.get('turnover', 0),
            'drawdowns': backtest_results.get('drawdowns', 0)
        })
        main_data.to_parquet(result_dir / "portfolio_performance.parquet", compression='snappy')
        
        # Portfolio weights
        if 'portfolio_weights' in backtest_results:
            backtest_results['portfolio_weights'].to_parquet(
                result_dir / "portfolio_weights.parquet", compression='snappy'
            )
        
        # Asset returns
        if 'asset_returns' in backtest_results:
            backtest_results['asset_returns'].to_parquet(
                result_dir / "asset_returns.parquet", compression='snappy'
            )
        
        # Signals
        if 'signals' in backtest_results:
            backtest_results['signals'].to_parquet(
                result_dir / "signals.parquet", compression='snappy'
            )
    
    def _save_optimization_data(self, optimization_results: Dict[str, Any], result_dir: Path):
        """Save optimization data."""
        # Save main results as JSON
        main_file = result_dir / "optimization_results.json"
        with open(main_file, 'w') as f:
            # Prepare for JSON serialization
            serializable_results = self._prepare_for_json(optimization_results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save detailed results as parquet if available
        if 'all_results' in optimization_results:
            results_df = optimization_results['all_results']
            if isinstance(results_df, pd.DataFrame):
                results_df.to_parquet(result_dir / "detailed_results.parquet", compression='snappy')
    
    def _load_time_series_data(self, result_dir: Path) -> Dict[str, Any]:
        """Load time series data from parquet files."""
        results = {}
        
        # Load main portfolio data
        main_file = result_dir / "portfolio_performance.parquet"
        if main_file.exists():
            main_data = pd.read_parquet(main_file)
            for col in main_data.columns:
                results[col] = main_data[col]
        
        # Load portfolio weights
        weights_file = result_dir / "portfolio_weights.parquet"
        if weights_file.exists():
            results['portfolio_weights'] = pd.read_parquet(weights_file)
        
        # Load asset returns
        returns_file = result_dir / "asset_returns.parquet"
        if returns_file.exists():
            results['asset_returns'] = pd.read_parquet(returns_file)
        
        # Load signals
        signals_file = result_dir / "signals.parquet"
        if signals_file.exists():
            results['signals'] = pd.read_parquet(signals_file)
        
        return results
    
    def _load_optimization_data(self, result_dir: Path) -> Dict[str, Any]:
        """Load optimization data."""
        # Load main results
        main_file = result_dir / "optimization_results.json"
        with open(main_file, 'r') as f:
            results = json.load(f)
        
        # Load detailed results if available
        detailed_file = result_dir / "detailed_results.parquet"
        if detailed_file.exists():
            results['all_results'] = pd.read_parquet(detailed_file)
        
        return results
    
    def _save_metadata(self, metadata: BacktestMetadata, result_dir: Path, description: str, tags: List[str]):
        """Save metadata to JSON file."""
        metadata_file = result_dir / "metadata.json"
        metadata_dict = asdict(metadata)
        
        # Add additional info
        metadata_dict['description'] = description
        metadata_dict['tags'] = tags or []
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def _update_metadata_database(self, result_id: str, metadata: BacktestMetadata, description: str, tags: List[str]):
        """Update metadata database."""
        conn = sqlite3.connect(self.base_directory / "metadata.db")
        
        conn.execute("""
            INSERT OR REPLACE INTO backtest_metadata 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_id,
            metadata.strategy_name,
            metadata.start_date,
            metadata.end_date,
            metadata.total_return,
            metadata.sharpe_ratio,
            metadata.max_drawdown,
            metadata.creation_timestamp,
            description,
            json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
    
    def _update_optimization_database(self, result_id: str, metadata: OptimizationMetadata, description: str, tags: List[str]):
        """Update optimization database."""
        conn = sqlite3.connect(self.base_directory / "metadata.db")
        
        conn.execute("""
            INSERT OR REPLACE INTO optimization_metadata 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result_id,
            metadata.optimization_method,
            metadata.optimization_metric,
            metadata.best_metric_value,
            metadata.n_trials,
            metadata.creation_timestamp,
            description,
            json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
    
    def _calculate_total_return(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate total return from backtest results."""
        portfolio_value = backtest_results.get('portfolio_value')
        if portfolio_value is not None and len(portfolio_value) > 0:
            initial_value = backtest_results['metadata']['initial_capital']
            return (portfolio_value.iloc[-1] / initial_value) - 1.0
        return 0.0
    
    def _calculate_sharpe_ratio(self, backtest_results: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio from backtest results."""
        portfolio_returns = backtest_results.get('portfolio_returns')
        if portfolio_returns is not None and len(portfolio_returns) > 1:
            clean_returns = portfolio_returns.dropna()
            if len(clean_returns) > 0 and clean_returns.std() > 0:
                return clean_returns.mean() / clean_returns.std() * np.sqrt(252)
        return 0.0
    
    def _calculate_data_hash(self, backtest_results: Dict[str, Any]) -> str:
        """Calculate hash of key result data for integrity checking."""
        portfolio_returns = backtest_results.get('portfolio_returns', pd.Series())
        hash_content = str(portfolio_returns.sum()) + str(len(portfolio_returns))
        return hashlib.md5(hash_content.encode()).hexdigest()[:16]
    
    def _extract_best_metric_value(self, optimization_results: Dict[str, Any]) -> float:
        """Extract best metric value from optimization results."""
        best_metrics = optimization_results.get('best_metrics', {})
        if best_metrics:
            # Find validation metric
            for key, value in best_metrics.items():
                if key.startswith('val_') and isinstance(value, (int, float)):
                    return float(value)
        return 0.0
    
    def _extract_n_trials(self, optimization_results: Dict[str, Any]) -> int:
        """Extract number of trials from optimization results."""
        metadata = optimization_results.get('metadata', {})
        return metadata.get('n_calls', metadata.get('n_combinations_tested', 0))
    
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate standard portfolio metrics."""
        clean_returns = returns.dropna()
        
        if len(clean_returns) == 0:
            return {}
        
        total_return = (1 + clean_returns).prod() - 1.0
        annualized_return = (1 + total_return) ** (252 / len(clean_returns)) - 1.0
        volatility = clean_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
        
        # Calculate max drawdown
        cumulative = (1 + clean_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (clean_returns > 0).mean()
        }
    
    def _delete_backtest_result(self, result_id: str):
        """Delete a backtest result."""
        result_dir = self.base_directory / "backtests" / result_id
        
        if result_dir.exists():
            shutil.rmtree(result_dir)
        
        # Remove from database
        conn = sqlite3.connect(self.base_directory / "metadata.db")
        conn.execute("DELETE FROM backtest_metadata WHERE result_id = ?", (result_id,))
        conn.commit()
        conn.close()
    
    def _cleanup_old_results(self):
        """Automatic cleanup based on storage limits."""
        storage_info = self.get_storage_usage()
        
        if storage_info['usage_percentage'] > 90:  # Over 90% of limit
            logger.info("Storage usage over 90%, performing automatic cleanup")
            self.cleanup_results(older_than_days=15, keep_best_n=20, dry_run=False)
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj) or obj is None:
            return None
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _export_to_excel(self, results: Dict[str, Any], output_file: Path):
        """Export results to Excel format."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Start Date', 'End Date'],
                'Value': [
                    results['metadata'].get('total_return', 0),
                    results['metadata'].get('sharpe_ratio', 0),
                    results['metadata'].get('max_drawdown', 0),
                    results['metadata'].get('start_date', ''),
                    results['metadata'].get('end_date', '')
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Portfolio performance
            if 'portfolio_returns' in results:
                perf_data = pd.DataFrame({
                    'Date': results['portfolio_returns'].index,
                    'Returns': results['portfolio_returns'].values,
                    'Value': results.get('portfolio_value', 0)
                })
                perf_data.to_excel(writer, sheet_name='Performance', index=False)
            
            # Portfolio weights
            if 'portfolio_weights' in results:
                results['portfolio_weights'].to_excel(writer, sheet_name='Weights')
    
    def _export_to_csv(self, results: Dict[str, Any], output_file: Path):
        """Export results to CSV format."""
        if 'portfolio_returns' in results:
            export_data = pd.DataFrame({
                'Date': results['portfolio_returns'].index,
                'Returns': results['portfolio_returns'].values,
                'Value': results.get('portfolio_value', 0)
            })
            export_data.to_csv(output_file, index=False)
    
    def _export_to_json(self, results: Dict[str, Any], output_file: Path):
        """Export results to JSON format."""
        serializable_results = self._prepare_for_json(results)
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("ResultsManager module loaded successfully")
    print("Module ready for use. Import and use with backtesting framework.")