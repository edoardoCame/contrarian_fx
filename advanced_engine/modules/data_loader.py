#!/usr/bin/env python3
"""
Data Loader Module for Contrarian Forex Trading System

This module provides efficient data loading, alignment, and preprocessing
functions for the backtesting system. Ensures data integrity and avoids
lookahead bias.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ForexDataLoader:
    """
    Efficient forex data loader with built-in validation and preprocessing
    """
    
    def __init__(self, data_dir: str = "advanced_engine/data"):
        self.data_dir = Path(data_dir)
        self.cache = {}  # Simple caching mechanism
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_individual_pair(self, symbol: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load data for a single forex pair
        
        Args:
            symbol: Forex pair symbol (e.g., 'EURUSD' or 'EURUSD=X')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        # Normalize symbol name
        clean_symbol = symbol.replace('=X', '').replace('=', '_')
        if not clean_symbol.endswith('_X'):
            clean_symbol += '_X'
        
        # Check cache first
        cache_key = f"individual_{clean_symbol}"
        if use_cache and cache_key in self.cache:
            logger.debug(f"Returning cached data for {clean_symbol}")
            return self.cache[cache_key].copy()
        
        # Construct file path
        file_path = self.data_dir / f"{clean_symbol}.parquet"
        
        try:
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                return None
            
            # Load data with proper index handling
            data = pd.read_parquet(file_path, engine='pyarrow')
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    logger.error(f"Cannot identify date column for {clean_symbol}")
                    return None
            
            # Ensure proper column names
            expected_columns = ['open', 'high', 'low', 'close', 'volume']
            data.columns = data.columns.str.lower()
            
            # Validate required columns exist
            missing_cols = [col for col in expected_columns if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {clean_symbol}: {missing_cols}")
            
            # Ensure proper data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Sort by date to ensure chronological order
            data = data.sort_index()
            
            # Basic data validation
            if len(data) == 0:
                logger.warning(f"Empty dataset for {clean_symbol}")
                return None
            
            # Cache the data
            if use_cache:
                self.cache[cache_key] = data.copy()
            
            logger.info(f"Loaded {clean_symbol}: {len(data)} rows from {data.index.min()} to {data.index.max()}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {clean_symbol}: {str(e)}")
            return None
    
    def load_multiple_pairs(self, symbols: List[str], use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple forex pairs
        
        Args:
            symbols: List of forex pair symbols
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Loading {len(symbols)} forex pairs")
        
        data_dict = {}
        successful_loads = 0
        
        for symbol in symbols:
            data = self.load_individual_pair(symbol, use_cache=use_cache)
            if data is not None:
                data_dict[symbol] = data
                successful_loads += 1
            else:
                logger.warning(f"Failed to load {symbol}")
        
        logger.info(f"Successfully loaded {successful_loads}/{len(symbols)} pairs")
        return data_dict
    
    def load_unified_prices(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load the unified prices dataset
        
        Returns:
            DataFrame with all pairs' close prices or None if not found
        """
        cache_key = "unified_prices"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        file_path = self.data_dir / "all_pairs_data.parquet"
        
        try:
            if not file_path.exists():
                logger.warning(f"Unified prices file not found: {file_path}")
                return None
            
            data = pd.read_parquet(file_path, engine='pyarrow')
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            if use_cache:
                self.cache[cache_key] = data.copy()
            
            logger.info(f"Loaded unified prices: {data.shape} from {data.index.min()} to {data.index.max()}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading unified prices: {str(e)}")
            return None
    
    def load_unified_returns(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load the unified returns dataset
        
        Returns:
            DataFrame with all pairs' returns or None if not found
        """
        cache_key = "unified_returns"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        file_path = self.data_dir / "all_pairs_returns.parquet"
        
        try:
            if not file_path.exists():
                logger.warning(f"Unified returns file not found: {file_path}")
                return None
            
            data = pd.read_parquet(file_path, engine='pyarrow')
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            if use_cache:
                self.cache[cache_key] = data.copy()
            
            logger.info(f"Loaded unified returns: {data.shape} from {data.index.min()} to {data.index.max()}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading unified returns: {str(e)}")
            return None
    
    def calculate_returns_from_prices(self, price_data: pd.DataFrame, 
                                    method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            price_data: DataFrame with price data
            method: 'simple' for arithmetic returns, 'log' for log returns
            
        Returns:
            DataFrame with calculated returns
        """
        if method == 'simple':
            returns = price_data.pct_change()
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(1))
        else:
            raise ValueError("method must be 'simple' or 'log'")
        
        # Remove first row with NaN values
        returns = returns.dropna()
        
        logger.info(f"Calculated {method} returns: {returns.shape}")
        return returns
    
    def align_data_by_dates(self, data_dict: Dict[str, pd.DataFrame], 
                          method: str = 'inner') -> Dict[str, pd.DataFrame]:
        """
        Align multiple datasets by common dates
        
        Args:
            data_dict: Dictionary of DataFrames to align
            method: 'inner' for common dates only, 'outer' for all dates
            
        Returns:
            Dictionary of aligned DataFrames
        """
        if not data_dict:
            return {}
        
        logger.info(f"Aligning {len(data_dict)} datasets using {method} method")
        
        # Find common date range
        all_dates = []
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                all_dates.append(set(data.index))
        
        if not all_dates:
            logger.warning("No valid data found for alignment")
            return data_dict
        
        if method == 'inner':
            # Use intersection of all dates
            common_dates = set.intersection(*all_dates)
        elif method == 'outer':
            # Use union of all dates
            common_dates = set.union(*all_dates)
        else:
            raise ValueError("method must be 'inner' or 'outer'")
        
        common_dates = sorted(list(common_dates))
        logger.info(f"Aligned to {len(common_dates)} common dates")
        
        # Reindex all datasets
        aligned_data = {}
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                aligned = data.reindex(common_dates)
                
                if method == 'outer':
                    # Forward fill missing values for outer join
                    aligned = aligned.fillna(method='ffill')
                    # Remove any remaining NaN values
                    aligned = aligned.dropna()
                
                aligned_data[symbol] = aligned
                logger.debug(f"Aligned {symbol}: {len(aligned)} rows")
        
        return aligned_data
    
    def get_data_for_period(self, start_date: str, end_date: str, 
                          symbols: Optional[List[str]] = None,
                          data_type: str = 'prices') -> Optional[pd.DataFrame]:
        """
        Get data for a specific time period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: List of symbols to include (None for all)
            data_type: 'prices' or 'returns'
            
        Returns:
            DataFrame with data for the specified period
        """
        logger.info(f"Loading {data_type} data for period {start_date} to {end_date}")
        
        # Load appropriate dataset
        if data_type == 'prices':
            data = self.load_unified_prices()
        elif data_type == 'returns':
            data = self.load_unified_returns()
        else:
            raise ValueError("data_type must be 'prices' or 'returns'")
        
        if data is None:
            logger.error(f"Failed to load {data_type} data")
            return None
        
        # Filter by date range
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Handle timezone-aware data
            if data.index.tz is not None:
                # If data has timezone, convert filter dates to same timezone
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize('UTC').tz_convert(data.index.tz)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC').tz_convert(data.index.tz)
            
            mask = (data.index >= start_dt) & (data.index <= end_dt)
            filtered_data = data[mask]
            
            # Filter by symbols if specified
            if symbols is not None:
                # Normalize symbol names
                available_cols = data.columns.tolist()
                matched_cols = []
                
                for symbol in symbols:
                    clean_symbol = symbol.replace('=X', '').replace('=', '_')
                    if clean_symbol in available_cols:
                        matched_cols.append(clean_symbol)
                    else:
                        # Try variations
                        variations = [
                            clean_symbol,
                            clean_symbol.replace('_', ''),
                            f"{clean_symbol}_X" if not clean_symbol.endswith('_X') else clean_symbol[:-2]
                        ]
                        for var in variations:
                            if var in available_cols:
                                matched_cols.append(var)
                                break
                        else:
                            logger.warning(f"Symbol {symbol} not found in data")
                
                if matched_cols:
                    filtered_data = filtered_data[matched_cols]
                else:
                    logger.error("No matching symbols found")
                    return None
            
            logger.info(f"Retrieved data: {filtered_data.shape} for period {start_date} to {end_date}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering data by period: {str(e)}")
            return None
    
    def validate_data_integrity(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict:
        """
        Validate data integrity and return quality metrics
        
        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating data integrity for {symbol}")
        
        validation_results = {
            'symbol': symbol,
            'total_rows': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}" if len(data) > 0 else "No data",
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'has_duplicates': data.index.duplicated().any(),
            'is_sorted': data.index.is_monotonic_increasing,
            'data_types': data.dtypes.to_dict(),
            'gaps_in_dates': 0,
            'outliers_detected': {},
            'issues': []
        }
        
        # Check for date gaps (only for business days)
        if len(data) > 1:
            expected_dates = pd.bdate_range(start=data.index.min(), end=data.index.max())
            actual_dates = set(data.index)
            expected_dates_set = set(expected_dates)
            missing_dates = expected_dates_set - actual_dates
            validation_results['gaps_in_dates'] = len(missing_dates)
        
        # Check for outliers using IQR method
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)) & data[col].notna()
                validation_results['outliers_detected'][col] = outliers.sum()
        
        # Identify issues
        if validation_results['has_duplicates']:
            validation_results['issues'].append("Duplicate dates found")
        
        if not validation_results['is_sorted']:
            validation_results['issues'].append("Data is not sorted chronologically")
        
        if validation_results['gaps_in_dates'] > 0:
            validation_results['issues'].append(f"{validation_results['gaps_in_dates']} date gaps found")
        
        missing_pct = sum(validation_results['missing_percentage'].values()) / len(validation_results['missing_percentage'])
        if missing_pct > 5:  # More than 5% missing data
            validation_results['issues'].append(f"High missing data percentage: {missing_pct:.2f}%")
        
        # Log results
        if validation_results['issues']:
            logger.warning(f"Data integrity issues for {symbol}: {', '.join(validation_results['issues'])}")
        else:
            logger.info(f"Data integrity validation passed for {symbol}")
        
        return validation_results
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available forex pairs in the data directory
        
        Returns:
            List of available symbol names
        """
        try:
            parquet_files = list(self.data_dir.glob("*_X.parquet"))
            symbols = [f.stem for f in parquet_files]
            
            # Exclude unified datasets
            symbols = [s for s in symbols if not s.startswith('all_pairs')]
            
            logger.info(f"Found {len(symbols)} available forex pairs")
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
    def clear_cache(self):
        """
        Clear the data cache
        """
        self.cache.clear()
        logger.info("Data cache cleared")


# Convenience functions for quick data access
def load_forex_data(data_dir: str = "advanced_engine/data", 
                   symbols: Optional[List[str]] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   data_type: str = 'prices') -> Optional[pd.DataFrame]:
    """
    Quick function to load forex data
    
    Args:
        data_dir: Path to data directory
        symbols: List of symbols to load (None for all)
        start_date: Start date filter
        end_date: End date filter
        data_type: 'prices' or 'returns'
        
    Returns:
        DataFrame with requested data
    """
    loader = ForexDataLoader(data_dir)
    
    if start_date and end_date:
        return loader.get_data_for_period(start_date, end_date, symbols, data_type)
    elif data_type == 'prices':
        return loader.load_unified_prices()
    elif data_type == 'returns':
        return loader.load_unified_returns()
    else:
        raise ValueError("Invalid data_type. Must be 'prices' or 'returns'")


def validate_forex_data(data_dir: str = "advanced_engine/data") -> pd.DataFrame:
    """
    Quick function to validate all forex data and return summary
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with validation results for all pairs
    """
    loader = ForexDataLoader(data_dir)
    symbols = loader.get_available_symbols()
    
    validation_results = []
    for symbol in symbols:
        data = loader.load_individual_pair(symbol)
        if data is not None:
            result = loader.validate_data_integrity(data, symbol)
            validation_results.append(result)
    
    if validation_results:
        return pd.DataFrame(validation_results)
    else:
        return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load all forex data
    loader = ForexDataLoader()
    
    # Get available symbols
    symbols = loader.get_available_symbols()
    print(f"Available symbols: {symbols}")
    
    # Load unified prices
    prices = loader.load_unified_prices()
    if prices is not None:
        print(f"Unified prices shape: {prices.shape}")
    
    # Load specific period
    period_data = loader.get_data_for_period('2020-01-01', '2020-12-31', data_type='returns')
    if period_data is not None:
        print(f"2020 returns shape: {period_data.shape}")
    
    # Validate data
    validation_summary = validate_forex_data()
    if not validation_summary.empty:
        print("Data validation complete")
        print(validation_summary[['symbol', 'total_rows', 'gaps_in_dates']].head())