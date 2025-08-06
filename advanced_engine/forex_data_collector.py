#!/usr/bin/env python3
"""
Forex Data Collection System for Contrarian Trading Strategy

This module downloads and processes forex data for backtesting purposes.
Ensures data quality, handles missing values, and creates analysis-ready datasets.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ForexDataCollector:
    """
    Comprehensive forex data collection and processing system
    """
    
    def __init__(self, data_dir: str = "advanced_engine/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 20 most liquid forex pairs as specified
        self.forex_pairs = [
            # Major pairs
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X',
            # Cross pairs  
            'EURGBP=X', 'EURJPY=X', 'EURCHF=X', 'EURAUD=X', 
            'EURCAD=X', 'EURNZD=X', 'GBPJPY=X', 'GBPCHF=X', 
            'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X', 'AUDJPY=X', 'AUDCHF=X'
        ]
        
        self.start_date = '2000-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Data quality tracking
        self.data_quality_report = {}
        
    def download_single_pair(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single forex pair with error handling
        
        Args:
            symbol: Forex pair symbol (e.g., 'EURUSD=X')
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Downloading data for {symbol}")
            
            # Download data with extended period for maximum coverage
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval='1d',
                auto_adjust=False,  # We'll handle adjustment manually
                prepost=False,
                back_adjust=False,
                repair=True  # Fix bad data
            )
            
            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
            
            # Standardize column names and ensure proper data types
            data.columns = data.columns.str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                return None
            
            # Use Close instead of Adj Close for forex (no dividends/splits)
            data = data[required_cols].copy()
            
            # Convert to optimal data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            # Validate OHLC relationships
            data = self._validate_ohlc_data(data, symbol)
            
            # Log data quality metrics
            self._log_data_quality(data, symbol)
            
            logger.info(f"Successfully downloaded {len(data)} rows for {symbol}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {symbol}: {str(e)}")
            return None
    
    def _validate_ohlc_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate OHLC relationships and fix obvious errors
        """
        logger.info(f"Validating OHLC data for {symbol}")
        
        # Check for invalid OHLC relationships
        invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
        invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
        
        if invalid_high.any():
            logger.warning(f"Found {invalid_high.sum()} rows with invalid high prices for {symbol}")
            # Fix by setting high to max of open/close
            data.loc[invalid_high, 'high'] = data.loc[invalid_high, ['open', 'close']].max(axis=1)
        
        if invalid_low.any():
            logger.warning(f"Found {invalid_low.sum()} rows with invalid low prices for {symbol}")
            # Fix by setting low to min of open/close
            data.loc[invalid_low, 'low'] = data.loc[invalid_low, ['open', 'close']].min(axis=1)
        
        # Check for extreme outliers using IQR method
        for col in ['open', 'high', 'low', 'close']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} potential outliers in {col} for {symbol}")
                # Don't automatically remove outliers for forex - they might be valid extreme moves
        
        return data
    
    def _log_data_quality(self, data: pd.DataFrame, symbol: str):
        """
        Log comprehensive data quality metrics
        """
        total_rows = len(data)
        missing_data = data.isnull().sum()
        
        quality_metrics = {
            'symbol': symbol,
            'total_rows': total_rows,
            'date_range': f"{data.index.min()} to {data.index.max()}",
            'missing_open': missing_data.get('open', 0),
            'missing_high': missing_data.get('high', 0),
            'missing_low': missing_data.get('low', 0),
            'missing_close': missing_data.get('close', 0),
            'missing_volume': missing_data.get('volume', 0),
            'missing_percentage': (missing_data.sum() / (total_rows * 5)) * 100,
            'zero_volume_days': (data['volume'] == 0).sum() if 'volume' in data.columns else 0
        }
        
        self.data_quality_report[symbol] = quality_metrics
        
        logger.info(f"Data quality for {symbol}:")
        logger.info(f"  Total rows: {total_rows}")
        logger.info(f"  Missing data percentage: {quality_metrics['missing_percentage']:.2f}%")
        logger.info(f"  Zero volume days: {quality_metrics['zero_volume_days']}")
    
    def handle_missing_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Handle missing data using forward fill for financial data
        """
        logger.info(f"Handling missing data for {symbol}")
        
        # Count missing values before processing
        missing_before = data.isnull().sum().sum()
        
        # Forward fill missing values (appropriate for financial time series)
        data_filled = data.fillna(method='ffill')
        
        # Backward fill any remaining missing values at the beginning
        data_filled = data_filled.fillna(method='bfill')
        
        # Count missing values after processing
        missing_after = data_filled.isnull().sum().sum()
        
        if missing_before > 0:
            logger.info(f"Filled {missing_before - missing_after} missing values for {symbol}")
        
        return data_filled
    
    def synchronize_trading_dates(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Synchronize all forex pairs to common trading dates
        """
        logger.info("Synchronizing trading dates across all pairs")
        
        # Find common date range
        all_dates = []
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                all_dates.extend(data.index.tolist())
        
        if not all_dates:
            logger.error("No valid data found for synchronization")
            return data_dict
        
        # Create complete business day range (forex trades Mon-Fri)
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Generate business days (excluding weekends)
        complete_date_range = pd.bdate_range(start=min_date, end=max_date, freq='B')
        
        logger.info(f"Synchronizing to date range: {min_date} to {max_date}")
        logger.info(f"Total business days in range: {len(complete_date_range)}")
        
        # Reindex all datasets to common date range
        synchronized_data = {}
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                # Reindex to complete business day range
                synchronized = data.reindex(complete_date_range)
                
                # Forward fill missing values (forex markets are continuous Mon-Fri)
                synchronized = synchronized.fillna(method='ffill')
                
                # Remove any remaining NaN values at the beginning
                synchronized = synchronized.dropna()
                
                synchronized_data[symbol] = synchronized
                logger.info(f"Synchronized {symbol}: {len(synchronized)} rows")
            else:
                logger.warning(f"Skipping {symbol} - no valid data")
        
        return synchronized_data
    
    def calculate_returns(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate daily returns from close prices
        """
        returns_data = pd.DataFrame(index=data.index)
        
        # Calculate simple daily returns
        returns_data[f'{symbol}_returns'] = data['close'].pct_change()
        
        # Calculate log returns for risk calculations
        returns_data[f'{symbol}_log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Remove first row with NaN
        returns_data = returns_data.dropna()
        
        return returns_data
    
    def save_individual_files(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Save individual parquet files for each forex pair
        """
        logger.info("Saving individual parquet files")
        
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                # Clean symbol name for filename
                clean_symbol = symbol.replace('=', '_')
                filename = self.data_dir / f"{clean_symbol}.parquet"
                
                # Save with compression for optimal file size
                data.to_parquet(
                    filename,
                    compression='snappy',
                    engine='pyarrow'
                )
                
                logger.info(f"Saved {clean_symbol} to {filename}")
    
    def create_unified_datasets(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Create unified datasets with all pairs and returns
        """
        logger.info("Creating unified datasets")
        
        # Create unified OHLCV dataset
        all_close_prices = {}
        all_returns = {}
        
        for symbol, data in data_dict.items():
            if data is not None and not data.empty:
                clean_symbol = symbol.replace('=X', '').replace('=', '_')
                
                # Add close prices to unified dataset
                all_close_prices[clean_symbol] = data['close']
                
                # Calculate and add returns
                returns = data['close'].pct_change().dropna()
                all_returns[clean_symbol] = returns
        
        # Create DataFrames
        if all_close_prices:
            unified_prices = pd.DataFrame(all_close_prices)
            unified_prices.to_parquet(
                self.data_dir / "all_pairs_data.parquet",
                compression='snappy',
                engine='pyarrow'
            )
            logger.info(f"Saved unified prices dataset: {unified_prices.shape}")
        
        if all_returns:
            unified_returns = pd.DataFrame(all_returns)
            unified_returns.to_parquet(
                self.data_dir / "all_pairs_returns.parquet",
                compression='snappy',
                engine='pyarrow'
            )
            logger.info(f"Saved unified returns dataset: {unified_returns.shape}")
    
    def generate_data_summary_report(self) -> pd.DataFrame:
        """
        Generate comprehensive data summary report
        """
        logger.info("Generating data summary report")
        
        if not self.data_quality_report:
            logger.warning("No data quality information available")
            return pd.DataFrame()
        
        # Convert to DataFrame
        report_df = pd.DataFrame.from_dict(self.data_quality_report, orient='index')
        
        # Add summary statistics
        report_df['data_completeness'] = 100 - report_df['missing_percentage']
        report_df['has_volume_issues'] = report_df['zero_volume_days'] > 0
        
        # Sort by data completeness
        report_df = report_df.sort_values('data_completeness', ascending=False)
        
        # Save report
        report_path = self.data_dir / "data_quality_report.parquet"
        report_df.to_parquet(report_path, compression='snappy', engine='pyarrow')
        
        # Also save as CSV for easy reading
        csv_path = self.data_dir / "data_quality_report.csv"
        report_df.to_csv(csv_path, index=True)
        
        logger.info(f"Data quality report saved to {report_path}")
        
        # Log summary
        logger.info("=== DATA COLLECTION SUMMARY ===")
        logger.info(f"Total pairs processed: {len(report_df)}")
        logger.info(f"Average data completeness: {report_df['data_completeness'].mean():.2f}%")
        logger.info(f"Best quality pair: {report_df.index[0]} ({report_df['data_completeness'].iloc[0]:.2f}%)")
        logger.info(f"Lowest quality pair: {report_df.index[-1]} ({report_df['data_completeness'].iloc[-1]:.2f}%)")
        
        return report_df
    
    def run_complete_collection(self) -> bool:
        """
        Execute the complete data collection pipeline
        """
        logger.info("=== STARTING FOREX DATA COLLECTION ===")
        logger.info(f"Target pairs: {len(self.forex_pairs)}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        
        try:
            # Step 1: Download all pairs
            logger.info("Step 1: Downloading forex data")
            raw_data = {}
            successful_downloads = 0
            
            for symbol in self.forex_pairs:
                data = self.download_single_pair(symbol)
                if data is not None:
                    raw_data[symbol] = data
                    successful_downloads += 1
                else:
                    logger.error(f"Failed to download {symbol}")
            
            logger.info(f"Successfully downloaded {successful_downloads}/{len(self.forex_pairs)} pairs")
            
            if not raw_data:
                logger.error("No data downloaded successfully. Aborting.")
                return False
            
            # Step 2: Handle missing data
            logger.info("Step 2: Processing missing data")
            for symbol in raw_data:
                raw_data[symbol] = self.handle_missing_data(raw_data[symbol], symbol)
            
            # Step 3: Synchronize trading dates
            logger.info("Step 3: Synchronizing trading dates")
            synchronized_data = self.synchronize_trading_dates(raw_data)
            
            # Step 4: Save individual files
            logger.info("Step 4: Saving individual parquet files")
            self.save_individual_files(synchronized_data)
            
            # Step 5: Create unified datasets
            logger.info("Step 5: Creating unified datasets")
            self.create_unified_datasets(synchronized_data)
            
            # Step 6: Generate summary report
            logger.info("Step 6: Generating summary report")
            self.generate_data_summary_report()
            
            logger.info("=== DATA COLLECTION COMPLETED SUCCESSFULLY ===")
            return True
            
        except Exception as e:
            logger.error(f"Critical error during data collection: {str(e)}")
            return False


def main():
    """
    Main execution function
    """
    collector = ForexDataCollector()
    success = collector.run_complete_collection()
    
    if success:
        print("✅ Forex data collection completed successfully!")
        print(f"📁 Data saved to: {collector.data_dir}")
        print("📊 Check data_quality_report.csv for detailed metrics")
    else:
        print("❌ Data collection failed. Check logs for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())