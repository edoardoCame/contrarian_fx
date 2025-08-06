"""
Daily Operations Analyzer per Strategia Contrarian Forex

Modulo che implementa tutte le funzioni di calcolo per analizzare
le operazioni giornaliere e i profitti della strategia contrarian
usando il portafoglio risk parity ottimizzato.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Tuple, List, Optional
warnings.filterwarnings('ignore')


class DailyOperationsAnalyzer:
    """Analyzer per operazioni giornaliere della strategia contrarian forex"""
    
    def __init__(self, data_dir: str = "data/backtest_results"):
        """
        Inizializza l'analyzer con directory dei dati
        
        Args:
            data_dir: Directory contenente i risultati backtest
        """
        self.data_dir = Path(data_dir)
        self.portfolio_data = None
        self.individual_returns = {}
        self.individual_equity = {}
        self.currency_list = []
        
    def load_portfolio_data(self) -> Dict:
        """
        Carica tutti i dati necessari: portfolio risk parity e dati individuali
        
        Returns:
            Dict con tutti i dati caricati
        """
        # Carica portfolio risk parity ottimizzato
        portfolio_file = self.data_dir / "risk_parity_portfolio.parquet"
        self.portfolio_data = pd.read_parquet(portfolio_file)
        
        # Assicurati che l'index sia DatetimeIndex
        if not isinstance(self.portfolio_data.index, pd.DatetimeIndex):
            self.portfolio_data.index = pd.to_datetime(self.portfolio_data.index)
        
        # Carica dati individuali
        individual_dir = self.data_dir / "individual_results"
        
        # Trova tutte le valute disponibili
        equity_files = list(individual_dir.glob("*_equity.parquet"))
        self.currency_list = [f.stem.replace("_equity", "") for f in equity_files]
        
        # Carica returns e equity per ogni valuta
        for currency in self.currency_list:
            # Carica returns
            returns_file = individual_dir / f"{currency}_returns.parquet"
            if returns_file.exists():
                self.individual_returns[currency] = pd.read_parquet(returns_file)
                # Assicura DatetimeIndex
                if not isinstance(self.individual_returns[currency].index, pd.DatetimeIndex):
                    self.individual_returns[currency].index = pd.to_datetime(
                        self.individual_returns[currency].index
                    )
            
            # Carica equity
            equity_file = individual_dir / f"{currency}_equity.parquet"
            if equity_file.exists():
                self.individual_equity[currency] = pd.read_parquet(equity_file)
                # Assicura DatetimeIndex
                if not isinstance(self.individual_equity[currency].index, pd.DatetimeIndex):
                    self.individual_equity[currency].index = pd.to_datetime(
                        self.individual_equity[currency].index
                    )
        
        return {
            'portfolio': self.portfolio_data,
            'individual_returns': self.individual_returns,
            'individual_equity': self.individual_equity,
            'currencies': self.currency_list
        }
    
    def calculate_daily_operations(self) -> pd.DataFrame:
        """
        Identifica quando ogni valuta opera (signal activation)
        La strategia contrarian opera quando il return del giorno precedente è negativo
        
        Returns:
            DataFrame con signal activation per ogni valuta
        """
        operations_df = pd.DataFrame()
        
        for currency in self.currency_list:
            if currency in self.individual_returns:
                returns_data = self.individual_returns[currency].copy()
                
                # Calcola i return giornalieri dell'underlying (non strategy returns)
                # Per ottenere i return dell'underlying, dobbiamo lavorare sull'equity
                if currency in self.individual_equity:
                    equity_data = self.individual_equity[currency]
                    # Converte equity cumulativa in price series
                    price_series = 1 + equity_data.iloc[:, 0]  # Assumendo prima colonna sia equity
                    underlying_returns = price_series.pct_change()
                    
                    # Signal activation: opera quando return precedente è negativo
                    # La strategia contrarian: entra long quando return precedente < 0
                    signal_activation = (underlying_returns.shift(1) < 0).astype(int)
                    operations_df[currency] = signal_activation
        
        return operations_df
    
    def calculate_daily_pnl(self) -> Dict[str, pd.Series]:
        """
        Calcola P&L giornaliero per portfolio e singole valute
        
        Returns:
            Dict con P&L giornaliero del portfolio e contributi per valuta
        """
        # P&L del portfolio risk parity
        portfolio_equity = self.portfolio_data['equity']
        portfolio_pnl = portfolio_equity.pct_change().fillna(0)
        
        # P&L individuale per ogni valuta
        individual_pnl = {}
        for currency in self.currency_list:
            if currency in self.individual_returns:
                # Strategy returns sono già i ritorni giornalieri della strategia
                returns_data = self.individual_returns[currency]
                if 'strategy_returns' in returns_data.columns:
                    individual_pnl[currency] = returns_data['strategy_returns'].fillna(0)
                else:
                    # Fallback: calcola da equity
                    if currency in self.individual_equity:
                        equity_data = self.individual_equity[currency]
                        equity_series = equity_data.iloc[:, 0]
                        individual_pnl[currency] = equity_series.pct_change().fillna(0)
        
        # Calcola contribution di ogni valuta al portfolio (approssimato)
        # Usando i pesi dal portfolio risk parity
        currency_contributions = {}
        
        # Cerca le colonne peso nel portfolio
        weight_columns = [col for col in self.portfolio_data.columns if 'weight' in col]
        
        for i, currency in enumerate(self.currency_list):
            if i < len(weight_columns):
                weight_col = weight_columns[i]
                if currency in individual_pnl:
                    # Contributo = peso * return individuale
                    weight_series = self.portfolio_data[weight_col].fillna(0)
                    individual_return = individual_pnl[currency]
                    
                    # Allinea gli indici
                    aligned_weight, aligned_return = weight_series.align(individual_return, join='inner')
                    contribution = aligned_weight * aligned_return
                    currency_contributions[currency] = contribution
        
        return {
            'portfolio_pnl': portfolio_pnl,
            'individual_pnl': individual_pnl,
            'currency_contributions': currency_contributions
        }
    
    def calculate_performance_metrics(self, pnl_series: pd.Series, 
                                    periods_per_year: int = 252) -> Dict:
        """
        Calcola metriche di performance per una serie di P&L
        
        Args:
            pnl_series: Serie di ritorni giornalieri
            periods_per_year: Periodi per anno (252 per giorni lavorativi)
            
        Returns:
            Dict con tutte le metriche di performance
        """
        pnl_clean = pnl_series.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Metriche di base
        total_return = pnl_clean.sum()
        annualized_return = pnl_clean.mean() * periods_per_year
        volatility = pnl_clean.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Hit rate (% giorni profittevoli)
        profitable_days = (pnl_clean > 0).sum()
        total_days = len(pnl_clean[pnl_clean != 0])  # Escludi giorni senza operazioni
        hit_rate = profitable_days / total_days if total_days > 0 else 0
        
        # Win/Loss ratio
        wins = pnl_clean[pnl_clean > 0]
        losses = pnl_clean[pnl_clean < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        # Drawdown
        cumulative_returns = (1 + pnl_clean).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'hit_rate': hit_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'total_days': total_days,
            'profitable_days': profitable_days
        }
    
    def calculate_attribution(self, currency_contributions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calcola contributo di ogni valuta al rendimento totale
        
        Args:
            currency_contributions: Dict con contributi per valuta
            
        Returns:
            DataFrame con analisi di attribuzione
        """
        attribution_data = []
        
        for currency, contribution in currency_contributions.items():
            if len(contribution) > 0:
                total_contribution = contribution.sum()
                avg_daily_contribution = contribution.mean()
                volatility_contribution = contribution.std()
                
                attribution_data.append({
                    'Currency': currency,
                    'Total_Contribution': total_contribution,
                    'Avg_Daily_Contribution': avg_daily_contribution,
                    'Volatility_Contribution': volatility_contribution,
                    'Contribution_Sharpe': avg_daily_contribution / volatility_contribution 
                                         if volatility_contribution > 0 else 0
                })
        
        attribution_df = pd.DataFrame(attribution_data)
        if len(attribution_df) > 0:
            attribution_df = attribution_df.sort_values('Total_Contribution', ascending=False)
        
        return attribution_df
    
    def calculate_rolling_statistics(self, pnl_series: pd.Series, 
                                   windows: List[int] = [7, 22, 66]) -> Dict[str, pd.DataFrame]:
        """
        Calcola statistiche su finestre temporali mobili
        
        Args:
            pnl_series: Serie di ritorni giornalieri
            windows: Lista di finestre in giorni
            
        Returns:
            Dict con DataFrames delle statistiche rolling
        """
        rolling_stats = {}
        
        for window in windows:
            window_name = f"{window}d"
            
            # Calcola metriche rolling
            rolling_return = pnl_series.rolling(window=window).sum()
            rolling_vol = pnl_series.rolling(window=window).std() * np.sqrt(252)
            rolling_sharpe = (pnl_series.rolling(window=window).mean() * 252) / rolling_vol
            
            # Hit rate rolling
            rolling_hit_rate = pnl_series.rolling(window=window).apply(
                lambda x: (x > 0).sum() / len(x[x != 0]) if len(x[x != 0]) > 0 else 0
            )
            
            rolling_stats[window_name] = pd.DataFrame({
                'Rolling_Return': rolling_return,
                'Rolling_Volatility': rolling_vol,
                'Rolling_Sharpe': rolling_sharpe,
                'Rolling_Hit_Rate': rolling_hit_rate
            })
        
        return rolling_stats
    
    def get_operation_frequency(self, operations_df: pd.DataFrame) -> Dict:
        """
        Calcola frequenza operazioni per valuta e periodo
        
        Args:
            operations_df: DataFrame con signal activation
            
        Returns:
            Dict con statistiche frequenza operazioni
        """
        frequency_stats = {}
        
        for currency in operations_df.columns:
            operations = operations_df[currency]
            
            # Statistiche di base
            total_operations = operations.sum()
            total_days = len(operations)
            operation_frequency = total_operations / total_days if total_days > 0 else 0
            
            # Frequenza per mese
            monthly_ops = operations.resample('M').sum()
            avg_monthly_ops = monthly_ops.mean()
            
            # Frequenza per anno
            yearly_ops = operations.resample('Y').sum()
            avg_yearly_ops = yearly_ops.mean()
            
            frequency_stats[currency] = {
                'total_operations': total_operations,
                'total_days': total_days,
                'operation_frequency': operation_frequency,
                'avg_monthly_operations': avg_monthly_ops,
                'avg_yearly_operations': avg_yearly_ops,
                'monthly_operations': monthly_ops,
                'yearly_operations': yearly_ops
            }
        
        return frequency_stats
    
    def calculate_correlation_matrix(self, individual_pnl: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Calcola matrice di correlazione tra rendimenti delle valute
        
        Args:
            individual_pnl: Dict con P&L per ogni valuta
            
        Returns:
            DataFrame matrice di correlazione
        """
        # Crea DataFrame con tutti i P&L
        pnl_df = pd.DataFrame()
        
        for currency, pnl in individual_pnl.items():
            pnl_df[currency] = pnl
        
        # Calcola correlazioni
        correlation_matrix = pnl_df.corr()
        
        return correlation_matrix
    
    def get_comprehensive_analysis(self) -> Dict:
        """
        Esegue analisi completa e restituisce tutti i risultati
        
        Returns:
            Dict con tutti i risultati dell'analisi
        """
        # Carica dati
        data = self.load_portfolio_data()
        
        # Calcola operazioni
        operations = self.calculate_daily_operations()
        
        # Calcola P&L
        pnl_data = self.calculate_daily_pnl()
        
        # Metriche portfolio
        portfolio_metrics = self.calculate_performance_metrics(pnl_data['portfolio_pnl'])
        
        # Metriche individuali
        individual_metrics = {}
        for currency, pnl in pnl_data['individual_pnl'].items():
            individual_metrics[currency] = self.calculate_performance_metrics(pnl)
        
        # Attribution analysis
        attribution = self.calculate_attribution(pnl_data['currency_contributions'])
        
        # Rolling statistics
        rolling_stats = self.calculate_rolling_statistics(pnl_data['portfolio_pnl'])
        
        # Frequency analysis
        frequency_stats = self.get_operation_frequency(operations)
        
        # Correlation analysis
        correlation_matrix = self.calculate_correlation_matrix(pnl_data['individual_pnl'])
        
        return {
            'data': data,
            'operations': operations,
            'pnl_data': pnl_data,
            'portfolio_metrics': portfolio_metrics,
            'individual_metrics': individual_metrics,
            'attribution': attribution,
            'rolling_stats': rolling_stats,
            'frequency_stats': frequency_stats,
            'correlation_matrix': correlation_matrix
        }


# Funzioni di convenienza per uso diretto
def load_and_analyze(data_dir: str = "data/backtest_results") -> Dict:
    """
    Funzione di convenienza per caricare dati ed eseguire analisi completa
    
    Args:
        data_dir: Directory con i dati backtest
        
    Returns:
        Dict con tutti i risultati dell'analisi
    """
    analyzer = DailyOperationsAnalyzer(data_dir)
    return analyzer.get_comprehensive_analysis()


def calculate_benchmark_comparison(portfolio_pnl: pd.Series, 
                                 individual_equity: Dict[str, pd.DataFrame]) -> Dict:
    """
    Confronta performance del portfolio con equal weight benchmark
    
    Args:
        portfolio_pnl: P&L del portfolio risk parity
        individual_equity: Equity curves individuali
        
    Returns:
        Dict con confronto benchmark
    """
    # Crea equal weight benchmark
    equity_df = pd.DataFrame()
    for currency, equity_data in individual_equity.items():
        equity_df[currency] = equity_data.iloc[:, 0]  # Prima colonna
    
    # Equal weight returns
    equity_returns = equity_df.pct_change().fillna(0)
    equal_weight_returns = equity_returns.mean(axis=1)
    
    # Performance comparison
    analyzer_temp = DailyOperationsAnalyzer()
    portfolio_metrics = analyzer_temp.calculate_performance_metrics(portfolio_pnl)
    benchmark_metrics = analyzer_temp.calculate_performance_metrics(equal_weight_returns)
    
    return {
        'portfolio_metrics': portfolio_metrics,
        'benchmark_metrics': benchmark_metrics,
        'equal_weight_returns': equal_weight_returns,
        'outperformance': portfolio_metrics['annualized_return'] - benchmark_metrics['annualized_return']
    }