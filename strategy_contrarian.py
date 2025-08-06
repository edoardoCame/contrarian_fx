import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def strategy(data, timeframe='D', return_full_data=False):
    # Raggruppa per giorno e prendi l'ultimo prezzo, tieni solo le close, calcola i ritorni percentuali giornalieri
    prices = data.resample('D').last()
    prices = prices[['Close']]
    # Forward-fill missing values before calculating percent change to avoid deprecation warning
    returns = prices.ffill().pct_change(fill_method=None)

    # Strategia Contrarian - usa i ritorni percentuali, non i valori assoluti
    returns['strategy_returns'] = np.where(returns['Close'].shift(1) < 0, returns['Close'], 0)

    # Calcola i ritorni cumulativi della strategia
    cumulative_returns = (1 + returns['strategy_returns']).cumprod() - 1
    returns['Cumulative_Returns'] = cumulative_returns

    # Calcola i ritorni cumulativi del buy and hold
    cumulative_buy_and_hold = (1 + returns['Close']).cumprod() - 1
    returns['Cumulative_Buy_and_Hold'] = cumulative_buy_and_hold

    # Restituisci la serie dei ritorni cumulativi della strategia contrarian o tutti i dati
    if return_full_data:
        return returns
    else:
        return returns['Cumulative_Returns']



def rebalance_risk_parity(df_balance, n, threshold, shift):
    """ 
    Vengono escluse le strategie che non hanno un ritorno positivo negli ultimi n giorni.
    I pesi vengono assegnati in base alla proporzionalità inversa della volatilità, quindi il peso di ciascuna strategia sarà 1/sd"
    """
    """
    Vengono escluse le strategie che non hanno un ritorno positivo negli ultimi n giorni.
    Viene calcolato il ritorno percentuale per ogni valuta, e se il ritorno è negativo, viene impostato a zero.
    I pesi sono calcolati in base ai ritorni percentuali, e vengono normalizzati in modo che la somma dei pesi sia 1.
    La strategia viene ribilanciata ogni giorno, e i ritorni della strategia vengono calcolati come il prodotto dei pesi LAGGATI e dei ritorni percentuali delle valute.
    """
    df = df_balance.copy()
    colonne_originali = list(df.columns)
    n_valute = len(colonne_originali)

    # Prima sincronizziamo i prezzi con ffill
    df_ffill = df.ffill()

    # Calcoliamo i ritorni percentuali per ogni valuta SENZA fill_method
    for i in range(n_valute):
        df[f'ptc_change_valuta_{i+1}'] = df_ffill.iloc[:, i].pct_change(fill_method=None)

    # Calcola i ritorni degli ultimi n giorni, per ogni valuta (daily rebalancing)
    for i in range(n_valute):
        df[f'ritorni_valuta_{i+1}'] = df_ffill.iloc[:, i].pct_change(periods=n, fill_method=None)

    # Calcola la volatilità degli ultimi n giorni sui RITORNI, per ogni valuta (daily rebalancing)
    for i in range(n_valute):
        # Calcola volatilità sui ritorni percentuali, non sui prezzi
        df[f'volatilità_valuta_{i+1}'] = df[f'ptc_change_valuta_{i+1}'].rolling(window=n).std()
        
    """    Calcoliamo i pesi basati sui ritorni.  """
    # Calcoliamo i pesi basati sui ritorni
    # Calcola i pesi (non normalizzati)
    for i in range(n_valute):
        # Calcola il peso: se i ritorni sono maggiori della soglia E volatilità > 0, il peso è inversamente proporzionale alla volatilità, altrimenti è 0
        volatility = df[f'volatilità_valuta_{i+1}']
        # Evita divisione per zero: usa solo volatilità > 1e-8
        weight = np.where(
            (df[f'ritorni_valuta_{i+1}'] > threshold) & (volatility > 1e-8), 
            1 / volatility, 
            0
        )
        # Assegna il peso calcolato giornalmente (daily rebalancing)
        df[f'valuta_{i+1}_weight'] = weight

    # Normalizza i pesi in modo che la somma sia 1 (daily rebalancing)
    sum_weights = df[[f'valuta_{j+1}_weight' for j in range(n_valute)]].sum(axis=1)
    
    for i in range(n_valute):
        # Normalizza solo quando sum_weights > 1e-8, altrimenti imposta peso = 1/n_valute (equal weight)
        df[f'valuta_{i+1}_weight'] = np.where(
            sum_weights > 1e-8,
            df[f'valuta_{i+1}_weight'] / sum_weights,
            1.0 / n_valute  # Equal weight fallback
        )
        # Se il peso è NaN, imposta il peso a 0
        df[f'valuta_{i+1}_weight'] = np.where(df[f'valuta_{i+1}_weight'].isna(), 0, df[f'valuta_{i+1}_weight'])

    final_df = pd.DataFrame(index=df.index)

    # Creiamo un DataFrame finale solo con le colonne necessarie: ptc_change e weight per ogni valuta
    for i in range(n_valute):
        final_df[f'valuta_{i+1}_ptc_change'] = df[f'ptc_change_valuta_{i+1}']
    for i in range(n_valute):
        final_df[f'valuta_{i+1}_weight'] = df[f'valuta_{i+1}_weight']

    # Facciamo un forward fill dei pesi per le date mancanti dei pesi
    final_df = final_df.sort_index().ffill()

    # Calcoliamo i ritorni della strategia per ogni valuta come il prodotto dei pesi DEL GIORNO PRECEDENTE e dei ritorni percentuali delle valute
    for i in range(n_valute):
        final_df[f'ritorni_strategia_per_{i+1}'] = final_df[f'valuta_{i+1}_weight'].shift(shift) * final_df[f'valuta_{i+1}_ptc_change']

    # Calcoliamo i ritorni totali della strategia come la somma dei ritorni delle singole valute con i relativi pesi
    final_df['ritorni_strategia_totali'] = final_df[[f'ritorni_strategia_per_{i+1}' for i in range(n_valute)]].sum(axis=1)

    # Calcoliamo l'equity della strategia come il prodotto cumulativo dei ritorni totali
    final_df['equity'] = (1 + final_df['ritorni_strategia_totali']).cumprod()

    return final_df