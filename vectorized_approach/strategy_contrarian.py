import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_transaction_costs(strategy_returns, ticker, cost_per_trade_pips=0.5):
    """
    Calcola i costi di transazione basati sui segnali di trading per FOREX.
    
    Parameters:
    - strategy_returns: Serie pandas dei ritorni della strategia (0 quando non in trade)
    - ticker: Nome della coppia forex (es. 'USDJPY=X')
    - cost_per_trade_pips: Costo in pip per ogni entrata/uscita (default 0.5)
    
    Returns:
    - transaction_costs: Serie pandas dei costi di transazione per ogni trade
    """
    # Determina il valore del pip come frazione per il calcolo dei costi
    # 0.5 pip come frazione molto piccola dei ritorni giornalieri
    if any(jpy in ticker.upper() for jpy in ['JPY']):
        # Per le coppie JPY: 0.5 pip = frazione minima dei ritorni giornalieri
        pip_value = 0.00005  # 0.005% del valore (molto conservativo)
    else:
        # Per tutte le altre coppie: 0.5 pip = frazione minima
        pip_value = 0.00005  # 0.005% del valore (molto conservativo)
    
    # Identifica i segnali di trading (entrate/uscite)
    # Un trade inizia quando strategy_returns passa da 0 a != 0
    # Un trade finisce quando strategy_returns passa da != 0 a 0
    
    is_in_trade = (strategy_returns != 0)
    trade_starts = is_in_trade & ~is_in_trade.shift(1).fillna(False)  # Entrate
    trade_ends = ~is_in_trade & is_in_trade.shift(1).fillna(False)    # Uscite
    
    # Calcola i costi: 0.5 pip per entrata + 0.5 pip per uscita
    transaction_costs = pd.Series(0.0, index=strategy_returns.index)
    
    # Costo per entrate (0.5 pip)
    transaction_costs[trade_starts] = cost_per_trade_pips * pip_value
    
    # Costo per uscite (0.5 pip) 
    transaction_costs[trade_ends] = cost_per_trade_pips * pip_value
    
    return transaction_costs

def calculate_futures_transaction_costs_ibkr(strategy_returns, ticker, volume_tier=1):
    """
    Calcola i costi di transazione per FUTURES basati sulla struttura commissioni IBKR realistica.
    
    Parameters:
    - strategy_returns: Serie pandas dei ritorni della strategia (0 quando non in trade)
    - ticker: Nome del future commodity (es. 'CL=F', 'GC=F')  
    - volume_tier: Tier volumetrico IBKR (1-4, default 1 per retail)
                  1: ≤1000 contracts ($0.85 execution)
                  2: 1001-10000 contracts ($0.65 execution)
                  3: 10001-20000 contracts ($0.45 execution)  
                  4: >20000 contracts ($0.25 execution)
    
    Returns:
    - transaction_costs: Serie pandas dei costi di transazione per ogni trade
    """
    
    # IBKR Execution Fees per volume tier
    execution_fees = {
        1: 0.85,  # Retail trader
        2: 0.65,  # Active trader
        3: 0.45,  # High volume
        4: 0.25   # Institutional
    }
    
    # Exchange fees specifici per categoria di futures (basati su ricerca IBKR)
    exchange_fees = {
        # Energia (NYMEX)
        'CL=F': 1.38,  # Crude Oil WTI - più liquido
        'NG=F': 1.50,  # Natural Gas - volatile
        'BZ=F': 1.40,  # Brent Crude
        'RB=F': 1.45,  # RBOB Gasoline  
        'HO=F': 1.45,  # Heating Oil
        
        # Metalli Preziosi (COMEX)
        'GC=F': 1.25,  # Gold - molto liquido
        'SI=F': 1.30,  # Silver
        'PA=F': 1.35,  # Palladium
        
        # Metalli Industriali
        'HG=F': 1.20,  # Copper - molto liquido
        
        # Agricoltura (CBOT/CME)
        'ZC=F': 1.00,  # Corn - molto liquido
        'ZW=F': 1.05,  # Wheat
        'ZS=F': 1.00,  # Soybeans - molto liquido
        
        # Soft Commodities
        'SB=F': 1.15,  # Sugar
        'CT=F': 1.20,  # Cotton
        'CC=F': 1.25   # Cocoa
    }
    
    # Regulatory fees (tipici per futures)
    regulatory_fee = 0.02
    
    # Calcola commissione totale per contratto
    execution_fee = execution_fees.get(volume_tier, 0.85)
    exchange_fee = exchange_fees.get(ticker, 1.30)  # Default per futures non listati
    total_fee_per_contract = execution_fee + exchange_fee + regulatory_fee
    
    # Identifica i segnali di trading (entrate/uscite)
    is_in_trade = (strategy_returns != 0)
    trade_starts = is_in_trade & ~is_in_trade.shift(1).fillna(False)  # Entrate
    trade_ends = ~is_in_trade & is_in_trade.shift(1).fillna(False)    # Uscite
    
    # Per i futures, applichiamo il costo come frazione del movimento medio
    # Assumiamo che ogni trade rappresenti 1 contratto
    # Il costo viene espresso come frazione dei ritorni per compatibilità con il sistema esistente
    
    # Stima il valore nozionale medio del contratto per convertire $ in frazione
    # Basato sui prezzi tipici dei contratti (approssimazione conservativa)
    contract_values = {
        # Energia (valore nozionale approssimativo)
        'CL=F': 75000,    # 1000 barili * ~$75/barile
        'NG=F': 30000,    # 10000 MMBtu * ~$3/MMBtu
        'BZ=F': 80000,    # 1000 barili * ~$80/barile
        'RB=F': 60000,    # 42000 galloni * ~$1.40/gallon
        'HO=F': 65000,    # 42000 galloni * ~$1.55/gallon
        
        # Metalli Preziosi  
        'GC=F': 200000,   # 100 oz * ~$2000/oz
        'SI=F': 120000,   # 5000 oz * ~$24/oz
        'PA=F': 100000,   # 100 oz * ~$1000/oz
        
        # Metalli Industriali
        'HG=F': 90000,    # 25000 lbs * ~$3.60/lb
        
        # Agricoltura
        'ZC=F': 25000,    # 5000 bushels * ~$5/bushel
        'ZW=F': 30000,    # 5000 bushels * ~$6/bushel  
        'ZS=F': 70000,    # 5000 bushels * ~$14/bushel
        
        # Soft Commodities
        'SB=F': 25000,    # 112000 lbs * ~$0.22/lb
        'CT=F': 35000,    # 50000 lbs * ~$0.70/lb
        'CC=F': 30000     # 10 metric tons * ~$3000/ton
    }
    
    contract_value = contract_values.get(ticker, 50000)  # Default $50k nozionale
    cost_as_fraction = total_fee_per_contract / contract_value
    
    # Calcola i costi di transazione
    transaction_costs = pd.Series(0.0, index=strategy_returns.index)
    
    # Costo per entrate
    transaction_costs[trade_starts] = cost_as_fraction
    
    # Costo per uscite  
    transaction_costs[trade_ends] = cost_as_fraction
    
    return transaction_costs

def strategy(data, timeframe='D', return_full_data=False, apply_transaction_costs=True, instrument_type='forex', volume_tier=1):
    # Raggruppa per giorno e prendi l'ultimo prezzo, tieni solo le close, calcola i ritorni percentuali giornalieri
    prices = data.resample('D').last()
    prices = prices[['Close']]
    # Forward-fill missing values before calculating percent change to avoid deprecation warning
    returns = prices.ffill().pct_change(fill_method=None)

    # Strategia Contrarian - usa i ritorni percentuali, non i valori assoluti
    returns['strategy_returns'] = np.where(returns['Close'].shift(1) < 0, returns['Close'], 0)

    # Applica i costi di transazione se richiesto
    if apply_transaction_costs:
        # Estrai il ticker dal dataframe o usa un valore di default
        ticker = getattr(data, 'ticker', 'GENERIC')  # Fallback per ticker generico
        
        # Calcola i costi di transazione in base al tipo di strumento
        if instrument_type == 'futures':
            transaction_costs = calculate_futures_transaction_costs_ibkr(returns['strategy_returns'], ticker, volume_tier)
        else:  # forex (default)
            transaction_costs = calculate_transaction_costs(returns['strategy_returns'], ticker)
        
        # Sottrai i costi dai ritorni della strategia
        returns['strategy_returns_gross'] = returns['strategy_returns'].copy()
        returns['transaction_costs'] = transaction_costs
        returns['strategy_returns'] = returns['strategy_returns'] - transaction_costs

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