# Contrarian FX

> Analisi e backtest di una strategia contrarian multi-valuta sul Forex con Python e Jupyter Notebook

## 🚀 Overview
Questo progetto esplora una strategia di trading **contrarian** applicata a diverse coppie di valute Forex. L'obiettivo è testare la robustezza della strategia su più mercati, visualizzare i risultati e fornire una base per ulteriori sviluppi quantitativi.

## 📈 Caratteristiche principali
- **Backtest multi-asset**: applicazione della strategia su 18 coppie Forex dal 2010 a oggi
- **Visualizzazione equity line**: confronto grafico tra le performance delle valute
- **Notebook didattico**: spiegazione divulgativa della logica e dei risultati
- **Funzione risk parity**: esempio avanzato di gestione del rischio e ribilanciamento

## 🗂️ Struttura della repository
```
contrarian_fx/
├── main.ipynb                # Notebook principale: spiegazione, test e visualizzazione
├── strategy_contrarian.py    # Modulo Python con la strategia e funzioni avanzate
├── __pycache__/              # File compilati Python (ignorabili)
```

## ⚙️ Setup rapido
1. **Clona la repo**
   ```bash
   git clone https://github.com/edoardocamerinelli/contrarian_fx.git
   cd contrarian_fx
   ```
2. **Crea un ambiente virtuale (opzionale ma consigliato)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Installa le dipendenze**
   ```bash
   pip install yfinance pandas numpy matplotlib
   ```
4. **Avvia Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
5. **Apri `main.ipynb` e segui le istruzioni**

## 🧠 Logica della strategia
- **Contrarian**: se il rendimento del giorno precedente è negativo, si entra long il giorno dopo; altrimenti si resta flat.
- **Risk Parity (opzionale)**: ribilanciamento settimanale dei pesi sulle valute in base a rendimento e volatilità.

## 📊 Esempio di output
Il notebook produce un grafico con tutte le equity line delle valute, permettendo un confronto immediato delle performance.

## 🔬 Estensioni possibili
- Analisi su altri time frame o periodi storici
- Integrazione di filtri di volatilità o money management
- Confronto con strategie alternative (momentum, buy-and-hold, ecc.)

## 📚 Riferimenti
- [Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Jesse - Python Trading Framework](https://github.com/jesse-ai/jesse)
- [Zipline - Backtesting Library](https://github.com/quantopian/zipline)

## ⚠️ Disclaimer
Questo progetto ha scopo puramente didattico. Nessuna strategia garantisce profitti. Testa sempre a fondo prima di applicare in reale.

## 📝 Licenza
MIT License
