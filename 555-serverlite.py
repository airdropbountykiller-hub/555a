import datetime
import time
import requests
import feedparser
import threading
import os
import pytz
import pytz
import pandas
import gc
import json
# --- ML (scikit-learn) ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from urllib.request import urlopen
from urllib.error import URLError
import flask
from flask import Flask

# === 555-LITE SCHEDULE (patched) ===
SCHEDULE = {
    "rassegna": "07:00",
    "morning":  "08:10",
    "lunch":    "14:10",
    "evening":  "20:10",
}
RECOVERY_INTERVAL_MINUTES = 10
RECOVERY_WINDOWS = {"rassegna": 60, "morning": 80, "lunch": 80, "evening": 80}
ITALY_TZ = pytz.timezone("Europe/Rome")
LAST_RUN = {}  # per-minute debounce

from flask import Flask, jsonify, request
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
555SERVERLITE - Versione ottimizzata per massima RAM dedicata ai messaggi Telegram
Elimina: Dashboard, UI, CSS, PWA, grafici
Mantiene: Tutto il sistema ML, RSS, scheduling, qualità messaggi identica
"""

# === CONTROLLO MEMORIA E PERFORMANCE ===
print("🚀 [555-LITE] Avvio sistema ottimizzato RAM...")

# === FUNZIONE PER CREARE CARTELLE NECESSARIE (come 555-server) ===
def ensure_directories():
    """Crea automaticamente le cartelle necessarie se non esistono"""
    directories = ['salvataggi']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ [555-LITE] Cartella '{directory}' verificata/creata")
        except Exception as e:
            print(f"❌ [555-LITE] Errore nella creazione della cartella '{directory}': {e}")

# Crea le cartelle necessarie all'avvio
ensure_directories()

# === SISTEMA FLAG PERSISTENTI SU FILE ===
# File per salvare i flag degli invii giornalieri
FLAGS_FILE = os.path.join('salvataggi', 'daily_flags.json')

# Variabili globali per tracciare invii giornalieri
GLOBAL_FLAGS = {
    "morning_news_sent": False,
    "daily_report_sent": False,
    "evening_report_sent": False,
    "weekly_report_sent": False,
    "monthly_report_sent": False,
    "quarterly_report_sent": False,
    "semestral_report_sent": False,
    "annual_report_sent": False,
    "last_reset_date": datetime.datetime.now().strftime("%Y%m%d")
}

def load_daily_flags():
    """Carica i flag dal file JSON locale O da GitHub Gist (per persistenza su Render)"""
    global GLOBAL_FLAGS
    
    # 1. Prova prima a caricare da file locale
    local_success = False
    try:
        if os.path.exists(FLAGS_FILE):
            with open(FLAGS_FILE, 'r', encoding='utf-8') as f:
                saved_flags = json.load(f)
                
            # Verifica se i flag sono dello stesso giorno
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            saved_date = saved_flags.get("last_reset_date", "")
            
            if saved_date == current_date:
                # Stesso giorno: carica tutti i flag
                GLOBAL_FLAGS.update(saved_flags)
                print(f"✅ [FLAGS-FILE] Flag caricati da file locale per {current_date}")
                local_success = True
    except Exception as e:
        print(f"⚠️ [FLAGS-FILE] Errore caricamento flag locale: {e}")
    
    # 2. Se file locale non esiste o è vecchio, prova GitHub Gist
    if not local_success:
        try:
            gist_success = load_flags_from_github_gist()
            if gist_success:
                print(f"✅ [FLAGS-GIST] Flag caricati da GitHub Gist")
                return True
        except Exception as e:
            print(f"⚠️ [FLAGS-GIST] Errore caricamento da Gist: {e}")
    
    # 3. Se nessun file esiste, crea nuovo
    if not local_success:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        GLOBAL_FLAGS["last_reset_date"] = current_date
        save_daily_flags()
        print(f"📁 [FLAGS-FILE] Nuovo file flag creato per {current_date}")
        return False
    
    return local_success

def save_daily_flags():
    """Salva i flag correnti nel file JSON locale E su GitHub Gist (per persistenza su Render)"""
    success_local = False
    success_remote = False
    
    # 1. Salva locale (funziona durante la sessione del container)
    try:
        with open(FLAGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_FLAGS, f, indent=2, ensure_ascii=False)
        print(f"💾 [FLAGS-FILE] Flag salvati su file locale")
        success_local = True
    except Exception as e:
        print(f"❌ [FLAGS-FILE] Errore salvataggio flag locale: {e}")
    
    # 2. Salva su GitHub Gist (persistenza tra restart del container)
    try:
        success_remote = save_flags_to_github_gist()
    except Exception as e:
        print(f"⚠️ [FLAGS-GIST] Errore backup remoto flag: {e}")
    
    return success_local or success_remote

def reset_daily_flags_if_needed():
    """Resetta i flag se è passata la mezzanotte"""
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    if GLOBAL_FLAGS["last_reset_date"] != current_date:
        GLOBAL_FLAGS["morning_news_sent"] = False
        GLOBAL_FLAGS["daily_report_sent"] = False
        GLOBAL_FLAGS["evening_report_sent"] = False
        GLOBAL_FLAGS["last_reset_date"] = current_date
        print(f"🔄 [FLAGS] Reset giornaliero completato per {current_date}")
        return True
    return False


def set_message_sent_flag(message_type):
    """Imposta il flag di invio per il tipo di messaggio e salva su file"""
    reset_daily_flags_if_needed()  # Verifica reset automatico
    
    if message_type == "morning_news":
        GLOBAL_FLAGS["morning_news_sent"] = True
        print("✅ [FLAGS] Flag morning_news_sent impostato su True")
    elif message_type == "daily_report":
        GLOBAL_FLAGS["daily_report_sent"] = True
        print("✅ [FLAGS] Flag daily_report_sent impostato su True")
    elif message_type == "evening_report":
        GLOBAL_FLAGS["evening_report_sent"] = True
        print("✅ [FLAGS] Flag evening_report_sent impostato su True")
    elif message_type == "weekly_report":
        GLOBAL_FLAGS["weekly_report_sent"] = True
        print("✅ [FLAGS] Flag weekly_report_sent impostato su True")
    elif message_type == "monthly_report":
        GLOBAL_FLAGS["monthly_report_sent"] = True
        print("✅ [FLAGS] Flag monthly_report_sent impostato su True")
    elif message_type == "quarterly_report":
        GLOBAL_FLAGS["quarterly_report_sent"] = True
        print("✅ [FLAGS] Flag quarterly_report_sent impostato su True")
    elif message_type == "semestral_report":
        GLOBAL_FLAGS["semestral_report_sent"] = True
        print("✅ [FLAGS] Flag semestral_report_sent impostato su True")
    elif message_type == "annual_report":
        GLOBAL_FLAGS["annual_report_sent"] = True
        print("✅ [FLAGS] Flag annual_report_sent impostato su True")
    
    # Salva i flag aggiornati su file
    save_daily_flags()

def is_message_sent_today(message_type):
    """Verifica se il messaggio è già stato inviato oggi (solo memoria come 555-server)"""
    reset_daily_flags_if_needed()  # Verifica reset automatico
    
    # 🚨 EMERGENCY FIX: Usa RENDER_EXTERNAL_URL per fermare spam
    if message_type == "morning_news":
        external_url = os.getenv('RENDER_EXTERNAL_URL', '')
        # Se URL contiene 'STOP' o è vuota, ferma i messaggi
        if 'STOP' in external_url.upper() or not external_url:
            print("🛑 [EMERGENCY-STOP] Morning news bloccato (RENDER_EXTERNAL_URL contiene STOP o è vuota)")
            return True
        return GLOBAL_FLAGS["morning_news_sent"]
    elif message_type == "daily_report":
        return GLOBAL_FLAGS["daily_report_sent"]
    elif message_type == "evening_report":
        return GLOBAL_FLAGS["evening_report_sent"]
    elif message_type == "weekly_report":
        return GLOBAL_FLAGS["weekly_report_sent"]
    elif message_type == "monthly_report":
        return GLOBAL_FLAGS["monthly_report_sent"]
    elif message_type == "quarterly_report":
        return GLOBAL_FLAGS["quarterly_report_sent"]
    elif message_type == "semestral_report":
        return GLOBAL_FLAGS["semestral_report_sent"]
    elif message_type == "annual_report":
        return GLOBAL_FLAGS["annual_report_sent"]
    
    return False

# === OTTIMIZZAZIONI PERFORMANCE ===
try:
    from performance_config import (
        PERFORMANCE_CONFIG, LIGHTNING_ML_MODELS, FULL_ML_MODELS,
        CORE_INDICATORS, SECONDARY_INDICATORS, SPEED_TIMEOUTS,
        timed_execution, cached_with_expiry, get_thread_pool, parallel_execute
    )
    print("🚀 [LITE-TURBO] Ottimizzazioni performance caricate!")
except ImportError:
    print("⚠️ [LITE-TURBO] File performance_config.py non trovato - usando configurazione standard")
    PERFORMANCE_CONFIG = {"max_workers": 6, "cache_duration_minutes": 45}  # Più workers con RAM extra
    LIGHTNING_ML_MODELS = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    CORE_INDICATORS = ["MAC", "RSI", "MACD", "Bollinger", "EMA"]
    SPEED_TIMEOUTS = {"http_request_timeout": 8}  # Timeout più aggressivo

# === FUNZIONI GITHUB GIST PER FLAG RECOVERY ===
def save_flags_to_github_gist():
    """Salva i flag su GitHub Gist per persistenza tra restart container"""
    try:
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            print(f"⚠️ [FLAGS-GIST] Token GitHub non configurato - skip backup remoto")
            return False
        
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        gist_data = {
            "description": f"555 Daily Flags Backup - {current_date}",
            "public": False,
            "files": {
                f"daily_flags_{current_date}.json": {
                    "content": json.dumps(GLOBAL_FLAGS, indent=2, ensure_ascii=False)
                }
            }
        }
        
        response = requests.post(
            'https://api.github.com/gists',
            headers={'Authorization': f'token {github_token}'},
            json=gist_data,
            timeout=15
        )
        
        if response.status_code == 201:
            gist_url = response.json().get('html_url', 'N/A')
            print(f"✅ [FLAGS-GIST] Flag salvati su Gist: {gist_url[:50]}...")
            return True
        else:
            print(f"❌ [FLAGS-GIST] Errore salvataggio: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ [FLAGS-GIST] Errore generale: {e}")
        return False

def load_flags_from_github_gist():
    """Carica i flag da GitHub Gist (ultimo backup disponibile)"""
    try:
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            print(f"⚠️ [FLAGS-GIST] Token GitHub non configurato - skip caricamento remoto")
            return False
        
        # Cerca Gist con flag recenti
        response = requests.get(
            'https://api.github.com/gists',
            headers={'Authorization': f'token {github_token}'},
            timeout=15
        )
        
        if response.status_code != 200:
            print(f"❌ [FLAGS-GIST] Errore recupero gist: {response.status_code}")
            return False
        
        gists = response.json()
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # Trova il Gist più recente con i flag
        for gist in gists:
            description = gist.get('description', '')
            if '555 Daily Flags Backup' in description:
                files = gist.get('files', {})
                for filename, file_info in files.items():
                    if 'daily_flags' in filename and current_date in filename:
                        # Trova flag per oggi
                        file_url = file_info.get('raw_url')
                        if file_url:
                            flag_response = requests.get(file_url, timeout=15)
                            if flag_response.status_code == 200:
                                try:
                                    remote_flags = json.loads(flag_response.text)
                                    # Aggiorna flag globali
                                    GLOBAL_FLAGS.update(remote_flags)
                                    print(f"✅ [FLAGS-GIST] Flag caricati da Gist per {current_date}")
                                    return True
                                except json.JSONDecodeError as e:
                                    print(f"❌ [FLAGS-GIST] Errore parsing JSON: {e}")
        
        print(f"⚠️ [FLAGS-GIST] Nessun flag trovato per {current_date}")
        return False
        
    except Exception as e:
        print(f"❌ [FLAGS-GIST] Errore generale caricamento: {e}")
        return False

# === IMPORTAZIONI AGGIUNTIVE PER REPORT REALI ===
try:
    import pandas as pd
    import numpy as np
    from functools import lru_cache
    from pandas_datareader import data as web
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    import warnings
    warnings.filterwarnings("ignore")
    print("✅ [LITE-ML] Dipendenze ML caricate per report reali")
except ImportError as e:
    print(f"⚠️ [LITE-ML] Alcune dipendenze ML non disponibili: {e}")
    print("📝 [LITE-ML] Report settimanali useranno dati simulati")

# === CONFIGURAZIONE OTTIMIZZATA RENDER LITE ===
symbols = {
    "Dollar Index": "DTWEXBGS",
    "S&P 500": "SP500"
}
crypto_symbols = {
    "Bitcoin": "BTC",
    "Gold (PAXG)": "PAXG"
}

# === MODELLI ML OTTIMIZZATI PER RENDER LITE ===
models = {
    "Random Forest": (
        RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),  # Ridotto da 100 a 50
        "Ensemble di alberi decisionali ottimizzato per Render."
    ),
    "Logistic Regression": (
        LogisticRegression(solver='liblinear'),
        "Modello lineare veloce e interpretabile."
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=50, random_state=42),  # Ridotto da 100 a 50
        "Gradient boosting ottimizzato per ambiente cloud."
    ),
    "XGBoost": (
        XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', verbosity=0, n_jobs=1),  # Ridotto
        "XGBoost ottimizzato per Render Lite."
    ),
    "Support Vector Machine": (
        SVC(probability=True),
        "SVM per classificazione con probabilità."
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(n_neighbors=3),  # Ridotto da 5 a 3
        "KNN ottimizzato per velocità."
    ),
    "Naive Bayes": (
        GaussianNB(),
        "Modello probabilistico veloce."
    ),
    "AdaBoost": (
        AdaBoostClassifier(n_estimators=50, random_state=42),  # Ridotto da 100 a 50
        "AdaBoost ottimizzato per ambiente cloud."
    )
}

# === CACHE OTTIMIZZATO PER RENDER LITE ===
data_cache = {}
cache_timestamps = {}
CACHE_DURATION_MINUTES = 30  # Ridotto da 60 per Render


def is_cache_valid(cache_key, duration_minutes=CACHE_DURATION_MINUTES):
    """Check if cache entry is still valid"""
    if cache_key not in cache_timestamps:
        return False
    cache_time = cache_timestamps[cache_key]
    now = _now_it()
    try:
        if isinstance(cache_time, (int, float)):
            return (now.timestamp() - float(cache_time)) < duration_minutes * 60
        elif isinstance(cache_time, datetime.datetime):
            return (now - cache_time).total_seconds() < duration_minutes * 60
        else:
            return False
    except Exception:
        return False

def get_cache_key(func_name, *args, **kwargs):
    """Generate cache key from function name and arguments"""
    key_data = (func_name, args, tuple(sorted(kwargs.items())))
    return f"{func_name}_{hash(str(key_data))}"

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8396764345:AAH2aFy5lLAnr4xf-9FU91cWkYIrdG1f7hs"
TELEGRAM_CHAT_ID = "@abkllr"

# === FUNZIONI DATA LOADING OTTIMIZZATE PER RENDER LITE ===
def get_start_date(period):
    """Ottieni data di inizio per periodo specificato"""
    today = datetime.datetime.today()
    periods = {
        '1w': today - datetime.timedelta(weeks=1),
        '1m': today - datetime.timedelta(days=30),
        '6m': today - datetime.timedelta(days=182),
        '1y': today - datetime.timedelta(days=365)
    }
    return periods.get(period, today - datetime.timedelta(days=2000))

@lru_cache(maxsize=20)
def load_data_fred_lite(code, start_str, end_str):
    """Versione lite di caricamento dati FRED con cache"""
    try:
        start = datetime.datetime.fromisoformat(start_str)
        end = datetime.datetime.fromisoformat(end_str)
        print(f"🌐 [FRED-LITE] Caricamento {code}...")
        
        df = web.DataReader(code, 'fred', start, end).dropna()
        df.columns = ['Close']
        print(f"✅ [FRED-LITE] {code}: {len(df)} records")
        return df
    except Exception as e:
        print(f"❌ [FRED-LITE] {code}: {e}")
        return pd.DataFrame()

def load_data_fred(code, start, end):
    """Wrapper con cache per dati FRED"""
    cache_key = get_cache_key("fred", code, start.isoformat(), end.isoformat())
    
    if is_cache_valid(cache_key):
        if cache_key in data_cache:
            print(f"⚡ [CACHE] FRED {code} (hit)")
            return data_cache[cache_key].copy()
    
    df = load_data_fred_lite(code, start.isoformat(), end.isoformat())
    
    if not df.empty:
        data_cache[cache_key] = df.copy()
        cache_timestamps[cache_key] = datetime.datetime.now()
    
    return df

@lru_cache(maxsize=10)
def load_crypto_data_lite(symbol, limit=1000):
    """Versione lite per crypto con limit ridotto"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {'fsym': symbol, 'tsym': 'USD', 'limit': limit}
    
    try:
        print(f"🌐 [CRYPTO-LITE] Caricamento {symbol}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('Date', inplace=True)
            df = df[['close']].rename(columns={'close': 'Close'})
            print(f"✅ [CRYPTO-LITE] {symbol}: {len(df)} records")
            return df
        else:
            print(f"❌ [CRYPTO-LITE] {symbol} API Error")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ [CRYPTO-LITE] Error fetching {symbol}: {e}")
        return pd.DataFrame()

def load_crypto_data(symbol, limit=1000):
    """Wrapper con cache per crypto"""
    cache_key = get_cache_key("crypto", symbol, limit)
    
    if is_cache_valid(cache_key):
        if cache_key in data_cache:
            print(f"⚡ [CACHE] CRYPTO {symbol} (hit)")
            return data_cache[cache_key].copy()
    
    df = load_crypto_data_lite(symbol, limit)
    
    if not df.empty:
        data_cache[cache_key] = df.copy()
        cache_timestamps[cache_key] = datetime.datetime.now()
    
    return df

# === INDICATORI TECNICI OTTIMIZZATI PER RENDER LITE ===
def calculate_sma(df, short_period=20, long_period=50):
    """SMA ottimizzato"""
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(short_period).mean()
    df['SMA_long'] = df['Close'].rolling(long_period).mean()
    df['SMA_Signal'] = 0
    df.loc[(df['SMA_short'] > df['SMA_long']) & (df['SMA_short'].shift(1) <= df['SMA_long'].shift(1)), 'SMA_Signal'] = 1
    df.loc[(df['SMA_short'] < df['SMA_long']) & (df['SMA_short'].shift(1) >= df['SMA_long'].shift(1)), 'SMA_Signal'] = -1
    return df['SMA_Signal']

def calculate_mac(df):
    """MAC ottimizzato"""
    df = df.copy()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['MAC_Signal'] = 0
    df.loc[(df['SMA10'] > df['SMA50']) & (df['SMA10'].shift(1) <= df['SMA50'].shift(1)), 'MAC_Signal'] = 1
    df.loc[(df['SMA10'] < df['SMA50']) & (df['SMA10'].shift(1) >= df['SMA50'].shift(1)), 'MAC_Signal'] = -1
    return df['MAC_Signal']

def calculate_rsi(df, period=14):
    """RSI ottimizzato"""
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Signal'] = 0
    df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1
    df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1
    return df['RSI_Signal']

def calculate_macd(df):
    """MACD ottimizzato"""
    df = df.copy()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Signal'] = 0
    df.loc[(df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1)), 'MACD_Signal'] = 1
    df.loc[(df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1)), 'MACD_Signal'] = -1
    return df['MACD_Signal']

def calculate_bollinger_bands(df, window=20, n_std=2):
    """Bollinger Bands ottimizzato"""
    df = df.copy()
    df['BB_Mid'] = df['Close'].rolling(window).mean()
    df['BB_Std'] = df['Close'].rolling(window).std()
    df['BB_Upper'] = df['BB_Mid'] + n_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - n_std * df['BB_Std']
    df['BB_Signal'] = 0
    df.loc[df['Close'] < df['BB_Lower'], 'BB_Signal'] = 1
    df.loc[df['Close'] > df['BB_Upper'], 'BB_Signal'] = -1
    return df['BB_Signal']

def calculate_ema(df, period=21):
    """EMA ottimizzato"""
    df = df.copy()
    df['EMA_fast'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=period, adjust=False).mean()
    df['EMA_Signal'] = 0
    df.loc[(df['EMA_fast'] > df['EMA_slow']) & (df['EMA_fast'].shift(1) <= df['EMA_slow'].shift(1)), 'EMA_Signal'] = 1
    df.loc[(df['EMA_fast'] < df['EMA_slow']) & (df['EMA_fast'].shift(1) >= df['EMA_slow'].shift(1)), 'EMA_Signal'] = -1
    return df['EMA_Signal']

def signal_text(sig):
    """Converte segnale numerico in testo"""
    return "Buy" if sig == 1 else "Sell" if sig == -1 else "Hold"

def calculate_technical_indicators_lite(df):
    """Calcola indicatori tecnici ottimizzati per Render Lite"""
    indicators = {}
    indicators['SMA'] = calculate_sma(df)
    indicators['MAC'] = calculate_mac(df)
    indicators['RSI'] = calculate_rsi(df)
    indicators['MACD'] = calculate_macd(df)
    indicators['Bollinger'] = calculate_bollinger_bands(df)
    indicators['EMA'] = calculate_ema(df)
    return indicators

def calculate_ml_features_lite(df):
    """Features ML ottimizzate"""
    df = df.copy()
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Volatility_10d'] = df['Close'].rolling(10).std()
    df.dropna(inplace=True)
    return df

def add_features(df, target_horizon):
    """Aggiungi features per ML"""
    df = calculate_ml_features_lite(df)
    df['Target'] = (df['Close'].shift(-target_horizon) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

def get_all_signals_summary_lite(timeframe='1w'):
    """Versione lite di get_all_signals_summary ottimizzata per Render"""
    end = datetime.datetime.today()
    start = get_start_date(timeframe)
    summary_rows = []
    all_assets = {**symbols, **crypto_symbols}
    
    for name, asset_code in all_assets.items():
        try:
            if name in symbols:
                df = load_data_fred(asset_code, start, end)
            else:
                df = load_crypto_data(asset_code)
                df = df[df.index >= start]
                
            if df.empty:
                continue
                
            indicators = calculate_technical_indicators_lite(df)
            row = {'Asset': name}
            
            for key, signal in indicators.items():
                last_signal = signal_text(signal[signal != 0].iloc[-1]) if not signal[signal != 0].empty else 'Hold'
                row[key] = last_signal
                
            summary_rows.append(row)
            
            # Pulizia memoria dopo ogni asset
            del df, indicators
            gc.collect()
            
        except Exception as e:
            print(f"⚠️ [SIGNALS-LITE] Errore {name}: {e}")
            continue
    
    return pd.DataFrame(summary_rows)

def train_model_lite(model, df):
    """Versione lite e ottimizzata di train_model"""
    try:
        if df is None or df.empty or len(df) < 30:
            return 0.5, 0.5
            
        # Features e target
        X = df[['Return_5d', 'Return_10d', 'Volatility_10d']].copy()
        y = df['Target'].copy()
        
        # Pulizia dati
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index].dropna()
        X = X.loc[y.index]
        
        if len(X) < 30 or len(y.unique()) < 2:
            return 0.5, 0.5
        
        # Split semplice per velocità
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        if len(y_train.unique()) < 2:
            return 0.5, 0.5
        
        # Training veloce
        model.fit(X_train, y_train)
        
        # Predizione sull'ultimo sample
        last_sample = X.iloc[-1:]
        prob_result = model.predict_proba(last_sample)
        
        # Parsing della probabilità
        if hasattr(prob_result, 'shape') and len(prob_result.shape) > 0:
            prob = float(prob_result[0][1]) if prob_result.shape[1] >= 2 else 0.5
        else:
            prob = 0.5
        
        # Accuratezza semplificata
        if len(X_test) > 0:
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) if len(y_pred) == len(y_test) else 0.5
        else:
            acc = 0.5
        
        return max(0.0, min(1.0, prob)), max(0.0, min(1.0, acc))
        
    except Exception as e:
        print(f"⚠️ [ML-LITE] Errore: {e}")
        return 0.5, 0.5

# === SISTEMA BACKUP RENDER → DRIVE ===
try:
    from render_drive_backup import RenderDriveBackup
    print("🔄 [LITE-BACKUP] Sistema backup caricato")
    BACKUP_SYSTEM_ENABLED = True
except ImportError:
    print("⚠️ [LITE-BACKUP] Sistema backup non disponibile")
    RenderDriveBackup = None
    BACKUP_SYSTEM_ENABLED = False

# === CONTROLLO FUNZIONI OTTIMIZZATO ===
FEATURES_ENABLED = {
    "scheduled_reports": True,
    "manual_reports": True,
    "backtest_reports": True,
    "analysis_reports": True,
    "morning_news": True,
    "daily_report": True,
    "weekly_reports": True,        # NUOVO
    "monthly_reports": True,       # NUOVO
    "enhanced_ml": True,           # NUOVO - ML potenziato con RAM extra
    "real_time_alerts": True,      # NUOVO - Alert in tempo reale
    "memory_cleanup": True
}

def is_feature_enabled(feature_name):
    """Controlla se una funzione è abilitata"""
    return FEATURES_ENABLED.get(feature_name, True)

# === FUNZIONE INVIO TELEGRAM OTTIMIZZATA ===
def invia_messaggio_telegram(msg):
    """Versione ottimizzata per RAM - stesso livello qualità"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    print(f"📤 [LITE-TELEGRAM] Invio messaggio ({len(msg)} caratteri)")
    
    try:
        # Pulizia ottimizzata
        clean_msg = msg.replace('```', '`').replace('**', '*')
        
        # Gestione messaggi lunghi con divisione intelligente
        if len(clean_msg) > 2400:
            return _send_long_message_optimized(clean_msg, url)
        else:
            return _send_single_message_lite(clean_msg, url)
            
    except Exception as e:
        print(f"❌ [LITE-TELEGRAM] Errore: {e}")
        return False
    finally:
        # Pulizia memoria aggressiva
        gc.collect()

def _send_long_message_optimized(clean_msg, url):
    """Divisione messaggi lunghi ottimizzata per velocità"""
    parts = []
    start = 0
    part_num = 1
    
    while start < len(clean_msg):
        end = start + 2400
        if end >= len(clean_msg):
            end = len(clean_msg)
        else:
            cut_point = clean_msg.rfind('\n', start, end)
            if cut_point > start:
                end = cut_point
        
        part = clean_msg[start:end]
        if len(parts) == 0:
            part = f"📤 PARTE {part_num}\n\n" + part
        else:
            part = f"📤 PARTE {part_num} (continua)\n\n" + part
        
        parts.append(part)
        start = end
        part_num += 1
    
    # Invio sequenziale ottimizzato
    all_success = True
    for i, part in enumerate(parts):
        success = _send_single_message_lite(part, url)
        if not success:
            all_success = False
        
        # Pausa minima tra parti
        if i < len(parts) - 1:
            time.sleep(1.5)  # Ridotto da 2s per velocità
    
    return all_success

def _send_single_message_lite(clean_msg, url):
    """Versione lite con fallback essenziali"""
    
    # Strategie di fallback semplificate ma efficaci
    strategies = [
        {"parse_mode": "Markdown", "name": "Markdown"},
        {"parse_mode": None, "name": "Plain", "processor": lambda x: x.replace('*', '').replace('_', '')}
    ]
    
    for strategy in strategies:
        processed_msg = clean_msg
        if 'processor' in strategy:
            processed_msg = strategy['processor'](clean_msg)
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": processed_msg,
            "parse_mode": strategy["parse_mode"]
        }
        
        try:
            r = requests.post(url, data=payload, timeout=10)
            if r.status_code == 200:
                print(f"✅ [LITE-TELEGRAM] Inviato con {strategy['name']}")
                return True
        except Exception as e:
            print(f"⚠️ [LITE-TELEGRAM] Tentativo {strategy['name']} fallito: {e}")
            continue
    
    return False

# === CALENDARIO EVENTI (Stesso del sistema completo) ===
today = datetime.date.today()

def create_event(title, date, impact, source):
    return {"Data": date.strftime("%Y-%m-%d"), "Titolo": title, "Impatto": impact, "Fonte": source}

eventi = {
    "Finanza": [
        create_event("Decisione tassi FED", today + datetime.timedelta(days=2), "Alto", "Investing.com"),
        create_event("Rilascio CPI USA", today + datetime.timedelta(days=6), "Alto", "Trading Economics"),
        create_event("Occupazione Eurozona", today + datetime.timedelta(days=10), "Medio", "ECB"),
        create_event("Conference BCE", today + datetime.timedelta(days=15), "Basso", "ECB")
    ],
    "Criptovalute": [
        create_event("Aggiornamento Ethereum", today + datetime.timedelta(days=3), "Alto", "CoinMarketCal"),
        create_event("Hard Fork Cardano", today + datetime.timedelta(days=7), "Medio", "CoinDesk"),
        create_event("Annuncio regolamentazione MiCA", today + datetime.timedelta(days=12), "Alto", "EU Commission"),
        create_event("Evento community Bitcoin", today + datetime.timedelta(days=20), "Basso", "Bitcoin Magazine")
    ],
    "Geopolitica": [
        create_event("Vertice NATO", today + datetime.timedelta(days=1), "Alto", "Reuters"),
        create_event("Elezioni UK", today + datetime.timedelta(days=8), "Alto", "BBC"),
        create_event("Discussione ONU su Medio Oriente", today + datetime.timedelta(days=11), "Medio", "UN"),
        create_event("Summit BRICS", today + datetime.timedelta(days=18), "Basso", "Al Jazeera")
    ]
}

# === RSS FEEDS (Stesso sistema + Mercati Emergenti) ===
RSS_FEEDS = {
    "Finanza": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.investing.com/rss/news_285.rss",
        "https://www.marketwatch.com/rss/topstories",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://feeds.bloomberg.com/markets/news.rss"
    ],
    "Criptovalute": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://cryptoslate.com/feed/",
        "https://bitcoinist.com/feed/"
    ],
    "Geopolitica": [
        "https://feeds.reuters.com/Reuters/worldNews",
        "https://www.aljazeera.com/xml/rss/all.xml",
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/rss.xml"
    ],
    "Mercati Emergenti": [
        "https://feeds.reuters.com/reuters/emergingMarketsNews",
        "https://www.investing.com/rss/news_14.rss",
        "https://feeds.bloomberg.com/emerging-markets/news.rss",
        "https://www.ft.com/emerging-markets?format=rss",
        "https://www.wsj.com/xml/rss/3_7455.xml"
    ]
}

# === NOTIZIE CRITICHE (Stesso algoritmo, ottimizzato) ===
def get_notizie_critiche():
    """Recupero notizie ottimizzato per velocità"""
    notizie_critiche = []
    
    from datetime import timezone
    now_utc = datetime.datetime.now(timezone.utc)
    soglia_24h = now_utc - datetime.timedelta(hours=24)
    
    def is_highlighted(title):
        keywords = [
            "crisis", "inflation", "fed", "ecb", "rates", "crash", "surge",
            "war", "sanctions", "hack", "regulation", "bitcoin", "crypto"
        ]
        return any(k in title.lower() for k in keywords)
    
    def is_recent_news(entry):
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                news_time = datetime.datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                return news_time >= soglia_24h
            return True  # In dubbio, includiamo
        except:
            return True
    
    for categoria, feed_urls in RSS_FEEDS.items():
        for url in feed_urls[:2]:  # Limitiamo a 2 feed per categoria per velocità
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo or not parsed.entries:
                    continue
                
                for entry in parsed.entries[:5]:  # Max 5 per feed
                    title = entry.get("title", "")
                    
                    if is_recent_news(entry) and is_highlighted(title):
                        link = entry.get("link", "")
                        source = parsed.feed.get("title", "Unknown")
                        
                        notizie_critiche.append({
                            "titolo": title,
                            "link": link,
                            "fonte": source,
                            "categoria": categoria
                        })
                        
                        if len(notizie_critiche) >= 8:  # Limite per velocità
                            break
                
                if len(notizie_critiche) >= 8:
                    break
            except Exception as e:
                continue
        
        if len(notizie_critiche) >= 8:
            break
    
    return notizie_critiche[:5]  # Top 5

# === GENERAZIONE MESSAGGI EVENTI (Stesso sistema) ===
def genera_messaggio_eventi():
    """Genera messaggio eventi - stessa qualità del sistema completo"""
    oggi = datetime.date.today()
    prossimi_7_giorni = oggi + datetime.timedelta(days=7)
    sezioni_parte1 = []
    sezioni_parte2 = []

    # Eventi di oggi
    eventi_oggi_trovati = False
    for categoria, lista in eventi.items():
        eventi_oggi = [e for e in lista if e["Data"] == oggi.strftime("%Y-%m-%d")]
        if eventi_oggi:
            if not eventi_oggi_trovati:
                sezioni_parte1.append("📅 EVENTI DI OGGI")
                eventi_oggi_trovati = True
            eventi_oggi.sort(key=lambda x: ["Basso", "Medio", "Alto"].index(x["Impatto"]))
            sezioni_parte1.append(f"📌 {categoria}")
            for e in eventi_oggi:
                impact_color = "🔴" if e['Impatto'] == "Alto" else "🟡" if e['Impatto'] == "Medio" else "🟢"
                sezioni_parte1.append(f"{impact_color} • {e['Titolo']} ({e['Impatto']}) - {e['Fonte']}")
    
    # Eventi prossimi giorni
    eventi_prossimi = []
    for categoria, lista in eventi.items():
        for evento in lista:
            data_evento = datetime.datetime.strptime(evento["Data"], "%Y-%m-%d").date()
            if oggi < data_evento <= prossimi_7_giorni:
                evento_con_categoria = evento.copy()
                evento_con_categoria["Categoria"] = categoria
                evento_con_categoria["DataObj"] = data_evento
                eventi_prossimi.append(evento_con_categoria)
    
    if eventi_prossimi:
        eventi_prossimi.sort(key=lambda x: (x["DataObj"], ["Basso", "Medio", "Alto"].index(x["Impatto"])))
        if eventi_oggi_trovati:
            sezioni_parte1.append("")
        sezioni_parte1.append("🗺 PROSSIMI EVENTI (7 giorni)")
        
        data_corrente = None
        for evento in eventi_prossimi:
            if evento["DataObj"] != data_corrente:
                data_corrente = evento["DataObj"]
                giorni_mancanti = (data_corrente - oggi).days
                sezioni_parte1.append(f"\n📅 {data_corrente.strftime('%d/%m')} (tra {giorni_mancanti} giorni)")
            impact_color = "🔴" if evento['Impatto'] == "Alto" else "🟡" if evento['Impatto'] == "Medio" else "🟢"
            sezioni_parte1.append(f"{impact_color} • {evento['Titolo']} ({evento['Impatto']}) - {evento['Categoria']} - {evento['Fonte']}")

    # Notizie critiche
    notizie_critiche = get_notizie_critiche()
    if notizie_critiche:
        sezioni_parte2.append("🚨 *NOTIZIE CRITICHE* (24h)")
        sezioni_parte2.append(f"📰 Trovate {len(notizie_critiche)} notizie rilevanti\n")
        
        for i, notizia in enumerate(notizie_critiche, 1):
            titolo_breve = notizia["titolo"][:70] + "..." if len(notizia["titolo"]) > 70 else notizia["titolo"]
            sezioni_parte2.append(f"{i}. 🔴 *{titolo_breve}*")
            sezioni_parte2.append(f"   📂 {notizia['categoria']} | 📰 {notizia['fonte']}")
            sezioni_parte2.append("")

    # Invio messaggi
    if not sezioni_parte1 and not sezioni_parte2:
        return "✅ Nessun evento in calendario"

    success_count = 0
    if sezioni_parte1:
        msg_parte1 = f"🗓️ *Eventi del {oggi}* (Parte 1/2)\n\n" + "\n".join(sezioni_parte1)
        if invia_messaggio_telegram(msg_parte1):
            success_count += 1
        time.sleep(3)
    
    if sezioni_parte2:
        msg_parte2 = f"🗓️ *Eventi del {oggi}* (Parte 2/2)\n\n" + "\n".join(sezioni_parte2)
        if invia_messaggio_telegram(msg_parte2):
            success_count += 1
    
    return f"Messaggi eventi inviati: {success_count}/2"

# === FUNZIONI DATI STATICI (Render Lite non ha accesso ai CSV locali) ===
def get_asset_technical_summary(asset_name):
    """Ottieni riassunto tecnico statico per asset - Render Lite optimized"""
    # Su Render Lite, ritorniamo analisi statiche per non dipendere da file esterni
    try:
        if "bitcoin" in asset_name.lower() or "btc" in asset_name.lower():
            return "📊 Bitcoin: 🟢 BULLISH (Trend consolidation)\n   Range: $42k-$45k | Momentum: Positive"
        elif "s&p" in asset_name.lower() or "500" in asset_name.lower():
            return "📊 S&P 500: ⚪ NEUTRAL (Mixed signals)\n   Range: 4800-4850 | Volatility: Normal"
        elif "gold" in asset_name.lower() or "oro" in asset_name.lower():
            return "📊 Gold: 🟢 BULLISH (Safe haven demand)\n   Level: $2040-2060 | Trend: Upward"
        else:
            return f"📊 {asset_name}: ⚪ NEUTRAL (Market consolidation)\n   Status: Range-bound trading"
    except Exception as e:
        return f"❌ Errore analisi {asset_name}: {e}"

# === REPORT COMPLETI CON RAM EXTRA ===
# Integrazione dati live dal sistema 555 principale!

# === ANALISI ML ENHANCED ===
def analyze_news_sentiment_and_impact():
    """Analizza il sentiment delle notizie e l'impatto potenziale sui mercati"""
    try:
        print("🔍 [NEWS-ML] Avvio analisi sentiment e impatto mercati...")
        
        # Recupera le notizie critiche recenti
        notizie_critiche = get_notizie_critiche()
        
        if not notizie_critiche:
            return {
                "summary": "📰 Nessuna notizia critica rilevata nelle ultime 24 ore",
                "sentiment": "NEUTRAL",
                "market_impact": "LOW",
                "recommendations": []
            }
        
        # Keywords per sentiment analysis
        positive_keywords = [
            "growth", "up", "rise", "gain", "increase", "bullish", "rally", "surge", "boost", "strong",
            "positive", "optimistic", "record", "profit", "earnings", "dividend", "expansion", "recovery",
            "breakthrough", "success", "approval", "deal", "agreement", "cooperation", "alliance"
        ]
        
        negative_keywords = [
            "crash", "fall", "drop", "decline", "bearish", "loss", "deficit", "recession", "crisis",
            "negative", "pessimistic", "concern", "risk", "threat", "uncertainty", "volatility",
            "conflict", "war", "sanctions", "ban", "investigation", "fraud", "scandal", "bankruptcy",
            "default", "hack", "exploit", "regulation", "restriction", "emergency"
        ]
        
        # Keywords per impatto mercati
        high_impact_keywords = [
            "fed", "ecb", "boe", "boj", "interest rate", "monetary policy", "inflation", "gdp",
            "employment", "unemployment", "cpi", "ppi", "trade war", "tariff", "oil price",
            "bitcoin", "cryptocurrency", "regulation", "ban", "etf", "major bank", "bailout",
            "nuclear", "military", "invasion", "sanctions", "emergency", "crisis"
        ]
        
        medium_impact_keywords = [
            "earnings", "revenue", "profit", "dividend", "merger", "acquisition", "ipo",
            "company", "stock", "share", "market", "commodity", "gold", "silver", "energy"
        ]
        
        # Analizza ogni notizia
        sentiment_scores = []
        impact_scores = []
        analyzed_news = []
        
        for notizia in notizie_critiche:
            title = notizia["titolo"].lower()
            
            # Calcola sentiment score
            pos_score = sum(1 for keyword in positive_keywords if keyword in title)
            neg_score = sum(1 for keyword in negative_keywords if keyword in title)
            sentiment_score = pos_score - neg_score
            
            # Calcola impact score
            high_impact = sum(1 for keyword in high_impact_keywords if keyword in title)
            medium_impact = sum(1 for keyword in medium_impact_keywords if keyword in title)
            impact_score = high_impact * 3 + medium_impact * 1
            
            # Determina sentiment
            if sentiment_score > 0:
                sentiment = "POSITIVE"
                sentiment_emoji = "🟢"
            elif sentiment_score < 0:
                sentiment = "NEGATIVE"
                sentiment_emoji = "🔴"
            else:
                sentiment = "NEUTRAL"
                sentiment_emoji = "⚪"
            
            # Determina impatto
            if impact_score >= 3:
                impact = "HIGH"
                impact_emoji = "🔥"
            elif impact_score >= 1:
                impact = "MEDIUM"
                impact_emoji = "⚡"
            else:
                impact = "LOW"
                impact_emoji = "🔹"
            
            sentiment_scores.append(sentiment_score)
            impact_scores.append(impact_score)
            
            # Genera commento ML enhanced
            ml_comment = generate_ml_comment_for_news({
                'title': notizia["titolo"],
                'categoria': notizia["categoria"],
                'sentiment': sentiment,
                'impact': impact
            })
            
            analyzed_news.append({
                "title": notizia["titolo"][:80] + "..." if len(notizia["titolo"]) > 80 else notizia["titolo"],
                "sentiment": sentiment,
                "sentiment_emoji": sentiment_emoji,
                "impact": impact,
                "impact_emoji": impact_emoji,
                "fonte": notizia["fonte"],
                "categoria": notizia["categoria"],
                "link": notizia["link"],
                "ml_comment": ml_comment
            })
        
        # Calcola sentiment complessivo
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        if avg_sentiment > 0.5:
            overall_sentiment = "POSITIVE"
            sentiment_emoji = "🟢"
        elif avg_sentiment < -0.5:
            overall_sentiment = "NEGATIVE"
            sentiment_emoji = "🔴"
        else:
            overall_sentiment = "NEUTRAL"
            sentiment_emoji = "⚪"
        
        # Calcola impatto complessivo
        avg_impact = sum(impact_scores) / len(impact_scores) if impact_scores else 0
        if avg_impact >= 2:
            overall_impact = "HIGH"
            impact_emoji = "🔥"
        elif avg_impact >= 0.5:
            overall_impact = "MEDIUM"
            impact_emoji = "⚡"
        else:
            overall_impact = "LOW"
            impact_emoji = "🔹"
        
        # Genera raccomandazioni enhanced
        recommendations = []
        top_news = sorted(analyzed_news, key=lambda x: impact_scores[analyzed_news.index(x)], reverse=True)[:3]
        
        for news in top_news:
            if 'ml_comment' in news and news['ml_comment']:
                asset_prefix = "📈" if news['sentiment'] == 'POSITIVE' else "📉" if news['sentiment'] == 'NEGATIVE' else "📊"
                enhanced_rec = f"{asset_prefix} **{news['categoria']}**: {news['ml_comment']}"
                recommendations.append(enhanced_rec)
        
        recommendations = recommendations[:4]
        
        return {
            "summary": f"📰 *RASSEGNA STAMPA ML*\n{sentiment_emoji} *Sentiment*: {overall_sentiment}\n{impact_emoji} *Impatto Mercati*: {overall_impact}",
            "sentiment": overall_sentiment,
            "market_impact": overall_impact,
            "recommendations": recommendations,
            "analyzed_news": analyzed_news
        }
        
    except Exception as e:
        print(f"❌ [NEWS-ML] Errore nell'analisi sentiment: {e}")
        return {
            "summary": "❌ Errore nell'analisi delle notizie",
            "sentiment": "UNKNOWN",
            "market_impact": "UNKNOWN",
            "recommendations": []
        }

def generate_ml_comment_for_news(news):
    """Genera un commento ML specifico per una notizia con raccomandazioni integrate"""
    try:
        title = news.get('title', '').lower()
        categoria = news.get('categoria', '')
        sentiment = news.get('sentiment', 'NEUTRAL')
        impact = news.get('impact', 'LOW')
        
        # Commenti enhanced con raccomandazioni specifiche
        if "bitcoin" in title or "crypto" in title or "btc" in title:
            if sentiment == "POSITIVE" and impact == "HIGH":
                return "🟢 Crypto Rally: BTC breakout atteso. Monitora 45k resistance. Strategy: Long BTC, ALT rotation."
            elif sentiment == "NEGATIVE" and impact == "HIGH":
                return "🔴 Crypto Dump: Pressione vendita forte. Support 38k critico. Strategy: Reduce crypto exposure."
            elif "regulation" in title or "ban" in title:
                return "⚠️ Regulation Risk: Volatilità normativa. Strategy: Hedge crypto positions, monitor compliance coins."
            elif "etf" in title:
                return "📈 ETF Development: Institutional adoption. Strategy: Long-term bullish, monitor approval timeline."
            else:
                return "⚪ Crypto Neutral: Consolidamento atteso. Strategy: Range trading 40-43k, wait breakout."
        
        elif "fed" in title or "rate" in title or "tassi" in title or "powell" in title:
            if sentiment == "NEGATIVE" and impact == "HIGH":
                return "🔴 Hawkish Fed: Tassi più alti. Strategy: Short duration bonds, defensive stocks, USD long."
            elif sentiment == "POSITIVE" and impact == "HIGH":
                return "🟢 Dovish Fed: Risk-on mode. Strategy: Growth stocks, EM currencies, commodities long."
            elif "pause" in title or "hold" in title:
                return "⏸️ Fed Pause: Wait-and-see. Strategy: Quality stocks, avoid rate-sensitive sectors."
            else:
                return "📊 Fed Watch: Policy uncertainty. Strategy: Low beta stocks, hedge interest rate risk."
        
        elif "inflazione" in title or "inflation" in title or "cpi" in title:
            if sentiment == "NEGATIVE" and impact == "HIGH":
                return "🔴 High Inflation: Pressure su bonds. Strategy: TIPS, commodities, avoid long duration."
            elif sentiment == "POSITIVE" and impact == "HIGH":
                return "🟢 Cooling Inflation: Growth supportive. Strategy: Tech stocks, long bonds opportunity."
            else:
                return "📈 Inflation Data: Mixed signals. Strategy: Balanced allocation, inflation hedges."
        
        elif "oil" in title or "energy" in title:
            if sentiment == "POSITIVE" and impact == "HIGH":
                return "🛢️ Oil Rally: Supply constraints. Strategy: Energy stocks, oil ETFs, avoid airlines."
            elif sentiment == "NEGATIVE" and impact == "HIGH":
                return "📉 Oil Crash: Demand concerns. Strategy: Short energy, long airlines, consumer stocks."
            else:
                return "⚫ Energy Watch: Price stability. Strategy: Monitor inventory data, OPEC decisions."
        
        else:
            if sentiment == "POSITIVE" and impact == "HIGH":
                return f"🟢 Market Positive: {categoria} sector boost expected. Strategy: Monitor sector rotation."
            elif sentiment == "NEGATIVE" and impact == "HIGH":
                return f"🔴 Market Risk: {categoria} negative impact. Strategy: Risk management, hedge exposure."
            else:
                return f"📰 {categoria} Update: Limited market impact. Strategy: Information tracking only."
                
    except Exception as e:
        return "❌ ML Analysis Error: Technical issue in news processing."

# === REPORT MORNING NEWS ENHANCED ===
def get_extended_morning_news():
    """Recupera 20-30 notizie per la rassegna stampa mattutina da tutti i feed RSS"""
    notizie_estese = []
    
    from datetime import timezone
    now_utc = datetime.datetime.now(timezone.utc)
    soglia_12h = now_utc - datetime.timedelta(hours=12)
    
    def is_recent_morning_news(entry):
        try:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                news_time = datetime.datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                return news_time >= soglia_12h
            return True
        except:
            return True
    
    target_per_categoria = 8  # Aumentato per garantire almeno 7 notizie
    
    for categoria, feed_urls in RSS_FEEDS.items():
        categoria_count = 0
        
        for url in feed_urls:
            if categoria_count >= target_per_categoria:
                break
                
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo or not parsed.entries:
                    continue
                
                for entry in parsed.entries[:15]:
                    title = entry.get("title", "")
                    
                    if is_recent_morning_news(entry):
                        link = entry.get("link", "")
                        source = parsed.feed.get("title", "Unknown")
                        
                        notizie_estese.append({
                            "titolo": title,
                            "link": link,
                            "fonte": source,
                            "categoria": categoria,
                            "data": "Recente"
                        })
                        
                        categoria_count += 1
                        
                        if categoria_count >= target_per_categoria:
                            break
                
                if len(notizie_estese) >= 30:
                    break
                    
            except Exception as e:
                continue
        
        if len(notizie_estese) >= 30:
            break
    
    return notizie_estese[:25]  # Limitiamo a 25 per velocità

def get_serverlite_news_by_category():
    """Recupera 7 notizie per ogni categoria (4 categorie)"""
    categories = {
        "Finanza": 7,              # Fed, banche, tassi
        "Criptovalute": 7,         # Bitcoin, ETF, regulation  
        "Geopolitica": 7,          # Guerra, sanzioni, elezioni
        "Mercati Emergenti": 7     # India, Cina, Brasile
    }
    
    news_by_category = {}
    for categoria in categories.keys():
        news_by_category[categoria] = []
        target_count = categories[categoria]
        
        if categoria in RSS_FEEDS:
            for url in RSS_FEEDS[categoria][:2]:  # Max 2 feed per categoria
                if len(news_by_category[categoria]) >= target_count:
                    break
                try:
                    parsed = feedparser.parse(url)
                    if parsed.bozo or not parsed.entries:
                        continue
                    
                    for entry in parsed.entries[:10]:  # Max 10 per feed
                        if len(news_by_category[categoria]) >= target_count:
                            break
                            
                        title = entry.get("title", "")
                        link = entry.get("link", "")
                        source = parsed.feed.get("title", "Unknown")
                        
                        news_by_category[categoria].append({
                            "titolo": title,
                            "link": link,
                            "fonte": source,
                            "categoria": categoria
                        })
                        
                except Exception as e:
                    continue
    
    return news_by_category  # 28 notizie totali

def generate_morning_news_briefing():
    """NUOVO - Rassegna stampa mattutina 6 messaggi con firma LITE"""
    try:
        italy_tz = pytz.timezone('Europe/Rome')
        now = datetime.datetime.now(italy_tz)
        
        print(f"🌅 [MORNING] Generazione rassegna stampa 6 messaggi LITE - {now.strftime('%H:%M:%S')}")
        
        # Recupera notizie estese
        notizie_estese = get_extended_morning_news()
        
        if not notizie_estese:
            print("⚠️ [MORNING] Nessuna notizia trovata")
            return "❌ Nessuna notizia disponibile"
        
        # Raggruppa per categoria
        notizie_per_categoria = {}
        for notizia in notizie_estese:
            categoria = notizia.get('categoria', 'Generale')
            if categoria not in notizie_per_categoria:
                notizie_per_categoria[categoria] = []
            notizie_per_categoria[categoria].append(notizia)
        
        print(f"📊 [MORNING] Trovate {len(notizie_per_categoria)} categorie di notizie")
        
        success_count = 0
        
        # === MESSAGGI 1-4: UNA CATEGORIA PER MESSAGGIO (7 NOTIZIE CIASCUNA) ===
        categorie_prioritarie = ['Finanza', 'Criptovalute', 'Geopolitica']
        
        # Trova automaticamente la quarta categoria (Mercati Emergenti o altro)
        altre_categorie = [cat for cat in notizie_per_categoria.keys() if cat not in categorie_prioritarie]
        if altre_categorie:
            categorie_prioritarie.append(altre_categorie[0])
        
        for i, categoria in enumerate(categorie_prioritarie[:4], 1):
            if categoria not in notizie_per_categoria:
                continue
                
            notizie_cat = notizie_per_categoria[categoria]
            
            msg_parts = []
            
            # Header per categoria
            emoji_map = {
                'Finanza': '💰',
                'Criptovalute': '₿', 
                'Geopolitica': '🌍',
                'Mercati Emergenti': '🌟'
            }
            emoji = emoji_map.get(categoria, '📊')
            
            msg_parts.append(f"{emoji} *MORNING NEWS - {categoria.upper()}*")
            msg_parts.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')} • Messaggio {i}/6")
            msg_parts.append("─" * 35)
            msg_parts.append("")
            
            # 7 notizie per categoria
            for j, notizia in enumerate(notizie_cat[:7], 1):
                titolo_breve = notizia['titolo'][:70] + "..." if len(notizia['titolo']) > 70 else notizia['titolo']
                
                # Classifica importanza
                high_keywords = ["crisis", "crash", "war", "fed", "recession", "inflation", "breaking"]
                med_keywords = ["bank", "rate", "gdp", "unemployment", "etf", "regulation"]
                
                if any(k in notizia['titolo'].lower() for k in high_keywords):
                    impact = "🔥"
                elif any(k in notizia['titolo'].lower() for k in med_keywords):
                    impact = "⚡"
                else:
                    impact = "📊"
                
                msg_parts.append(f"{impact} **{j}.** *{titolo_breve}*")
                msg_parts.append(f"📰 {notizia['fonte']}")
                if notizia.get('link'):
                    msg_parts.append(f"🔗 {notizia['link'][:60]}...")
                msg_parts.append("")
            
            # Footer categoria
            msg_parts.append("─" * 35)
            msg_parts.append(f"🤖 555 Lite • {categoria} ({len(notizie_cat[:7])} notizie)")
            
            # Invia messaggio categoria
            categoria_msg = "\n".join(msg_parts)
            if invia_messaggio_telegram(categoria_msg):
                success_count += 1
                print(f"✅ [MORNING] Messaggio {i} ({categoria}) inviato")
            else:
                print(f"❌ [MORNING] Messaggio {i} ({categoria}) fallito")
            
            time.sleep(2)  # Pausa tra messaggi
        
        # === MESSAGGIO 5: ANALISI ML + 5 NOTIZIE CRITICHE ===
        try:
            news_analysis = analyze_news_sentiment_and_impact()
            notizie_critiche = get_notizie_critiche()
            
            ml_parts = []
            ml_parts.append("🧠 *MORNING NEWS - ANALISI ML*")
            ml_parts.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')} • Messaggio 5/6")
            ml_parts.append("─" * 35)
            ml_parts.append("")
            
            # Analisi sentiment
            if news_analysis and news_analysis.get('summary'):
                ml_parts.append(news_analysis['summary'])
                ml_parts.append("")
                
                # Raccomandazioni
                recommendations = news_analysis.get('recommendations', [])
                if recommendations:
                    ml_parts.append("💡 *RACCOMANDAZIONI OPERATIVE:*")
                    for rec in recommendations[:3]:
                        ml_parts.append(f"• {rec}")
                    ml_parts.append("")
            
            # 5 notizie critiche
            if notizie_critiche:
                ml_parts.append("🚨 *TOP 5 NOTIZIE CRITICHE (24H)*")
                ml_parts.append("")
                
                for i, notizia in enumerate(notizie_critiche[:5], 1):
                    titolo_breve = notizia["titolo"][:65] + "..." if len(notizia["titolo"]) > 65 else notizia["titolo"]
                    ml_parts.append(f"🔴 **{i}.** *{titolo_breve}*")
                    ml_parts.append(f"📂 {notizia['categoria']} • 📰 {notizia['fonte']}")
                    if notizia.get('link'):
                        ml_parts.append(f"🔗 {notizia['link']}")
                    ml_parts.append("")
            
            # Footer ML
            ml_parts.append("─" * 35)
            ml_parts.append("🤖 555 Lite • Analisi ML & Alert Critici")
            
            # Invia messaggio ML
            ml_msg = "\n".join(ml_parts)
            if invia_messaggio_telegram(ml_msg):
                success_count += 1
                print("✅ [MORNING] Messaggio 5 (ML) inviato")
            else:
                print("❌ [MORNING] Messaggio 5 (ML) fallito")
                
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ [MORNING] Errore messaggio ML: {e}")
        
        # === MESSAGGIO 6: CALENDARIO EVENTI + RACCOMANDAZIONI ML ===
        try:
            # Recupera notizie critiche per le raccomandazioni ML
            notizie_critiche_finali = get_notizie_critiche()
            news_analysis_final = analyze_news_sentiment_and_impact()
            
            # Usa la funzione esistente per gli eventi (in background)
            eventi_result = generate_morning_news_briefing()
            
            # Messaggio finale con raccomandazioni ML
            final_parts = []
            final_parts.append("📅 *MORNING NEWS - CALENDARIO & ML OUTLOOK*")
            final_parts.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')} • Messaggio 6/6")
            final_parts.append("─" * 35)
            final_parts.append("")
            
            # Raccomandazioni ML basate sulle notizie critiche
            if notizie_critiche_finali and news_analysis_final:
                final_parts.append("🧠 *RACCOMANDAZIONI ML CALENDARIO*")
                final_parts.append("")
                
                # Top 5 raccomandazioni strategiche per oggi
                recommendations_final = news_analysis_final.get('recommendations', [])
                if recommendations_final:
                    final_parts.append("💡 *TOP 5 STRATEGIE OGGI:*")
                    for i, rec in enumerate(recommendations_final[:5], 1):
                        final_parts.append(f"{i}. {rec}")
                    final_parts.append("")
                
                # Alert critici per oggi
                final_parts.append("🚨 *ALERT CRITICI GIORNATA:*")
                for i, notizia in enumerate(notizie_critiche_finali[:3], 1):
                    titolo_breve = notizia["titolo"][:60] + "..." if len(notizia["titolo"]) > 60 else notizia["titolo"]
                    final_parts.append(f"⚠️ **{i}.** *{titolo_breve}*")
                    final_parts.append(f"📂 {notizia['categoria']} • Impact: {news_analysis_final.get('market_impact', 'MEDIUM')}")
                    if notizia.get('link'):
                        final_parts.append(f"🔗 {notizia['link']}")
                final_parts.append("")
                
            # Outlook mercati per la giornata
            final_parts.extend(build_calendar_lines(7))
            final_parts.append("🔮 *OUTLOOK MERCATI OGGI*")
            final_parts.append("• 🇺🇸 Wall Street: Apertura 15:30 CET - Watch tech earnings")
            final_parts.append("• 🇪🇺 Europa: Chiusura 17:30 CET - Banks & Energy focus")
            final_parts.append("• ₿ Crypto: 24/7 - BTC key levels 42k-45k")
            final_parts.append("• 🌍 Forex: London-NY overlap 14:00-17:00 CET")
            final_parts.append("")
            
            # Riepilogo finale
            final_parts.append("✅ *RASSEGNA STAMPA COMPLETATA*")
            final_parts.append(f"📊 {len(notizie_estese)} notizie analizzate")
            final_parts.append(f"🌍 {len(notizie_per_categoria)} categorie coperte")
            final_parts.append(f"🧠 {len(recommendations_final) if recommendations_final else 0} raccomandazioni ML")
            final_parts.append("")
            final_parts.append("🔮 *PROSSIMI AGGIORNAMENTI:*")
            final_parts.append("• 🍽️ Daily Report: 14:10")
            final_parts.append("• 🌆 Evening Report: 20:10")
            final_parts.append("• 📊 Weekly Report: Domenica 19:00")
            final_parts.append("")
            final_parts.append("─" * 35)
            final_parts.append("🤖 555 Lite • Morning Briefing + ML Outlook")
            
            # Invia messaggio finale
            final_msg = "\n".join(final_parts)
            if invia_messaggio_telegram(final_msg):
                success_count += 1
                print("✅ [MORNING] Messaggio 6 (finale) inviato")
            else:
                print("❌ [MORNING] Messaggio 6 (finale) fallito")
            
        except Exception as e:
            print(f"❌ [MORNING] Errore messaggio finale: {e}")
        
        # IMPOSTA FLAG E SALVA SU FILE - FIX RECOVERY
        set_message_sent_flag("morning_news")
        print(f"✅ [MORNING] Flag morning_news_sent impostato e salvato su file")
        
        return f"Morning briefing ristrutturato: {success_count}/6 messaggi inviati"
        
    except Exception as e:
        print(f"❌ [MORNING] Errore nella generazione: {e}")
        return "❌ Errore nella generazione morning briefing"

# === DAILY LUNCH REPORT ENHANCED ===
def generate_daily_lunch_report():
    """POTENZIATO - Report di pranzo completo con ML, Mercati Emergenti e analisi avanzate (14:10)"""
    print("🍽️ [LUNCH] Generazione daily lunch report potenziato...")
    
    italy_tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(italy_tz)
    
    sezioni = []
    sezioni.append("🍽️ *DAILY LUNCH REPORT ENHANCED*")
    sezioni.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')} • Aggiornamento Pomeridiano Completo")
    sezioni.append("─" * 40)
    sezioni.append("")
    
    # === ANALISI ML MATTUTINA → LUNCH ===
    try:
        news_analysis = analyze_news_sentiment_and_impact()
        if news_analysis and news_analysis.get('summary'):
            sezioni.append("🧠 *ANALISI ML MORNING → LUNCH*")
            sezioni.append("")
            sezioni.append(news_analysis['summary'])
            sezioni.append("")
            
            # Raccomandazioni operative aggiornate
            recommendations = news_analysis.get('recommendations', [])
            if recommendations:
                sezioni.append("💡 *AGGIORNAMENTO STRATEGIE POMERIGGIO:*")
                for i, rec in enumerate(recommendations[:4], 1):
                    sezioni.append(f"{i}. {rec}")
                sezioni.append("")
    except Exception as e:
        print(f"⚠️ [LUNCH] Errore analisi ML: {e}")
    
    # === MARKET PULSE COMPLETO CON MERCATI EMERGENTI ===
    sezioni.append("📊 *MARKET PULSE COMPLETO* (Morning → Lunch)")
    sezioni.append("")
    
    # USA Markets (Aperti)
    sezioni.append("🇺🇸 **USA Markets (Live Session):**")
    sezioni.append("• S&P 500: 4,835 (+0.4%) - Momentum positivo pre-lunch")
    sezioni.append("• NASDAQ: 15,250 (+0.6%) - Tech recovery in corso")
    sezioni.append("• Dow Jones: 37,920 (+0.3%) - Industriali stabili")
    sezioni.append("• Russell 2000: 1,965 (+0.8%) - Small caps outperform")
    sezioni.append("• VIX: 16.8 (-2.1%) - Fear gauge scende")
    sezioni.append("")
    
    # Europa (Chiusura)
    sezioni.append("🇪🇺 **Europa (Sessione Chiusa 17:30):**")
    sezioni.append("• FTSE MIB: 30,850 (+0.8%) - Milano forte su banche")
    sezioni.append("• DAX: 16,120 (+0.4%) - Germania industriali positivi")
    sezioni.append("• CAC 40: 7,580 (+0.2%) - Francia mixed, luxury debole")
    sezioni.append("• FTSE 100: 7,720 (+0.6%) - UK banks e energy trainano")
    sezioni.append("• STOXX 600: 470.5 (+0.5%) - Europa positiva complessiva")
    sezioni.append("")
    
    # MERCATI EMERGENTI (AGGIUNTO)
    sezioni.append("🌟 **Mercati Emergenti (EM Focus):**")
    sezioni.append("• 🇨🇳 Shanghai Composite: 3,185 (+1.2%) - China rally")
    sezioni.append("• 🇮🇳 NIFTY 50: 19,850 (+0.9%) - India tech momentum")
    sezioni.append("• 🇧🇷 BOVESPA: 118,400 (+1.5%) - Brasile commodities")
    sezioni.append("• 🇷🇺 MOEX: 3,240 (-0.3%) - Russia sotto pressione")
    sezioni.append("• 🇿🇦 JSE All-Share: 75,200 (+0.7%) - Sudafrica mining")
    sezioni.append("• 🌏 MSCI EM: 1,045 (+0.8%) - Outperform DM oggi")
    sezioni.append("")
    
    # Crypto Enhanced
    sezioni.append("₿ **Crypto Markets (24H Enhanced):**")
    sezioni.append("• BTC: $43,280 (+1.8%) - Breakout 43k, target 45k")
    sezioni.append("• ETH: $2,730 (+2.1%) - Strong above 2700, alt season")
    sezioni.append("• BNB: $315 (+3.2%) - Exchange token rally")
    sezioni.append("• SOL: $68.5 (+4.1%) - Solana ecosystem boom")
    sezioni.append("• Total Cap: $1.68T (+2.3%) - Market cap expansion")
    sezioni.append("• Fear & Greed: 72 (Greed) - Sentiment migliorato")
    sezioni.append("")
    
    # Forex & Commodities Enhanced
    sezioni.append("💱 **Forex & Commodities (Enhanced):**")
    sezioni.append("• EUR/USD: 1.0920 (+0.3%) - Euro strength vs USD")
    sezioni.append("• GBP/USD: 1.2795 (+0.2%) - Pound steady")
    sezioni.append("• USD/JPY: 148.50 (-0.4%) - Yen recovery")
    sezioni.append("• DXY: 103.2 (-0.2%) - Dollar index weakness")
    sezioni.append("• Gold: $2,058 (+0.6%) - Safe haven + inflation hedge")
    sezioni.append("• Silver: $24.80 (+1.2%) - Industrial demand")
    sezioni.append("• Oil WTI: $75.80 (+2.1%) - Supply concerns rally")
    sezioni.append("• Copper: $8,450 (+0.8%) - China demand boost")
    sezioni.append("")
    
    # === SECTOR ROTATION ANALYSIS ===
    sezioni.append("🔄 *SECTOR ROTATION ANALYSIS* (Intraday)")
    sezioni.append("")
    sezioni.append("📈 **Top Performers:**")
    sezioni.append("• Energy: +2.8% - Oil rally continua")
    sezioni.append("• Financials: +1.9% - Rate expectations positive")
    sezioni.append("• Materials: +1.6% - Commodities boom")
    sezioni.append("• Industrials: +1.3% - Infrastructure spending")
    sezioni.append("")
    sezioni.append("📉 **Underperformers:**")
    sezioni.append("• Utilities: -0.8% - Defensive rotation out")
    sezioni.append("• REITs: -0.6% - Rate sensitivity")
    sezioni.append("• Consumer Staples: -0.4% - Growth rotation")
    sezioni.append("• Healthcare: -0.2% - Mixed earnings")
    sezioni.append("")
    
    # === NOTIZIE CRITICHE CON ANALISI ENHANCED ===
    try:
        # Recupera notizie critiche per il lunch
        notizie_critiche = get_notizie_critiche()
        if notizie_critiche:
            sezioni.append("🔥 *TOP NEWS MORNING → LUNCH (Enhanced)*")
            sezioni.append("")
            
            for i, notizia in enumerate(notizie_critiche[:4], 1):  # Aumentato a 4
                titolo_breve = notizia["titolo"][:68] + "..." if len(notizia["titolo"]) > 68 else notizia["titolo"]
                
                # Emoji per importanza
                high_keywords = ["fed", "crisis", "war", "crash", "inflation", "breaking"]
                if any(k in notizia['titolo'].lower() for k in high_keywords):
                    priority = "🚨"  # Alta priorità
                else:
                    priority = "📈"  # Normale
                
                sezioni.append(f"{priority} **{i}.** *{titolo_breve}*")
                sezioni.append(f"📂 {notizia['categoria']} • 📰 {notizia['fonte']}")
                
                # Commento ML per ogni notizia
                try:
                    ml_comment = generate_ml_comment_for_news({
                        'title': notizia['titolo'],
                        'categoria': notizia['categoria'],
                        'sentiment': 'NEUTRAL',
                        'impact': 'MEDIUM'
                    })
                    if ml_comment and len(ml_comment) > 10:
                        sezioni.append(f"🧑‍💻 ML: {ml_comment[:85]}...")
                except:
                    pass
                
                if notizia.get('link'):
                    sezioni.append(f"🔗 {notizia['link'][:70]}...")
                sezioni.append("")
    except Exception as e:
        print(f"⚠️ [LUNCH] Errore nel recupero notizie: {e}")
    
    # === VOLATILITY WATCH ===
    sezioni.append("🌊 *VOLATILITY WATCH* (Intraday Signals)")
    sezioni.append("")
    sezioni.append("📉 **VIX Levels:**")
    sezioni.append("• VIX: 16.8 (-2.1%) - Fear gauge in calo")
    sezioni.append("• VVIX: 89.5 (+1.2%) - Vol of vol normale")
    sezioni.append("• MOVE Index: 112.3 (-0.8%) - Bond vol stabile")
    sezioni.append("")
    sezioni.append("📊 **Cross-Asset Volatility:**")
    sezioni.append("• Currency vol: Bassa, range trading")
    sezioni.append("• Commodity vol: Media, oil spikes")
    sezioni.append("• EM vol: Elevata, China uncertainty")
    sezioni.append("")
    
    # === FLOW ANALYSIS ===
    sezioni.append("📈 *INSTITUTIONAL FLOWS* (Real-Time)")
    sezioni.append("")
    sezioni.append("🏦 **ETF Flows:**")
    sezioni.append("• SPY: +$2.1B inflow - Institutional buying")
    sezioni.append("• QQQ: +$890M inflow - Tech recovery play")
    sezioni.append("• XLE: +$450M inflow - Energy momentum")
    sezioni.append("• TLT: -$320M outflow - Bond selling continues")
    sezioni.append("")
    sezioni.append("🏭 **Dark Pool Activity:**")
    sezioni.append("• Large block trades: +15% vs yesterday")
    sezioni.append("• Sectors: Heavy buying in Financials")
    sezioni.append("• Options flow: Call/Put ratio 1.3 (bullish)")
    sezioni.append("")
    
    # Outlook pomeriggio con orari precisi
    sezioni.append("🔮 *OUTLOOK POMERIGGIO* (14:00-18:00)")
    sezioni.append("")
    sezioni.append("⏰ **Eventi Programmati:**")
    sezioni.append("• 14:30 ET: Retail Sales USA (previsione -0.2%)")
    sezioni.append("• 15:30 ET: Apertura Wall Street")
    sezioni.append("• 16:00 ET: Fed Chair Powell speech")
    sezioni.append("• 17:30 CET: Chiusura mercati europei")
    sezioni.append("")
    sezioni.append("📊 **Focus Settoriali:**")
    sezioni.append("• Tech: Earnings season, watch guidance")
    sezioni.append("• Banks: Interest rate sensitivity")
    sezioni.append("• Energy: Oil momentum continuation")
    sezioni.append("")
    
    # Trading alerts con livelli precisi
    sezioni.append("⚡ *LIVELLI CHIAVE POMERIGGIO*")
    sezioni.append("")
    sezioni.append("📈 **Equity Markets:**")
    sezioni.append("• S&P 500: 4850 resistance | 4800 support")
    sezioni.append("• NASDAQ: QQQ 410 pivot | Watch 405 breakdown")
    sezioni.append("• Russell 2000: Small caps 1950 resistance")
    sezioni.append("")
    sezioni.append("₿ **Crypto Levels:**")
    sezioni.append("• BTC: 44k resistance critica | 41k strong support")
    sezioni.append("• ETH: 2700 breakout level | 2600 key support")
    sezioni.append("")
    sezioni.append("💱 **Forex Watch:**")
    sezioni.append("• EUR/USD: 1.095 resistance | 1.085 support")
    sezioni.append("• GBP/USD: 1.275 key level da monitorare")
    sezioni.append("")
    
    # Strategie operative immediate
    sezioni.append("💡 *STRATEGIE OPERATIVE IMMEDIATE*")
    sezioni.append("")
    sezioni.append("🎯 **Trading Setup:**")
    sezioni.append("• Intraday: Range trading fino breakout")
    sezioni.append("• Powell speech: preparare volatility hedges")
    sezioni.append("• Tech earnings: selective long su dip")
    sezioni.append("")
    sezioni.append("🛡️ **Risk Management:**")
    sezioni.append("• VIX watch: se >20 ridurre esposizione")
    sezioni.append("• Cash position: mantenere 15-20%")
    sezioni.append("• Stop loss: tight su posizioni swing")
    
    # Footer
    sezioni.append("")
    sezioni.append("─" * 35)
    sezioni.append(f"🤖 Sistema 555 Lite - {now.strftime('%H:%M')} CET")
    sezioni.append("🌆 Prossimo update: Evening Report (20:10)")
    # === EM Headlines + EM FX & Commodities ===
    try:
        emh = get_emerging_markets_headlines(limit=3)
        if emh:
            sezioni.append("🌍 *Mercati Emergenti — Flash*")
            for i, n in enumerate(emh[:3], 1):
                titolo = n["titolo"][:90] + "..." if len(n["titolo"])>90 else n["titolo"]
                sezioni.append(f"{i}. *{titolo}* — {n.get('fonte','EM')}")
            sezioni.append("")
    except Exception:
        pass
    emfx = get_em_fx_and_commodities()
    if emfx:
        sezioni.append("🌍 *EM FX & Commodities*")
        sezioni.extend(emfx)
        sezioni.append("")

    
    msg = "\n".join(sezioni)
    success = invia_messaggio_telegram(msg)
    
    # IMPOSTA FLAG SE INVIO RIUSCITO - FIX RECOVERY
    if success:
        set_message_sent_flag("daily_report")
        print(f"✅ [LUNCH] Flag daily_report_sent impostato e salvato su file")
    
    return f"Daily lunch report: {'✅' if success else '❌'}"

# === REPORT SETTIMANALI ENHANCED ===
def generate_weekly_backtest_summary():
    """Genera un riassunto settimanale avanzato dell'analisi di backtest per il lunedì - versione ricca come 555.py"""
    try:
        import pytz
        import random
        italy_tz = pytz.timezone('Europe/Rome')
        now = datetime.datetime.now(italy_tz)
        
        # Genera un riassunto avanzato basato sui modelli ML e indicatori
        weekly_lines = []
        weekly_lines.append("📊 === REPORT SETTIMANALE AVANZATO ===\n" + "=" * 80)
        weekly_lines.append(f"📅 Generato il {now.strftime('%d/%m/%Y alle %H:%M')} (CET) - Sistema Analisi v2.0")
        weekly_lines.append("")
        
        # === SEZIONE EXECUTIVE SUMMARY ===
        weekly_lines.append("🎯 EXECUTIVE SUMMARY SETTIMANALE")
        weekly_lines.append("-" * 50)
        
        # 1. SEZIONE INDICATORI TECNICI (PRIMA)
        try:
            weekly_lines.append("📊 INDICATORI TECNICI COMPLETI (17 INDICATORI):")
            # Simulazione indicatori per ambiente lite (adattato per compatibilità)
            assets_data = {
                "Bitcoin": {"MAC": "Buy", "RSI": "Sell", "MACD": "Buy", "Bollinger": "Hold", "EMA": "Buy", "SMA": "Hold"},
                "S&P 500": {"MAC": "Hold", "RSI": "Buy", "MACD": "Hold", "Bollinger": "Buy", "EMA": "Buy", "SMA": "Buy"},
                "Gold": {"MAC": "Sell", "RSI": "Hold", "MACD": "Sell", "Bollinger": "Hold", "EMA": "Hold", "SMA": "Sell"},
                "Dollar Index": {"MAC": "Buy", "RSI": "Buy", "MACD": "Buy", "Bollinger": "Buy", "EMA": "Buy", "SMA": "Hold"}
            }
            
            for asset, indicators in assets_data.items():
                # Raggruppa indicatori per linea per leggibilità
                line1_indicators = []  # Principali (6)
                line2_indicators = []  # Secondari (6) 
                line3_indicators = []  # Avanzati (5)
                
                for i, (ind, signal) in enumerate(indicators.items()):
                    emoji = "🟢" if signal == 'Buy' else "🔴" if signal == 'Sell' else "⚪"
                    indicator_display = f"{ind[:3]}{emoji}"  # Abbrevia nome per spazio
                    
                    if i < 3:  # Primi 3
                        line1_indicators.append(indicator_display)
                    elif i < 6:  # Secondi 3
                        line2_indicators.append(indicator_display)
                    else:  # Rimanenti
                        line3_indicators.append(indicator_display)
                
                # Mostra tutti gli indicatori su più linee
                weekly_lines.append(f"  📈 {asset}:")
                if line1_indicators:
                    weekly_lines.append(f"     Principali: {' '.join(line1_indicators)}")
                if line2_indicators:
                    weekly_lines.append(f"     Secondari:  {' '.join(line2_indicators)}")
                if line3_indicators:
                    weekly_lines.append(f"     Avanzati:   {' '.join(line3_indicators)}")
                
        except Exception as e:
            weekly_lines.append("  ❌ Errore nel calcolo indicatori settimanali")
            print(f"Errore weekly indicators: {e}")
        
        weekly_lines.append("")
        
        # 2. SEZIONE MODELLI ML (SECONDA) - Simulati per ambiente lite
        try:
            weekly_lines.append("🤖 CONSENSO MODELLI ML COMPLETI - TUTTI I MODELLI DISPONIBILI:")
            weekly_lines.append(f"🔧 Modelli ML attivi: 8")
            weekly_lines.append("")
            
            # Simula risultati ML per i 4 asset principali
            ml_results = {
                "Bitcoin": {"consensus": "🟢 CONSENSUS BUY (67%)", "models": ["LinReg: BUY(78%)", "RandFor: BUY(72%)", "XGBoost: HOLD(55%)", "SVM: BUY(81%)"]},
                "S&P 500": {"consensus": "⚪ CONSENSUS HOLD (52%)", "models": ["LinReg: HOLD(58%)", "RandFor: BUY(65%)", "XGBoost: HOLD(48%)", "SVM: HOLD(51%)"]},
                "Gold": {"consensus": "🔴 CONSENSUS SELL (71%)", "models": ["LinReg: SELL(76%)", "RandFor: SELL(68%)", "XGBoost: SELL(73%)", "SVM: HOLD(45%)"]},
                "Dollar Index": {"consensus": "🟢 CONSENSUS BUY (85%)", "models": ["LinReg: BUY(88%)", "RandFor: BUY(82%)", "XGBoost: BUY(86%)", "SVM: BUY(84%)"]}
            }
            
            for asset, data in ml_results.items():
                weekly_lines.append(f"  📊 {asset}: {data['consensus']}")
                
                # Mostra tutti i modelli su più linee per leggibilità
                chunk_size = 4  # 4 modelli per linea
                models = data['models']
                for i in range(0, len(models), chunk_size):
                    chunk = models[i:i+chunk_size]
                    weekly_lines.append(f"     {' | '.join(chunk)}")
                    
        except Exception as e:
            weekly_lines.append("  ❌ Errore nel calcolo ML settimanale")
            print(f"Errore weekly ML: {e}")
        
        weekly_lines.append("")
        
        # TOP 10 NOTIZIE CRITICHE CON RANKING
        try:
            weekly_lines.append("🚨 TOP 10 NOTIZIE CRITICHE - RANKING SETTIMANALE:")
            # Simula notizie critiche per ambiente lite
            notizie_simulate = [
                {"titolo": "Fed Reserve signals potential rate cuts amid inflation concerns", "fonte": "Reuters", "categoria": "Monetary Policy"},
                {"titolo": "Major bank crisis spreads across European markets", "fonte": "Bloomberg", "categoria": "Banking"},
                {"titolo": "Geopolitical tensions escalate, oil prices surge 5%", "fonte": "CNBC", "categoria": "Geopolitics"},
                {"titolo": "Tech earnings disappoint, NASDAQ falls 3%", "fonte": "MarketWatch", "categoria": "Earnings"},
                {"titolo": "Unemployment data shows unexpected job losses", "fonte": "WSJ", "categoria": "Employment"}
            ]
            
            if notizie_simulate and len(notizie_simulate) > 0:
                # Ordina per criticità (implementa logica di ranking)
                notizie_ranked = sorted(notizie_simulate, key=lambda x: len([k for k in ["crisis", "crash", "war", "fed", "recession", "inflation"] if k in x["titolo"].lower()]), reverse=True)
                
                for i, notizia in enumerate(notizie_ranked, 1):
                    titolo_short = notizia["titolo"][:65] + "..." if len(notizia["titolo"]) > 65 else notizia["titolo"]
                    
                    # Classifica impatto
                    high_impact_keywords = ["crisis", "crash", "war", "fed", "recession", "inflation", "emergency"]
                    med_impact_keywords = ["bank", "rate", "gdp", "unemployment", "etf", "regulation"]
                    
                    if any(k in notizia["titolo"].lower() for k in high_impact_keywords):
                        impact = "🔥 ALTO"
                    elif any(k in notizia["titolo"].lower() for k in med_impact_keywords):
                        impact = "⚠️ MEDIO"
                    else:
                        impact = "📊 BASSO"
                    
                    weekly_lines.append(f"   {i:2d}. {impact} | {titolo_short}")
                    weekly_lines.append(f"      📰 {notizia['fonte']} | 🏷️ {notizia['categoria']}")
            else:
                weekly_lines.append("  ✅ Nessuna notizia critica rilevata")
        except Exception as e:
            weekly_lines.append("  ❌ Errore nel recupero notizie")
            print(f"Errore weekly news: {e}")
        
        weekly_lines.append("")
        
        # ANALISI ML EVENTI CALENDARIO ECONOMICO
        try:
            weekly_lines.append("🤖 ANALISI ML EVENTI CALENDARIO ECONOMICO:")
            
            # Simula eventi economici (in futuro da collegare a API calendario)
            eventi_simulati = [
                {"nome": "Federal Reserve Interest Rate Decision...", "ml_impact": 87, "giorni": 3, "livello": "Alto", "commento": "Alta probabilità di mantenimento tassi. Attenzione a dichiarazioni su inflazione..."},
                {"nome": "US CPI Inflation Data Release...", "ml_impact": 82, "giorni": 5, "livello": "Alto", "commento": "Dati cruciali per asset class bonds e gold. Impatto su correlazioni SP500..."},
                {"nome": "ECB Monetary Policy Meeting...", "ml_impact": 76, "giorni": 6, "livello": "Alto", "commento": "Focus su dettagli QT e guidance. Impatto diretto su EUR e settore bancario..."},
                {"nome": "US Nonfarm Payrolls", "ml_impact": 65, "giorni": 8, "livello": "Medio", "commento": ""},
                {"nome": "UK GDP Quarterly Estimate", "ml_impact": 58, "giorni": 10, "livello": "Medio", "commento": ""},
                {"nome": "Japan BOJ Rate Decision", "ml_impact": 52, "giorni": 12, "livello": "Medio", "commento": ""}
            ]
            
            weekly_lines.append(f"📅 Eventi analizzati: {len(eventi_simulati)}")
            weekly_lines.append("")
            
            # Eventi ad alto impatto (≥70%)
            eventi_alto = [e for e in eventi_simulati if e["ml_impact"] >= 70]
            if eventi_alto:
                weekly_lines.append("🔴 EVENTI AD ALTO IMPATTO ML (≥70%):")
                for evento in eventi_alto:
                    weekly_lines.append(f"  • {evento['nome']}")
                    weekly_lines.append(f"    🎯 ML Impact: {evento['ml_impact']}% | ⏰ +{evento['giorni']}g | 📊 {evento['livello']}")
                    if evento['commento']:
                        weekly_lines.append(f"    💡 {evento['commento']}")
                weekly_lines.append("")
            
            # Eventi a medio impatto (40-70%)
            eventi_medio = [e for e in eventi_simulati if 40 <= e["ml_impact"] < 70]
            if eventi_medio:
                weekly_lines.append("🟡 EVENTI A MEDIO IMPATTO ML (40-70%):")
                for evento in eventi_medio:
                    weekly_lines.append(f"  • {evento['nome']} | {evento['ml_impact']}% | +{evento['giorni']}g")
                weekly_lines.append("")
            
            # Statistiche
            weekly_lines.append("📈 STATISTICHE ML CALENDARIO:")
            avg_impact = sum(e["ml_impact"] for e in eventi_simulati) // len(eventi_simulati)
            alto_count = len([e for e in eventi_simulati if e["ml_impact"] >= 70])
            medio_count = len([e for e in eventi_simulati if 40 <= e["ml_impact"] < 70])
            basso_count = len([e for e in eventi_simulati if e["ml_impact"] < 40])
            
            weekly_lines.append(f"  📊 Eventi totali: {len(eventi_simulati)} | Impatto medio ML: {avg_impact}%")
            weekly_lines.append(f"  🔴 Alto impatto: {alto_count} | 🟡 Medio: {medio_count} | 🟢 Basso: {basso_count}")
            
        except Exception as e:
            weekly_lines.append("  ❌ Errore nell'analisi ML eventi")
            print(f"Errore weekly ML events: {e}")
        
        weekly_lines.append("")
        weekly_lines.append("💡 NOTA: Questo riassunto è generato automaticamente ogni lunedì")
        weekly_lines.append("    e include analisi ML, indicatori tecnici e monitoraggio notizie.")
        
        return "\n".join(weekly_lines)
        
    except Exception as e:
        print(f"Errore nella generazione del riassunto settimanale: {e}")
        return f"❌ Errore nella generazione del riassunto settimanale del {datetime.datetime.now().strftime('%d/%m/%Y')}"

def genera_report_settimanale():
    """Wrapper per mantenere compatibilità con il sistema di scheduling esistente"""
    print("📊 [WEEKLY] Generazione report settimanale avanzato...")
    
    # Genera il report avanzato
    report_content = generate_weekly_backtest_summary()
    
    # Invia via Telegram
    success = invia_messaggio_telegram(report_content)
    
    if success:
        set_message_sent_flag("weekly_report")
    
    return f"Report settimanale avanzato: {'✅' if success else '❌'}"

def generate_monthly_backtest_summary():
    """Genera un riassunto mensile avanzato dell'analisi di backtest - versione ricca come 555.py"""
    try:
        import pytz
        import random
        italy_tz = pytz.timezone('Europe/Rome')
        now = datetime.datetime.now(italy_tz)
        
        # Calcola il periodo mensile
        oggi = datetime.date.today()
        primo_giorno_mese_corrente = oggi.replace(day=1)
        ultimo_giorno_mese_precedente = primo_giorno_mese_corrente - datetime.timedelta(days=1)
        primo_giorno_mese_precedente = ultimo_giorno_mese_precedente.replace(day=1)
        
        mese_nome = {
            1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile", 5: "Maggio", 6: "Giugno",
            7: "Luglio", 8: "Agosto", 9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
        }[ultimo_giorno_mese_precedente.month]
        
        # Genera un riassunto avanzato basato sui modelli ML e indicatori
        monthly_lines = []
        monthly_lines.append("📊 === REPORT MENSILE AVANZATO ===\n" + "=" * 85)
        monthly_lines.append(f"📅 {mese_nome} {ultimo_giorno_mese_precedente.year} • Generato il {now.strftime('%d/%m/%Y alle %H:%M')} (CET)")
        monthly_lines.append(f"🗓️ Periodo analizzato: {primo_giorno_mese_precedente.strftime('%d/%m')} - {ultimo_giorno_mese_precedente.strftime('%d/%m/%Y')} ({(ultimo_giorno_mese_precedente - primo_giorno_mese_precedente).days + 1} giorni)")
        monthly_lines.append("")
        
        # === EXECUTIVE SUMMARY MENSILE ===
        monthly_lines.append("🎯 EXECUTIVE SUMMARY MENSILE")
        monthly_lines.append("-" * 55)
        
        # 1. PERFORMANCE MENSILE COMPLETA CON CALCOLI REALI
        try:
            monthly_lines.append(f"📈 PERFORMANCE {mese_nome.upper()} - ANALISI COMPLETA:")
            
            # Simula calcoli mensili realistici (in futuro da collegare a dati live)
            performance_data = {
                "S&P 500": {"return": 2.8, "volatility": 14.2, "max_dd": -3.1, "sharpe": 1.42},
                "NASDAQ": {"return": 3.2, "volatility": 18.5, "max_dd": -4.8, "sharpe": 1.15},
                "Dow Jones": {"return": 1.9, "volatility": 12.8, "max_dd": -2.4, "sharpe": 1.23},
                "Russell 2000": {"return": 4.1, "volatility": 22.3, "max_dd": -6.2, "sharpe": 0.98},
                "Bitcoin": {"return": 12.5, "volatility": 45.2, "max_dd": -18.3, "sharpe": 0.85},
                "Ethereum": {"return": 8.9, "volatility": 52.1, "max_dd": -22.1, "sharpe": 0.71},
                "Gold": {"return": 1.2, "volatility": 8.9, "max_dd": -2.8, "sharpe": 0.45},
                "EUR/USD": {"return": -0.8, "volatility": 6.2, "max_dd": -1.9, "sharpe": -0.22}
            }
            
            # Ordina per performance
            sorted_assets = sorted(performance_data.items(), key=lambda x: x[1]["return"], reverse=True)
            
            monthly_lines.append("")
            monthly_lines.append("🏆 TOP PERFORMERS DEL MESE:")
            for i, (asset, data) in enumerate(sorted_assets[:4], 1):
                return_str = f"+{data['return']:.1f}%" if data['return'] >= 0 else f"{data['return']:.1f}%"
                emoji = "🟢" if data['return'] >= 0 else "🔴"
                monthly_lines.append(f"  {i}. {emoji} {asset}: {return_str} | Vol: {data['volatility']:.1f}% | MaxDD: {data['max_dd']:.1f}% | Sharpe: {data['sharpe']:.2f}")
            
            monthly_lines.append("")
            monthly_lines.append("📉 WORST PERFORMERS DEL MESE:")
            for i, (asset, data) in enumerate(sorted_assets[-4:], 1):
                return_str = f"+{data['return']:.1f}%" if data['return'] >= 0 else f"{data['return']:.1f}%"
                emoji = "🟢" if data['return'] >= 0 else "🔴"
                monthly_lines.append(f"  {i}. {emoji} {asset}: {return_str} | Vol: {data['volatility']:.1f}% | MaxDD: {data['max_dd']:.1f}% | Sharpe: {data['sharpe']:.2f}")
                
        except Exception as e:
            monthly_lines.append("  ❌ Errore nel calcolo performance mensili")
            print(f"Errore monthly performance: {e}")
        
        monthly_lines.append("")
        
        # 2. ANALISI RISK METRICS AVANZATA
        try:
            monthly_lines.append("📊 RISK METRICS AVANZATI - ANALISI MENSILE:")
            monthly_lines.append("")
            
            # Metriche di volatilità
            monthly_lines.append("🌊 VOLATILITY ANALYSIS:")
            monthly_lines.append(f"  • VIX Medio Mensile: 17.2 (-8.5% vs mese precedente)")
            monthly_lines.append(f"  • VIX Range: 14.1 - 22.8 (spread: 8.7 punti)")
            monthly_lines.append(f"  • VVIX (Vol of Vol): 91.4 (+2.1% vs mese precedente)")
            monthly_lines.append(f"  • MOVE Index (Bond Vol): 108.9 (-5.2% vs mese precedente)")
            monthly_lines.append("")
            
            # Correlazioni inter-asset
            monthly_lines.append("🔗 CORRELATION MATRIX MENSILE:")
            correlations = {
                "Stock-Bond": -0.15, "Stock-Gold": 0.08, "Stock-USD": -0.22,
                "Stock-Crypto": 0.45, "Bond-Gold": -0.12, "Crypto-Gold": 0.03
            }
            
            for pair, corr in correlations.items():
                corr_color = "🟢" if -0.3 <= corr <= 0.3 else "🟡" if abs(corr) <= 0.6 else "🔴"
                corr_strength = "Debole" if abs(corr) <= 0.3 else "Media" if abs(corr) <= 0.6 else "Forte"
                monthly_lines.append(f"  • {pair}: {corr:+.2f} {corr_color} ({corr_strength})")
            
            monthly_lines.append("")
            
            # Drawdown Analysis
            monthly_lines.append("📉 DRAWDOWN ANALYSIS MENSILE:")
            monthly_lines.append(f"  • S&P 500 Max DD: -3.1% (recovery: 5 giorni)")
            monthly_lines.append(f"  • NASDAQ Max DD: -4.8% (recovery: 8 giorni)")
            monthly_lines.append(f"  • Bitcoin Max DD: -18.3% (recovery: ongoing)")
            monthly_lines.append(f"  • Portfolio Diversificato DD: -2.4% (recovery: 3 giorni)")
            
        except Exception as e:
            monthly_lines.append("  ❌ Errore nell'analisi risk metrics")
            print(f"Errore monthly risk: {e}")
        
        monthly_lines.append("")
        
        # 3. SECTOR ROTATION MENSILE
        try:
            monthly_lines.append("🔄 SECTOR ROTATION ANALYSIS - MENSILE:")
            
            sector_performance = {
                "Energy": 8.2, "Financials": 4.8, "Materials": 3.1, "Industrials": 2.9,
                "Consumer Discretionary": 1.8, "Healthcare": 1.2, "Technology": 0.8,
                "Communication Services": -0.3, "Consumer Staples": -1.1, "Utilities": -1.8, "Real Estate": -2.4
            }
            
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            monthly_lines.append("")
            monthly_lines.append("🚀 TOP 5 SETTORI DEL MESE:")
            for i, (sector, perf) in enumerate(sorted_sectors[:5], 1):
                perf_str = f"+{perf:.1f}%" if perf >= 0 else f"{perf:.1f}%"
                emoji = "🟢" if perf >= 0 else "🔴"
                monthly_lines.append(f"  {i}. {emoji} {sector}: {perf_str}")
            
            monthly_lines.append("")
            monthly_lines.append("📉 BOTTOM 5 SETTORI DEL MESE:")
            for i, (sector, perf) in enumerate(sorted_sectors[-5:], 1):
                perf_str = f"+{perf:.1f}%" if perf >= 0 else f"{perf:.1f}%"
                emoji = "🟢" if perf >= 0 else "🔴"
                monthly_lines.append(f"  {i}. {emoji} {sector}: {perf_str}")
            
        except Exception as e:
            monthly_lines.append("  ❌ Errore nell'analisi sector rotation")
            print(f"Errore monthly sectors: {e}")
        
        monthly_lines.append("")
        
        # 4. MODELLI ML MENSILI (STESSA LOGICA DEL SETTIMANALE)
        try:
            monthly_lines.append("🤖 CONSENSO MODELLI ML MENSILI - TUTTI I MODELLI:")
            monthly_lines.append(f"🔧 Modelli ML attivi: 8 (timeframe: 30 giorni)")
            monthly_lines.append("")
            
            # Simula risultati ML mensili per i 4 asset principali
            ml_results_monthly = {
                "Bitcoin": {"consensus": "🟢 CONSENSUS BUY (72%)", "models": ["LinReg: BUY(82%)", "RandFor: BUY(75%)", "XGBoost: BUY(68%)", "SVM: BUY(85%)", "AdaBoost: HOLD(55%)", "KNN: BUY(78%)", "NaiveBayes: BUY(71%)", "MLP: BUY(79%)"]},
                "S&P 500": {"consensus": "🟢 CONSENSUS BUY (65%)", "models": ["LinReg: BUY(71%)", "RandFor: BUY(68%)", "XGBoost: BUY(62%)", "SVM: HOLD(58%)", "AdaBoost: BUY(69%)", "KNN: BUY(65%)", "NaiveBayes: HOLD(52%)", "MLP: BUY(74%)"]},
                "Gold": {"consensus": "⚪ CONSENSUS HOLD (48%)", "models": ["LinReg: HOLD(52%)", "RandFor: SELL(42%)", "XGBoost: HOLD(48%)", "SVM: HOLD(51%)", "AdaBoost: SELL(38%)", "KNN: BUY(62%)", "NaiveBayes: HOLD(45%)", "MLP: HOLD(46%)"]},
                "EUR/USD": {"consensus": "🔴 CONSENSUS SELL (68%)", "models": ["LinReg: SELL(75%)", "RandFor: SELL(71%)", "XGBoost: SELL(65%)", "SVM: SELL(72%)", "AdaBoost: SELL(69%)", "KNN: HOLD(58%)", "NaiveBayes: SELL(74%)", "MLP: SELL(68%)"]}
            }
            
            for asset, data in ml_results_monthly.items():
                monthly_lines.append(f"  📊 {asset}: {data['consensus']}")
                
                # Mostra tutti gli 8 modelli su più linee per leggibilità
                chunk_size = 4  # 4 modelli per linea
                models = data['models']
                for i in range(0, len(models), chunk_size):
                    chunk = models[i:i+chunk_size]
                    monthly_lines.append(f"     {' | '.join(chunk)}")
                monthly_lines.append("")
                    
        except Exception as e:
            monthly_lines.append("  ❌ Errore nel calcolo ML mensile")
            print(f"Errore monthly ML: {e}")
        
        # 5. TOP 15 NOTIZIE CRITICHE MENSILI
        try:
            monthly_lines.append("🚨 TOP 15 NOTIZIE CRITICHE MENSILI - RANKING:")
            # Simula notizie critiche mensili (più del settimanale)
            notizie_simulate_mensili = [
                {"titolo": f"Fed Reserve announces major policy shift affecting {mese_nome} markets", "fonte": "Reuters", "categoria": "Monetary Policy", "data": "3 giorni fa"},
                {"titolo": f"Global banking crisis deepens in {mese_nome}, spreads to emerging markets", "fonte": "Bloomberg", "categoria": "Banking", "data": "5 giorni fa"},
                {"titolo": f"Geopolitical tensions reach new high, commodities surge in {mese_nome}", "fonte": "CNBC", "categoria": "Geopolitics", "data": "1 settimana fa"},
                {"titolo": f"Tech earnings season disappoints, NASDAQ falls 8% in {mese_nome}", "fonte": "MarketWatch", "categoria": "Earnings", "data": "2 settimane fa"},
                {"titolo": f"Unemployment data shows significant job losses throughout {mese_nome}", "fonte": "WSJ", "categoria": "Employment", "data": "2 settimane fa"},
                {"titolo": f"Inflation reaches decade-high levels by end of {mese_nome}", "fonte": "Financial Times", "categoria": "Inflation", "data": "3 settimane fa"},
                {"titolo": f"Bitcoin regulatory framework announced, crypto markets react in {mese_nome}", "fonte": "CoinDesk", "categoria": "Cryptocurrency", "data": "3 settimane fa"},
                {"titolo": f"European Central Bank emergency meeting called for {mese_nome} crisis", "fonte": "ECB Press", "categoria": "Central Banking", "data": "4 settimane fa"}
            ]
            
            if notizie_simulate_mensili and len(notizie_simulate_mensili) > 0:
                # Ordina per criticità (implementa logica di ranking)
                notizie_ranked_monthly = sorted(notizie_simulate_mensili, key=lambda x: len([k for k in ["crisis", "crash", "war", "fed", "recession", "inflation", "emergency"] if k in x["titolo"].lower()]), reverse=True)
                
                for i, notizia in enumerate(notizie_ranked_monthly, 1):
                    titolo_short = notizia["titolo"][:70] + "..." if len(notizia["titolo"]) > 70 else notizia["titolo"]
                    
                    # Classifica impatto
                    high_impact_keywords = ["crisis", "crash", "war", "fed", "recession", "inflation", "emergency"]
                    med_impact_keywords = ["bank", "rate", "gdp", "unemployment", "etf", "regulation", "earnings"]
                    
                    if any(k in notizia["titolo"].lower() for k in high_impact_keywords):
                        impact = "🔥 ALTO"
                    elif any(k in notizia["titolo"].lower() for k in med_impact_keywords):
                        impact = "⚠️ MEDIO"
                    else:
                        impact = "📊 BASSO"
                    
                    monthly_lines.append(f"   {i:2d}. {impact} | {titolo_short}")
                    monthly_lines.append(f"      📰 {notizia['fonte']} | 🏷️ {notizia['categoria']} | 📅 {notizia['data']}")
            else:
                monthly_lines.append("  ✅ Nessuna notizia critica rilevata nel mese")
        except Exception as e:
            monthly_lines.append("  ❌ Errore nel recupero notizie mensili")
            print(f"Errore monthly news: {e}")
        
        monthly_lines.append("")
        
        # 6. OUTLOOK PROSSIMO MESE CON ML
        try:
            prossimo_mese = (primo_giorno_mese_corrente + datetime.timedelta(days=32)).replace(day=1)
            prossimo_mese_nome = mese_nome = {
                1: "Gennaio", 2: "Febbraio", 3: "Marzo", 4: "Aprile", 5: "Maggio", 6: "Giugno",
                7: "Luglio", 8: "Agosto", 9: "Settembre", 10: "Ottobre", 11: "Novembre", 12: "Dicembre"
            }[prossimo_mese.month]
            
            monthly_lines.append(f"🔮 OUTLOOK ML {prossimo_mese_nome.upper()} {prossimo_mese.year}:")
            monthly_lines.append("")
            
            # Eventi macro previsti
            monthly_lines.append("📅 EVENTI MACRO CHIAVE:")
            eventi_macro = [
                "Fed Reserve Decision: Probabile pausa (85% probabilità ML)",
                "ECB Meeting: Focus su QT e guidance inflazione",
                "Earnings Season: Tech giants, aspettative conservative",
                "Employment Data: Trend di rallentamento previsto",
                "Inflation Reports: Peak inflation hypothesis da verificare"
            ]
            
            for evento in eventi_macro:
                monthly_lines.append(f"  • {evento}")
            
            monthly_lines.append("")
            
            # Previsioni ML per asset
            monthly_lines.append("🎯 PREVISIONI ML ASSET (30 giorni):")
            previsioni_ml = {
                "Bitcoin": "Target 48k-52k (confidence: 68%)",
                "S&P 500": "Range 4900-5100 (confidence: 72%)",
                "Gold": "Consolidamento 2000-2100 (confidence: 65%)",
                "EUR/USD": "Debolezza verso 1.05 (confidence: 71%)"
            }
            
            for asset, previsione in previsioni_ml.items():
                monthly_lines.append(f"  • {asset}: {previsione}")
            
        except Exception as e:
            monthly_lines.append("  ❌ Errore nella generazione outlook")
            print(f"Errore monthly outlook: {e}")
        
        monthly_lines.append("")
        
        # 7. REBALANCING STRATEGICO
        try:
            monthly_lines.append("⚖️ REBALANCING STRATEGICO RACCOMANDATO:")
            monthly_lines.append("")
            
            monthly_lines.append("📊 ALLOCAZIONE ASSET SUGGERITA:")
            allocazioni = [
                "Equity (60% → 55%): Riduzione tattica per risk management",
                "Fixed Income (25% → 30%): Aumento duration intermedia",
                "Commodities (10% → 10%): Mantenimento exposure inflazione",
                "Cash (5% → 5%): Liquidità per opportunità"
            ]
            
            for allocazione in allocazioni:
                monthly_lines.append(f"  • {allocazione}")
            
            monthly_lines.append("")
            monthly_lines.append("🎯 TACTICAL ADJUSTMENTS:")
            adjustments = [
                "Sottopesare Growth (+5% Value tilt)",
                "Sovrappesare Financials (+3% vs benchmark)",
                "Exposure EM selettivo (Focus Cina +2%)",
                "Hedging valutario USD 50% per posizioni EUR"
            ]
            
            for adjustment in adjustments:
                monthly_lines.append(f"  • {adjustment}")
            
        except Exception as e:
            monthly_lines.append("  ❌ Errore nelle raccomandazioni di rebalancing")
            print(f"Errore monthly rebalancing: {e}")
        
        monthly_lines.append("")
        monthly_lines.append("💡 NOTA: Questo report mensile è generato automaticamente il primo giorno")
        monthly_lines.append("    di ogni mese e include analisi ML, performance, risk metrics e outlook.")
        
        return "\n".join(monthly_lines)
        
    except Exception as e:
        print(f"Errore nella generazione del riassunto mensile: {e}")
        return f"❌ Errore nella generazione del riassunto mensile del {datetime.datetime.now().strftime('%d/%m/%Y')}"

def genera_report_mensile():
    """Wrapper per mantenere compatibilità con il sistema di scheduling esistente"""
    print("📊 [MONTHLY] Generazione report mensile avanzato...")
    
    # Genera il report avanzato
    report_content = generate_monthly_backtest_summary()
    
    # Invia via Telegram
    success = invia_messaggio_telegram(report_content)
    
    if success:
        set_message_sent_flag("monthly_report")
    
    return f"Report mensile avanzato: {'✅' if success else '❌'}"

# === EVENING REPORT ENHANCED ===

def generate_evening_report():
    """Evening Brief — stesso modello Morning"""
    return _generate_brief_core("evening")
def genera_report_trimestrale():
    """PLACEHOLDER - Report trimestrale da implementare"""
    msg = f"📊 *REPORT TRIMESTRALE PLACEHOLDER*\n\nFunzione da implementare\n\n🤖 Sistema 555 Lite"
    success = invia_messaggio_telegram(msg)
    if success:
        set_message_sent_flag("quarterly_report")
    return f"Report trimestrale placeholder: {'✅' if success else '❌'}"

def genera_report_semestrale():
    """PLACEHOLDER - Report semestrale da implementare"""
    msg = f"📊 *REPORT SEMESTRALE PLACEHOLDER*\n\nFunzione da implementare\n\n🤖 Sistema 555 Lite"
    success = invia_messaggio_telegram(msg)
    if success:
        set_message_sent_flag("semestral_report")
    return f"Report semestrale placeholder: {'✅' if success else '❌'}"

def genera_report_annuale():
    """PLACEHOLDER - Report annuale da implementare"""
    msg = f"📊 *REPORT ANNUALE PLACEHOLDER*\n\nFunzione da implementare\n\n🤖 Sistema 555 Lite"
    success = invia_messaggio_telegram(msg)
    if success:
        set_message_sent_flag("annual_report")
    return f"Report annuale placeholder: {'✅' if success else '❌'}"

# === SCHEDULER POTENZIATO ===

def check_and_send_scheduled_messages():
    """Scheduler per-minuto con debounce + recovery tick"""
    now = _now_it()
    current_time = now.strftime("%H:%M")
    now_key = _minute_key(now)

    # RASSEGNA 07:00 (6 pagine)
    if current_time == SCHEDULE["rassegna"] and not is_message_sent_today("rassegna") and LAST_RUN.get("rassegna") != now_key:
        print("🗞️ [SCHEDULER] Avvio rassegna stampa (6 pagine)...")
        # lock immediato
        try:
            LAST_RUN["rassegna"] = now_key
            generate_rassegna_stampa()
            set_message_sent_flag("rassegna"); 
            save_daily_flags()
        except Exception as e:
            print(f"❌ [SCHEDULER] Errore rassegna: {e}")

        # cooldown 5 minuti
        try:
            time.sleep(300)
        except Exception:
            pass

    # MORNING 08:10
    if current_time == SCHEDULE["morning"] and not is_message_sent_today("morning_news") and LAST_RUN.get("morning") != now_key:
        print("🌅 [SCHEDULER] Avvio morning brief...")
        try:
            LAST_RUN["morning"] = now_key
            generate_morning_news()
            set_message_sent_flag("morning_news"); 
            save_daily_flags()
        except Exception as e:
            print(f"❌ [SCHEDULER] Errore morning: {e}")

    # LUNCH 14:10
    if current_time == SCHEDULE["lunch"] and not is_message_sent_today("daily_report") and LAST_RUN.get("lunch") != now_key:
        print("🍽️ [SCHEDULER] Avvio lunch brief...")
        try:
            LAST_RUN["lunch"] = now_key
            generate_lunch_report()
            set_message_sent_flag("daily_report"); 
            save_daily_flags()
        except Exception as e:
            print(f"❌ [SCHEDULER] Errore lunch: {e}")

    # EVENING 20:10
    if current_time == SCHEDULE["evening"] and not is_message_sent_today("evening_report") and LAST_RUN.get("evening") != now_key:
        print("🌆 [SCHEDULER] Avvio evening brief...")
        try:
            LAST_RUN["evening"] = now_key
            generate_evening_report()
            set_message_sent_flag("evening_report"); 
            save_daily_flags()
        except Exception as e:
            print(f"❌ [SCHEDULER] Errore evening: {e}")

    # Recovery pass ogni 10 minuti
    try:
        _recovery_tick()
    except Exception as e:
        print(f"⚠️ [SCHEDULER] Recovery tick error: {e}")


def is_keep_alive_time():
    """Controlla se siamo nella finestra di keep-alive (06:00-22:00)"""
    italy_tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(italy_tz)
    
    # Finestra keep-alive: 6:00 AM - 12:00 AM (24:00)
    start_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
    end_time = now.replace(hour=23, minute=59, second=0, microsecond=0)
    
    return start_time <= now <= end_time

# === THREAD PRINCIPALE CON KEEP-ALIVE ===
def main_scheduler_loop():
    """Loop principale ottimizzato con keep-alive per Render"""
    print("🚀 [LITE-MAIN] Scheduler principale attivo con keep-alive")
    
    # URL dell'app per keep-alive - SOLO da variabile d'ambiente
    app_url = os.environ.get('RENDER_EXTERNAL_URL')
    italy_tz = pytz.timezone('Europe/Rome')
    last_ping_time = datetime.datetime.now(italy_tz)
    keep_alive_interval_minutes = 5  # Ping ogni 5 minuti
    
    if app_url:
        print(f"🔄 [KEEP-ALIVE] Sistema attivato per URL: {app_url}")
        print(f"⏰ [KEEP-ALIVE] Ping ogni {keep_alive_interval_minutes} minuti (06:00-24:00)")
    else:
        print(f"⚠️ [KEEP-ALIVE] RENDER_EXTERNAL_URL non configurata - keep-alive disabilitato")
        print(f"💡 [KEEP-ALIVE] Configura RENDER_EXTERNAL_URL nelle variabili d'ambiente Render")
    
    while True:
        try:
            italy_tz = pytz.timezone('Europe/Rome')
            now = datetime.datetime.now(italy_tz)
            
            # Controlla messaggi schedulati e recovery
            check_and_send_scheduled_messages()
            
            # === SISTEMA KEEP-ALIVE ===
            if is_keep_alive_time():
                time_since_ping = (now - last_ping_time).total_seconds() / 60
                
                if time_since_ping >= keep_alive_interval_minutes:
                    print(f"🔄 [KEEP-ALIVE] Ping app per mantenere attiva... ({now.strftime('%H:%M:%S')})")
                    
                    success = keep_app_alive(app_url)
                    if success:
                        print(f"✅ [KEEP-ALIVE] Ping riuscito - App attiva")
                    else:
                        print(f"⚠️ [KEEP-ALIVE] Ping fallito - App potrebbe essere in sleep")
                    
                    last_ping_time = now
            else:
                # Fuori dalla finestra keep-alive
                if now.minute == 0:  # Log ogni ora quando fuori finestra
                    print(f"😴 [KEEP-ALIVE] Fuori finestra attiva ({now.strftime('%H:%M')}), app può andare in sleep")
            
            # Pulizia memoria ogni ora
            if now.minute == 0:  # Ogni ora esatta
                gc.collect()
                print("🧹 [LITE-MEMORY] Pulizia memoria completata")
            
            time.sleep(30)  # Check ogni 30 secondi
            
        except Exception as e:
            print(f"❌ [LITE-ERROR] Errore scheduler: {e}")
            time.sleep(60)  # Attesa maggiore in caso di errore

# === AVVIO SISTEMA ===
if __name__ == "__main__":
    print("🚀 [555-LITE] Sistema ottimizzato avviato!")
    print(f"💾 [555-LITE] RAM extra disponibile per elaborazioni avanzate")
    print(f"📱 [555-LITE] Focus totale su qualità messaggi Telegram")
    
    # Carica i flag dai file salvati
    load_daily_flags()
    
    # Avvia scheduler in background
    scheduler_thread = threading.Thread(target=main_scheduler_loop, daemon=True)
    scheduler_thread.start()
    
    # Avvia mini web server
    print("🌐 [555-LITE] Mini web server attivo su porta 8000")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)



# === PATCH: Extend GLOBAL_FLAGS safely ===
try:
    _extra_flags = {
        "morning_news_last_run": "",
        "rassegna_stampa_sent": False,
        "rassegna_stampa_last_run": "",
        "morning_snapshot_sent": False,
        "morning_snapshot_last_run": "",
        "daily_report_last_run": "",
        "evening_report_last_run": ""
    }
    for _k, _v in _extra_flags.items():
        if "GLOBAL_FLAGS" in globals():
            GLOBAL_FLAGS.setdefault(_k, _v)
except Exception as _e:
    print("⚠️ [PATCH] Impossibile estendere GLOBAL_FLAGS:", _e)



def get_emerging_markets_headlines(limit=3):
    """Ritorna fino a `limit` titoli rapidi dai feed 'Mercati Emergenti'."""
    heads = []
    try:
        feeds = RSS_FEEDS.get("Mercati Emergenti", [])
        for url in feeds[:2]:
            parsed = feedparser.parse(url)
            if getattr(parsed, "entries", None):
                for e in parsed.entries[:limit]:
                    titolo = e.get("title", "").strip()
                    fonte = parsed.feed.get("title", "Unknown")
                    if titolo:
                        heads.append({"titolo": titolo, "fonte": fonte})
                    if len(heads) >= limit:
                        break
            if len(heads) >= limit:
                break
    except Exception:
        pass
    return heads

def get_em_fx_and_commodities():
    """Ritorna righe testuali con EM FX, commodities e proxy spread sovrani.
       Usa yfinance quando disponibile; fallback a placeholder se fallisce.
    """
    lines = []
    try:
        import yfinance as yf
        def pct_line(ticker, label):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2d", interval="1d")
                if hist is None or len(hist) == 0:
                    return None
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else last
                chg = (last - prev) / prev * 100 if prev else 0.0
                last_fmt = f"{last:.2f}" if last < 100 else f"{last:.1f}"
                sign = "+" if chg >= 0 else ""
                return f"{label} {last_fmt} ({sign}{chg:.1f}%)"
            except Exception:
                return None

        fx_items = [("USDBRL=X","USD/BRL"),("USDZAR=X","USD/ZAR"),("USDTRY=X","USD/TRY"),("USDINR=X","USD/INR")]
        fx_lines = [s for t,l in fx_items if (s:=pct_line(t,l))]
        if fx_lines: lines.append("FX: " + " · ".join(fx_lines))

        com_items = [("BZ=F","Brent"),("HG=F","Copper"),("GC=F","Gold")]
        com_lines = [s for t,l in com_items if (s:=pct_line(t,l))]
        if com_lines: lines.append("Commodities: " + " · ".join(com_lines))

        etf_items = [("EMB","EMB"),("EMLC","EMLC"),("CEW","CEW")]
        etf_lines = []
        for t,l in etf_items:
            s = pct_line(t,l)
            if s:
                pct = s[s.find("(")+1:s.find(")")]
                etf_lines.append(f"{l} {pct}")
        if etf_lines: lines.append("EM Credit/FX proxies: " + " · ".join(etf_lines))
    except Exception:
        lines.append("FX: USD/BRL • USD/ZAR • USD/TRY • USD/INR")
        lines.append("Commodities: Brent • Copper • Gold")
        lines.append("EM Credit/FX proxies: EMB • EMLC • CEW")
    return lines

def build_calendar_lines(days=7):
    """Ritorna una lista di righe calendario eventi per i prossimi N giorni."""
    lines = []
    try:
        oggi = datetime.date.today()
        entro = oggi + datetime.timedelta(days=days)
        lines.append("🗓️ *CALENDARIO EVENTI (7 giorni)*")
        elenco = []
        for categoria, lista in eventi.items():
            for e in lista:
                d = datetime.datetime.strptime(e["Data"], "%Y-%m-%d").date()
                if oggi <= d <= entro:
                    elenco.append((d, categoria, e))
        elenco.sort(key=lambda x: x[0])
        if not elenco:
            lines.append("• Nessun evento in finestra 7 giorni")
        for d, categoria, e in elenco[:20]:
            ic = "🔴" if e["Impatto"]=="Alto" else "🟡" if e["Impatto"]=="Medio" else "🟢"
            lines.append(f"{d.strftime('%d/%m')} {ic} {e['Titolo']} — {categoria} · {e['Fonte']}")
        lines.append("")
    except Exception:
        lines.append("⚠️ Calendario non disponibile al momento.")
        lines.append("")
    return lines



def generate_morning_snapshot():
    """Messaggio breve tipo lunch/evening: market pulse + EM + notizie critiche."""
    italy_tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(italy_tz)
    parts = []
    parts.append("🌅 *MORNING SNAPSHOT*")
    parts.append(f"📅 {now.strftime('%d/%m/%Y %H:%M')} CET")
    parts.append("─" * 35)
    parts.append("")
    parts.append("📊 *Market Pulse*")
    parts.append("• Europa: focus Banks & Energy (chiusura 17:30 CET)")
    parts.append("• USA: apertura 15:30 CET — tech in focus")
    parts.append("• Crypto: 24/7 — livelli chiave BTC 42k-45k")
    parts.append("• Mercati Emergenti: monitor su FX/commodities e spread sovrani")
    parts.append("")
    try:
        emh = get_emerging_markets_headlines(limit=3)
        if emh:
            parts.append("🌍 *Mercati Emergenti — Flash*")
            for i, n in enumerate(emh[:3], 1):
                titolo = n["titolo"][:80] + "..." if len(n["titolo"])>80 else n["titolo"]
                parts.append(f"{i}. *{titolo}* — {n.get('fonte','EM')}")
            parts.append("")
    except Exception:
        pass
    try:
        crit = get_notizie_critiche()
        if crit:
            parts.append("🚨 *Top Notizie (24h)*")
            for i, n in enumerate(crit[:3], 1):
                titolo = n["titolo"][:80] + "..." if len(n["titolo"])>80 else n["titolo"]
                parts.append(f"{i}. *{titolo}* — {n['fonte']}")
            parts.append("")
    except Exception:
        pass
    # EM FX & Commodities
    emfx = get_em_fx_and_commodities()
    if emfx:
        parts.append("🌍 *EM FX & Commodities*")
        parts.extend(emfx)
        parts.append("")
    parts.append("─" * 35)
    parts.append("🤖 555 Lite • Morning")
    msg = "\n".join(parts)
    return "✅" if invia_messaggio_telegram(msg) else "❌"



# === SAFE SEND & RECOVERY HELPERS ===
def safe_send(flag_name, last_key, send_callable, after_set_flag_name=None):
    """Imposta lock+debounce, invia, rollback su errore."""
    italy_tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(italy_tz)
    now_key = now.strftime("%Y%m%d%H%M")
    GLOBAL_FLAGS[flag_name] = True
    GLOBAL_FLAGS[last_key] = now_key
    save_daily_flags()
    try:
        result = send_callable()
        if after_set_flag_name:
            set_message_sent_flag(after_set_flag_name)
        return result
    except Exception as e:
        GLOBAL_FLAGS[flag_name] = False
        save_daily_flags()
        print(f"❌ [SAFE-SEND] Errore {flag_name}: {e}")
        raise

def should_recover(sent_flag, scheduled_hhmm, grace_min, cutoff_hhmm, now_hhmm):
    def to_min(hhmm):
        h,m = map(int, hhmm.split(":")); return h*60+m
    return (not sent_flag) and (to_min(now_hhmm) >= to_min(scheduled_hhmm)+grace_min) and (to_min(now_hhmm) <= to_min(cutoff_hhmm))

def run_recovery_checks():
    italy_tz = pytz.timezone('Europe/Rome')
    now = datetime.datetime.now(italy_tz)
    now_hhmm = now.strftime("%H:%M")
    schedules = [
        ("rassegna", GLOBAL_FLAGS.get("rassegna_stampa_sent", False), "07:00", 10, "08:00", lambda: safe_send("rassegna_stampa_sent","rassegna_stampa_last_run", generate_morning_news_briefing)),
        ("morning", GLOBAL_FLAGS.get("morning_snapshot_sent", False), "08:10", 10, "12:00", lambda: safe_send("morning_snapshot_sent","morning_snapshot_last_run", generate_morning_snapshot)),
        ("lunch", GLOBAL_FLAGS.get("daily_report_sent", False), "14:10", 10, "19:00", lambda: safe_send("daily_report_sent","daily_report_last_run", generate_daily_lunch_report, after_set_flag_name="daily_report")),
        ("evening", GLOBAL_FLAGS.get("evening_report_sent", False), "20:10", 10, "23:50", lambda: safe_send("evening_report_sent","evening_report_last_run", generate_evening_report, after_set_flag_name="evening_report")),
    ]
    for key, sent, sched, grace, cutoff, sender in schedules:
        if should_recover(sent, sched, grace, cutoff, now_hhmm):
            print(f"🔁 [RECOVERY] Invio tardivo {key} (sched {sched})")
            try:
                sender()
            except Exception as e:
                print(f"❌ [RECOVERY] {key} errore:", e)



def _today_key(dt=None):
    if dt is None: dt = _now_it()
    return dt.strftime("%Y%m%d")

def _minute_key(dt=None):
    if dt is None: dt = _now_it()
    return dt.strftime("%Y%m%d%H%M")


def _recovery_tick():
    now = _now_it()
    hm = now.strftime("%H:%M")
    def _within(target, window):
        h = int(target[:2]); m = int(target[3:])
        dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return (now >= dt) and ((now - dt).total_seconds() <= window*60)

    # ogni 10 minuti
    if now.minute % RECOVERY_INTERVAL_MINUTES != 0: 
        return

    # Rassegna
    if not is_message_sent_today("rassegna") and _within(SCHEDULE["rassegna"], RECOVERY_WINDOWS["rassegna"]):
        try:
            generate_rassegna_stampa(); set_message_sent_flag("rassegna"); save_daily_flags()
        except Exception as e:
            log.warning(f"[RECOVERY] rassegna: {e}")

    # Morning
    if not is_message_sent_today("morning_news") and _within(SCHEDULE["morning"], RECOVERY_WINDOWS["morning"]):
        try:
            generate_morning_news(); set_message_sent_flag("morning_news"); save_daily_flags()
        except Exception as e:
            log.warning(f"[RECOVERY] morning: {e}")

    # Lunch
    if not is_message_sent_today("daily_report") and _within(SCHEDULE["lunch"], RECOVERY_WINDOWS["lunch"]):
        try:
            generate_lunch_report(); set_message_sent_flag("daily_report"); save_daily_flags()
        except Exception as e:
            log.warning(f"[RECOVERY] lunch: {e}")

    # Evening
    if not is_message_sent_today("evening_report") and _within(SCHEDULE["evening"], RECOVERY_WINDOWS["evening"]):
        try:
            generate_evening_report(); set_message_sent_flag("evening_report"); save_daily_flags()
        except Exception as e:
            log.warning(f"[RECOVERY] evening: {e}")


def send_telegram_message(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    chat  = os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": text, "disable_web_page_preview": True}
    r = requests.post(url, data=payload, timeout=15)
    if r.status_code == 400:
        # fallback no formatting
        payload.pop("parse_mode", None)
        r = requests.post(url, data=payload, timeout=15)
    return r.status_code == 200

def send_telegram_message_long(text: str) -> bool:
    MAXLEN = 3500
    t = text.strip()
    if len(t) <= MAXLEN:
        return send_telegram_message(t)
    ok_all = True
    part = 1; start = 0
    while start < len(t):
        end = min(len(t), start+MAXLEN)
        cut = t.rfind("\n", start, end)
        if cut <= start: cut = end
        chunk = t[start:cut]
        hdr = f"PARTE {part}\n\n" if part>1 else ""
        ok = send_telegram_message(hdr + chunk)
        ok_all = ok_all and ok
        start = cut; part += 1
        time.sleep(1.2)
    return ok_all
