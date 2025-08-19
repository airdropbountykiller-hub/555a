# === CONFIGURAZIONE OTTIMIZZAZIONI PERFORMANCE CIAO4 ===
# Gestione centralizzata di tutte le ottimizzazioni per velocizzare l'applicazione

import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# === CACHE E MEMORIA OTTIMIZZATA ===
PERFORMANCE_CONFIG = {
    # Cache più aggressiva
    "cache_duration_minutes": 90,  # Era 30, ora 90 minuti per ridurre API calls
    "max_cache_size": 100,         # Era 20, ora 100 elementi
    
    # Parallelizzazione avanzata
    "max_workers": 6,              # Thread pool per operazioni parallele
    "enable_parallel_ml": True,    # Abilita calcoli ML paralleli
    "enable_parallel_indicators": True,  # Abilita calcoli indicatori paralleli
    "enable_parallel_data_loading": True, # Carica dati in parallelo
    
    # Limiti per velocizzazione drastica
    "max_models_for_telegram": 4,  # Solo 4 modelli migliori per Telegram
    "max_indicators_dashboard": 8, # Solo 8 indicatori principali per dashboard
    "max_news_per_source": 5,     # Max 5 notizie per fonte RSS
    
    # Dataset ottimizzazioni aggressive
    "max_datapoints_ml": 500,     # Dimezzato: era 1000, ora 500 per velocità
    "min_datapoints_required": 50, # Dimezzato: era 100, ora 50
    "historical_days_limit": 365,  # Limita a 1 anno di dati storici
    
    # Scheduler ottimizzazioni super-rapide
    "scheduler_sleep_time": 45,    # Era 30, ora 45 secondi tra check
    "enable_smart_caching": True,  # Cache intelligente con pre-calcolo
    "precompute_before_reports": True,  # Pre-calcola tutto prima dei report
}

# === MODELLI ML SUPER-VELOCI (TOP 4) ===
LIGHTNING_ML_MODELS = [
    "Random Forest",      # Veloce e accurato - SEMPRE INCLUSO
    "Logistic Regression", # Velocissimo - SEMPRE INCLUSO  
    "Gradient Boosting",  # Buon compromesso velocità/accuratezza
    "Naive Bayes",       # Velocissimo per calcoli rapidi
]

# === MODELLI COMPLETI PER EXPORT CSV ===
FULL_ML_MODELS = [
    "Random Forest", "Logistic Regression", "Gradient Boosting", 
    "XGBoost", "Naive Bayes", "K-Nearest Neighbors"
]

# === INDICATORI ESSENZIALI (TOP 5 PIÙ IMPORTANTI) ===
CORE_INDICATORS = [
    "MAC",        # Moving Average Convergence - trend principale
    "RSI",        # Relative Strength Index - momentum
    "MACD",       # MACD - trend e momentum
    "Bollinger",  # Bollinger Bands - volatilità
    "EMA",        # Exponential Moving Average - trend veloce
]

# === INDICATORI SECONDARI (PER DASHBOARD COMPLETA) ===
SECONDARY_INDICATORS = [
    "SMA",        # Simple Moving Average
    "Stochastic", # Stochastic Oscillator
    "ATR",        # Average True Range
]

# === RSS FEEDS SUPER-OTTIMIZZATI (1 PER CATEGORIA) ===
ULTRA_FAST_RSS_FEEDS = {
    "Finanza": [
        "https://feeds.reuters.com/reuters/businessNews"  # Solo Reuters - il più veloce
    ],
    "Criptovalute": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/" # Solo CoinDesk - standard veloce
    ],
    "Geopolitica": [
        "https://feeds.reuters.com/Reuters/worldNews"     # Solo Reuters World - veloce
    ]
}

# === TIMEOUTS AGGRESSIVI ===
SPEED_TIMEOUTS = {
    "http_request_timeout": 8,     # Era 10, ora 8 secondi
    "rss_feed_timeout": 6,         # Era 8, ora 6 secondi
    "ml_training_timeout": 20,     # Era 30, ora 20 secondi
    "crypto_api_timeout": 10,      # Era 12, ora 10 secondi
    "total_report_timeout": 120,   # Timeout totale per generazione report
}

# === DECORATORI PER PERFORMANCE ===
def timed_execution(func_name="Function"):
    """Decoratore per misurare tempi di esecuzione"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"⚡ [{func_name}] Completato in {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator

def cached_with_expiry(minutes=90):
    """Decoratore per cache con scadenza personalizzata"""
    def decorator(func):
        cache = {}
        cache_timestamps = {}
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            # Controlla se in cache e non scaduto
            if (key in cache and 
                key in cache_timestamps and 
                now - cache_timestamps[key] < minutes * 60):
                print(f"🚀 [CACHE-HIT] {func.__name__}")
                return cache[key]
            
            # Calcola e salva in cache
            result = func(*args, **kwargs)
            cache[key] = result
            cache_timestamps[key] = now
            print(f"💾 [CACHE-MISS] {func.__name__} - Salvato in cache")
            return result
        return wrapper
    return decorator

# === GESTIONE THREAD POOL GLOBALE ===
_thread_pool = None
_thread_pool_lock = threading.Lock()

def get_thread_pool():
    """Ottieni il thread pool globale per operazioni parallele"""
    global _thread_pool
    with _thread_pool_lock:
        if _thread_pool is None:
            _thread_pool = ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG["max_workers"])
            print(f"🧵 [THREADING] Thread pool inizializzato con {PERFORMANCE_CONFIG['max_workers']} workers")
        return _thread_pool

def parallel_execute(functions_with_args):
    """Esegue funzioni in parallelo e restituisce i risultati"""
    thread_pool = get_thread_pool()
    futures = []
    
    for func, args, kwargs in functions_with_args:
        future = thread_pool.submit(func, *args, **kwargs)
        futures.append((func.__name__, future))
    
    results = {}
    for name, future in futures:
        try:
            results[name] = future.result(timeout=SPEED_TIMEOUTS["total_report_timeout"])
        except Exception as e:
            print(f"❌ [PARALLEL] Errore in {name}: {e}")
            results[name] = None
    
    return results

# === FLAGS DEBUG PERFORMANCE ===
DEBUG_PERFORMANCE = {
    "enable_timing_logs": True,    # Mostra tempi di esecuzione dettagliati
    "enable_memory_logs": False,   # Disabilita per velocità (attivare solo se necessario)
    "enable_cache_stats": True,    # Statistiche cache hit/miss
    "enable_parallel_logs": True,  # Log delle operazioni parallele
}

print("✅ [PERFORMANCE] Configurazione caricata - Modalità velocità attiva!")
