# 🚨 SCHEDULER CONFLICTS ANALYSIS

## ❌ CONFLITTI IDENTIFICATI

### **1. DUPLICAZIONE COSTANTI ORARI**

**PROBLEMA PRINCIPALE:**
```python
# SCHEDULE Dictionary
SCHEDULE = {
    "rassegna": "08:00",     # ⚠️ DUPLICATO
    "morning": "09:00",      # ⚠️ DUPLICATO  
    "evening": "17:00",      # ⚠️ CONFLITTO con market close
}

# Market Timing Constants
PRESS_REVIEW_TIME = "08:00"     # ⚠️ STESSO VALORE
EUROPE_MARKET_OPEN = "09:00"    # ⚠️ STESSO VALORE
EUROPE_MARKET_CLOSE = "17:30"   # ⚠️ DIVERSO da evening "17:00"
```

### **2. HARDCODED VALUES in FUNCTIONS**

**CONFLITTO:** Le funzioni usano ancora valori hardcoded invece delle costanti:

```python
# ❌ PROBLEMA in is_market_hours()
def is_market_hours():
    market_open = now.replace(hour=9, minute=0)      # HARDCODED!
    market_close = now.replace(hour=17, minute=30)   # HARDCODED!

# ❌ PROBLEMA in get_market_status()  
def get_market_status():
    if now.hour < 9:                                 # HARDCODED!
        return "PRE_MARKET"
```

### **3. TIMING LOGIC CONFLICT**

**❌ EVENING REPORT vs MARKET CLOSE:**
- Evening Report: `17:00` (da SCHEDULE)
- Market Close: `17:30` (da EUROPE_MARKET_CLOSE)
- **PROBLEMA:** Report parte 30 min prima della chiusura!

### **4. INCONSISTENT DATA STRUCTURES**

**❌ MULTIPLE SOURCES OF TRUTH:**
- `SCHEDULE` dictionary per scheduler automation
- Market constants per message content  
- Hardcoded values nelle funzioni
- Commenti con orari different

## 🔧 RACCOMANDAZIONI PER LA CORREZIONE

### **SOLUZIONE 1: CENTRALIZZAZIONE COMPLETA**

```python
# === SINGLE SOURCE OF TRUTH ===
MARKET_SCHEDULE = {
    "press_review": "08:00",
    "europe_open": "09:00", 
    "lunch_report": "13:00",
    "data_release_start": "14:00",
    "us_market_open": "15:30",
    "data_release_end": "16:00",
    "europe_close": "17:30",
    "evening_report": "17:30",    # ✅ ALIGN with market close
    "daily_summary": "18:00",
    "us_market_close": "22:00"
}

# Derive other constants from master schedule
PRESS_REVIEW_TIME = MARKET_SCHEDULE["press_review"]
US_MARKET_OPEN = MARKET_SCHEDULE["us_market_open"] 
# etc...
```

### **SOLUZIONE 2: FUNCTION REFACTORING**

```python
def is_market_hours():
    """Uses constants instead of hardcoded values"""
    now = _now_it()
    if is_weekend():
        return False
    
    # Parse market timing constants
    open_time = datetime.time.fromisoformat(EUROPE_MARKET_OPEN)
    close_time = datetime.time.fromisoformat(EUROPE_MARKET_CLOSE)
    
    market_open = now.replace(hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0)
    market_close = now.replace(hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0)
    
    return market_open <= now <= market_close
```

### **SOLUZIONE 3: EVENING REPORT TIMING FIX**

**OPZIONE A:** Spostare Evening Report dopo market close
```python
SCHEDULE = {"evening": "17:45"}  # 15 min after market close
```

**OPZIONE B:** Rinominare per chiarezza logica
```python
SCHEDULE = {"pre_close": "17:00", "evening": "18:30"}  # Separate reports
```

## 🎯 PRIORITÀ DI INTERVENTO

1. **HIGH:** Fix hardcoded values in is_market_hours() e get_market_status()
2. **HIGH:** Resolve evening report timing conflict (17:00 vs 17:30)  
3. **MEDIUM:** Centralizzare tutte le costanti temporali
4. **LOW:** Cleanup comments con orari obsoleti

## ✅ BENEFICI ATTESI

- **Consistency:** Single source of truth per timing
- **Maintainability:** Un solo posto da modificare per schedule changes
- **Logic Fix:** Evening report timing allineato con market close
- **Professional Code:** No more hardcoded magic numbers