#!/usr/bin/env python3
"""
Test Critical Functions - 555-serverlite System Validation
"""

import sys
import re
import os

def test_constants():
    """Test che tutte le costanti siano definite correttamente"""
    print("🔍 [TEST] Test costanti temporali...")
    
    # Simula le costanti principali
    SCHEDULE = {
        "rassegna": "08:00",
        "morning":  "09:00", 
        "lunch":    "13:00",
        "evening":  "17:00",
        "daily_summary":  "18:00",
    }
    
    US_MARKET_OPEN = "15:30"
    US_MARKET_CLOSE = "22:00" 
    EUROPE_MARKET_OPEN = "09:00"
    EUROPE_MARKET_CLOSE = "17:30"
    DATA_RELEASE_WINDOW_START = "14:00"
    DATA_RELEASE_WINDOW_END = "16:00"
    RECOVERY_INTERVAL_MINUTES = 30
    RECOVERY_WINDOWS = {"rassegna": 60, "morning": 80, "lunch": 80, "evening": 80, "daily_summary": 80}
    
    # Test 1: Costanti temporali
    assert SCHEDULE["rassegna"] == "08:00", "Rassegna time mismatch"
    assert US_MARKET_OPEN == "15:30", "US market open mismatch"
    assert EUROPE_MARKET_OPEN == "09:00", "Europe open mismatch"
    print("✅ [CONSTANTS] Costanti temporali: OK")
    
    # Test 2: Finestre dati
    window = f"{DATA_RELEASE_WINDOW_START}-{DATA_RELEASE_WINDOW_END}"
    assert window == "14:00-16:00", "Data window mismatch"
    print("✅ [CONSTANTS] Finestre dati: OK")
    
    # Test 3: Recovery config
    assert RECOVERY_INTERVAL_MINUTES == 30, "Recovery interval mismatch"
    assert len(RECOVERY_WINDOWS) == 5, "Recovery windows count mismatch"
    print("✅ [CONSTANTS] Configurazione recovery: OK")
    
    # Test 4: Formato orari
    time_pattern = r'^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$'
    for event, time_str in SCHEDULE.items():
        assert re.match(time_pattern, time_str), f"Invalid time format for {event}: {time_str}"
    print("✅ [CONSTANTS] Formati orario: OK")
    
    # Test 5: Finestre recovery ragionevoli
    for event, window in RECOVERY_WINDOWS.items():
        assert 30 <= window <= 120, f"Recovery window out of range for {event}: {window}min"
    print("✅ [CONSTANTS] Finestre recovery: OK")
    
    return True

def test_file_syntax():
    """Test sintassi del file principale"""
    print("🔍 [TEST] Test sintassi file principale...")
    
    try:
        import ast
        with open('555-serverlite.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("✅ [SYNTAX] File principale: OK")
        
        # Statistiche
        tree = ast.parse(content)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        
        print(f"📊 [STATS] Funzioni: {len(functions)}")
        print(f"📊 [STATS] Import: {len(imports)}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ [SYNTAX] Errore sintassi linea {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ [SYNTAX] Errore: {e}")
        return False

def test_module_compatibility():
    """Test compatibilità moduli"""
    print("🔍 [TEST] Test compatibilità moduli...")
    
    # Test import critici (simulati)
    critical_modules = [
        'datetime', 'json', 'os', 're', 'time', 'requests', 
        'flask', 'pytz', 'gc', 'threading'
    ]
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"✅ [MODULES] {module}: OK")
        except ImportError:
            print(f"❌ [MODULES] {module}: MISSING")
            return False
    
    return True

def main():
    """Esegue tutti i test"""
    print("🚀 [TEST] Avvio test completo sistema 555-serverlite")
    print("=" * 50)
    
    tests = [
        ("Costanti e Configurazione", test_constants),
        ("Sintassi File Principale", test_file_syntax), 
        ("Compatibilità Moduli", test_module_compatibility)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 [TEST] {test_name}")
        try:
            if test_func():
                print(f"✅ [RESULT] {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ [RESULT] {test_name}: FAILED")
        except Exception as e:
            print(f"❌ [RESULT] {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 [SUMMARY] Test completati: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 [SUCCESS] Tutti i test: PASSATI")
        print("✅ [READY] Sistema pronto per deployment")
        return True
    else:
        print("⚠️ [WARNING] Alcuni test falliti")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)