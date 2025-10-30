#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test per verificare la consistenza completa del sistema di scheduling
555 Lite - Test scheduling integrity
"""

import json
import os
import datetime
import sys
import re
from pathlib import Path

# Aggiungi il path del progetto per importare le funzioni
sys.path.append(str(Path(__file__).parent))

def test_scheduling_consistency():
    """Test per verificare la consistenza del sistema di scheduling"""
    print("🧪 [TEST] Avvio test consistenza scheduling")
    print("=" * 60)
    
    try:
        # Importa costanti dal modulo principale
        import importlib.util
        spec = importlib.util.spec_from_file_location("serverlite", "555-serverlite.py")
        serverlite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serverlite)
        
        SCHEDULE = serverlite.SCHEDULE
        WEEKEND_SCHEDULE = serverlite.WEEKEND_SCHEDULE
        RECOVERY_WINDOWS = serverlite.RECOVERY_WINDOWS
        GLOBAL_FLAGS = serverlite.GLOBAL_FLAGS
        
        print("✅ [TEST] Modulo serverlite caricato con successo")
        
        # Test 1: Verifica consistenza SCHEDULE
        print(f"\n📋 [TEST] Test 1 - Verifica SCHEDULE principal:")
        for event, time in SCHEDULE.items():
            print(f"  ✅ {event}: {time}")
            
            # Verifica formato orario
            if not re.match(r"^\d{2}:\d{2}$", time):
                print(f"  ❌ ERRORE: Formato orario invalido per {event}: {time}")
                return False
        
        # Test 2: Verifica WEEKEND_SCHEDULE
        print(f"\n🏖️ [TEST] Test 2 - Verifica Weekend Schedule:")
        for i, time in enumerate(WEEKEND_SCHEDULE, 1):
            print(f"  ✅ Slot {i}: {time}")
            if not re.match(r"^\d{2}:\d{2}$", time):
                print(f"  ❌ ERRORE: Formato orario invalido weekend: {time}")
                return False
        
        # Test 3: Verifica RECOVERY_WINDOWS
        print(f"\n🔄 [TEST] Test 3 - Verifica Recovery Windows:")
        for event in SCHEDULE.keys():
            if event in RECOVERY_WINDOWS:
                window = RECOVERY_WINDOWS[event]
                print(f"  ✅ {event}: {window} minuti")
            else:
                print(f"  ❌ ERRORE: Recovery window mancante per {event}")
                return False
        
        # Test 4: Verifica mapping flag
        print(f"\n🏴 [TEST] Test 4 - Verifica mapping eventi-flag:")
        flag_mapping_expected = {
            "rassegna": "rassegna_sent",
            "morning": "morning_news_sent", 
            "lunch": "daily_report_sent",
            "evening": "evening_report_sent",
            "daily_summary": "daily_summary_sent"
        }
        
        get_flag_name_for_event = serverlite.get_flag_name_for_event
        
        for event in SCHEDULE.keys():
            expected_flag = flag_mapping_expected[event]
            mapped_flag_key = get_flag_name_for_event(event)
            actual_flag_key = f"{mapped_flag_key}_sent" if not mapped_flag_key.endswith("_sent") else mapped_flag_key
            
            if expected_flag == actual_flag_key:
                print(f"  ✅ {event} -> {actual_flag_key}")
            else:
                print(f"  ❌ ERRORE: {event} -> expected {expected_flag}, got {actual_flag_key}")
                return False
        
        # Test 5: Verifica esistenza flag in GLOBAL_FLAGS
        print(f"\n📊 [TEST] Test 5 - Verifica esistenza flag in GLOBAL_FLAGS:")
        for event, expected_flag in flag_mapping_expected.items():
            if expected_flag in GLOBAL_FLAGS:
                print(f"  ✅ {expected_flag}: {GLOBAL_FLAGS[expected_flag]}")
            else:
                print(f"  ❌ ERRORE: Flag {expected_flag} mancante in GLOBAL_FLAGS")
                return False
        
        # Test 6: Verifica ordine cronologico
        print(f"\n⏰ [TEST] Test 6 - Verifica ordine cronologico SCHEDULE:")
        times = [(event, time) for event, time in SCHEDULE.items()]
        sorted_times = sorted(times, key=lambda x: x[1])
        
        expected_order = ["rassegna", "morning", "lunch", "evening", "daily_summary"]
        actual_order = [event for event, time in sorted_times]
        
        if expected_order == actual_order:
            print(f"  ✅ Ordine corretto: {' -> '.join(actual_order)}")
        else:
            print(f"  ❌ ERRORE: Ordine sbagliato")
            print(f"     Expected: {' -> '.join(expected_order)}")
            print(f"     Actual:   {' -> '.join(actual_order)}")
            return False
        
        # Test 7: Verifica weekend schedule è in ordine
        print(f"\n🏖️ [TEST] Test 7 - Verifica ordine Weekend Schedule:")
        weekend_sorted = sorted(WEEKEND_SCHEDULE)
        if WEEKEND_SCHEDULE == weekend_sorted:
            print(f"  ✅ Ordine weekend corretto: {' -> '.join(WEEKEND_SCHEDULE)}")
        else:
            print(f"  ❌ ERRORE: Weekend schedule non in ordine")
            print(f"     Actual:   {' -> '.join(WEEKEND_SCHEDULE)}")
            print(f"     Expected: {' -> '.join(weekend_sorted)}")
            return False
        
        # Test 8: Verifica no sovrapposizioni temporali
        print(f"\n🚫 [TEST] Test 8 - Verifica no sovrapposizioni:")
        all_times = list(SCHEDULE.values()) + WEEKEND_SCHEDULE
        unique_times = set(all_times)
        
        if len(all_times) == len(unique_times):
            print(f"  ✅ Nessuna sovrapposizione trovata")
        else:
            duplicates = [t for t in all_times if all_times.count(t) > 1]
            print(f"  ❌ ERRORE: Sovrapposizioni trovate: {duplicates}")
            return False
        
        # Test 9: Test timing calculation
        print(f"\n🧮 [TEST] Test 9 - Test calcolo timing:")
        now = datetime.datetime.now()
        lunch_time_parts = SCHEDULE["lunch"].split(":")
        lunch_hour = int(lunch_time_parts[0]) 
        lunch_minute = int(lunch_time_parts[1])
        
        print(f"  ✅ Lunch parsing: {SCHEDULE['lunch']} -> {lunch_hour:02d}:{lunch_minute:02d}")
        print(f"  ✅ Ora corrente: {now.strftime('%H:%M')}")
        
        # Test 10: Verifica flag reset date format
        print(f"\n📅 [TEST] Test 10 - Verifica formato last_reset_date:")
        last_reset = GLOBAL_FLAGS.get("last_reset_date", "")
        if re.match(r"^\d{8}$", last_reset):
            print(f"  ✅ Formato data reset corretto: {last_reset}")
        else:
            print(f"  ❌ ERRORE: Formato data reset invalido: {last_reset}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ [TEST] Tutti i test di consistenza scheduling SUPERATI!")
        
        # Riepilogo finale
        print(f"\n📊 [RIEPILOGO SCHEDULING]:")
        print(f"  • Schedule Events: {len(SCHEDULE)} eventi")
        print(f"  • Weekend Slots: {len(WEEKEND_SCHEDULE)} slot")
        print(f"  • Recovery Windows: {len(RECOVERY_WINDOWS)} finestre")
        print(f"  • Daily Flags: {len([k for k in GLOBAL_FLAGS.keys() if k.endswith('_sent')])} flag")
        print(f"  • Timeline: {SCHEDULE['rassegna']} → {SCHEDULE['daily_summary']}")
        print(f"  • Weekend: {WEEKEND_SCHEDULE[0]} → {WEEKEND_SCHEDULE[-1]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ [TEST] Errore import: {e}")
        return False
    except Exception as e:
        print(f"❌ [TEST] Errore generale: {e}")
        print(f"❌ [TEST] Tipo errore: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_scheduling_consistency()
    if success:
        print("\n🚀 [RESULT] Sistema di scheduling CONSISTENTE e pronto per deploy!")
    else:
        print("\n💥 [RESULT] Sistema di scheduling ha PROBLEMI che devono essere risolti!")
    
    sys.exit(0 if success else 1)