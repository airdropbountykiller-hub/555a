#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script per verificare il corretto funzionamento del reset dei flag giornalieri
555 Lite - Sistema di test per flag reset
"""

import json
import os
import datetime
from pathlib import Path

# Percorso del file flags
FLAGS_FILE = Path("salvataggi/daily_flags.json")

def test_flag_reset_system():
    """Test completo del sistema di reset dei flag"""
    print("🧪 [TEST] Avvio test sistema reset flag giornalieri")
    print("=" * 60)
    
    # 1. Leggi stato attuale
    if FLAGS_FILE.exists():
        with open(FLAGS_FILE, 'r', encoding='utf-8') as f:
            current_flags = json.load(f)
        
        print("📋 [TEST] Stato attuale flag:")
        for key, value in current_flags.items():
            status_emoji = "✅" if value else "❌" if key.endswith('_sent') else "📅"
            print(f"  {status_emoji} {key}: {value}")
        print()
        
        # 2. Verifica data reset
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        saved_date = current_flags.get("last_reset_date", "")
        
        print(f"📅 [TEST] Data corrente: {current_date}")
        print(f"📅 [TEST] Data salvata:  {saved_date}")
        
        if saved_date == current_date:
            print("✅ [TEST] Date corrispondono - flag validi per oggi")
        else:
            print("⚠️ [TEST] Date NON corrispondono - necessario reset!")
            
            # 3. Simula reset completo
            print("\n🔄 [TEST] Simulazione reset completo...")
            reset_flags = {
                "rassegna_sent": False,
                "morning_news_sent": False,
                "daily_report_sent": False,
                "evening_report_sent": False,
                "daily_summary_sent": False,
                "weekly_report_sent": False,
                "monthly_report_sent": False,
                "quarterly_report_sent": False,
                "semestral_report_sent": False,
                "annual_report_sent": False,
                "last_reset_date": current_date
            }
            
            # Backup del file originale
            backup_file = FLAGS_FILE.with_suffix('.backup.json')
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(current_flags, f, indent=2, ensure_ascii=False)
            print(f"💾 [TEST] Backup creato: {backup_file}")
            
            # Salva nuovi flag
            with open(FLAGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(reset_flags, f, indent=2, ensure_ascii=False)
            
            print("✅ [TEST] Flag resettati con successo!")
            print("\n📋 [TEST] Nuovo stato flag:")
            for key, value in reset_flags.items():
                status_emoji = "✅" if value else "❌" if key.endswith('_sent') else "📅"
                print(f"  {status_emoji} {key}: {value}")
            
        # 4. Verifica scheduling
        print("\n🕐 [TEST] Verifica orari di scheduling:")
        schedule_times = {
            "rassegna": "08:00",
            "morning": "09:00", 
            "lunch": "13:00",
            "evening": "17:00",
            "daily_summary": "18:00"
        }
        
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        
        print(f"⏰ [TEST] Ora attuale: {current_time}")
        for event, time in schedule_times.items():
            status = "🟢 PROSSIMO" if current_time < time else "🔴 PASSATO"
            print(f"  {status} {event.upper()}: {time}")
            
        # 5. Test continuità narrativa
        print(f"\n📖 [TEST] Verifica continuità narrativa:")
        narrative_file = Path("salvataggi/narrative_continuity.json")
        if narrative_file.exists():
            with open(narrative_file, 'r', encoding='utf-8') as f:
                narrative_data = json.load(f)
            print(f"✅ [TEST] File narrative_continuity.json presente")
            print(f"📊 [TEST] Ultimo update: {narrative_data.get('last_update', 'N/A')}")
            print(f"🎯 [TEST] Focus mattutino: {len(narrative_data.get('morning_focus', {}).get('focus_items', []))} elementi")
        else:
            print("⚠️ [TEST] File narrative_continuity.json NON trovato")
            
        print("\n" + "=" * 60)
        print("✅ [TEST] Test completato con successo!")
        
    else:
        print("❌ [TEST] File daily_flags.json non trovato!")
        return False
    
    return True

if __name__ == "__main__":
    test_flag_reset_system()