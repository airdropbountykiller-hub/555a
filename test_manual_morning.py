#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test per forzare l'invio del morning report mancato
555 Lite - Test manuale per recupero morning report
"""

import json
import os
import datetime
import sys
from pathlib import Path

# Aggiungi il path del progetto per importare le funzioni
sys.path.append(str(Path(__file__).parent))

def test_morning_recovery():
    """Test per recuperare il morning report mancato"""
    print("🧪 [TEST-MANUAL] Avvio test recovery morning report")
    print("=" * 60)
    
    try:
        # Importa le funzioni necessarie
        import importlib.util
        spec = importlib.util.spec_from_file_location("serverlite", "555-serverlite.py")
        serverlite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serverlite)
        
        generate_morning_news = serverlite.generate_morning_news
        set_message_sent_flag = serverlite.set_message_sent_flag
        save_daily_flags = serverlite.save_daily_flags
        is_message_sent_today = serverlite.is_message_sent_today
        load_daily_flags = serverlite.load_daily_flags
        
        # Carica flag attuali
        load_daily_flags()
        
        # Verifica stato attuale morning
        morning_sent = is_message_sent_today("morning_news")
        print(f"📋 [TEST] Morning report già inviato oggi: {morning_sent}")
        
        if morning_sent:
            print("⚠️ [TEST] Morning report risulta già inviato")
            print("🔄 [TEST] Procedo comunque con test manuale...")
        
        # Test 1: Verifica funzionalità generate_morning_news
        print("\n🧪 [TEST] Test 1 - Verifica funzionalità generate_morning_news()")
        try:
            print("🌅 [TEST] Chiamando generate_morning_news()...")
            generate_morning_news()
            print("✅ [TEST] generate_morning_news() completata con successo!")
            
            # Aggiorna flag
            set_message_sent_flag("morning_news")
            print("✅ [TEST] Flag morning_news_sent impostato a True")
            
            # Salva flag
            save_daily_flags()
            print("✅ [TEST] Flag salvati su file")
            
        except Exception as e:
            print(f"❌ [TEST] Errore durante generate_morning_news(): {e}")
            print(f"❌ [TEST] Tipo errore: {type(e).__name__}")
            return False
        
        # Verifica finale
        print(f"\n📊 [TEST] Verifica finale:")
        morning_sent_after = is_message_sent_today("morning_news")
        print(f"📋 [TEST] Morning report inviato dopo test: {morning_sent_after}")
        
        # Mostra stato finale flag
        flags_file = Path("salvataggi/daily_flags.json")
        if flags_file.exists():
            with open(flags_file, 'r', encoding='utf-8') as f:
                final_flags = json.load(f)
            
            print("\n📋 [TEST] Stato finale flag:")
            for key, value in final_flags.items():
                if key.endswith('_sent') or key == 'last_reset_date':
                    status_emoji = "✅" if value else "❌" if key.endswith('_sent') else "📅"
                    print(f"  {status_emoji} {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✅ [TEST-MANUAL] Test completato con successo!")
        return True
        
    except ImportError as e:
        print(f"❌ [TEST] Errore import: {e}")
        print("❌ [TEST] Verifica che il file 555-serverlite.py sia presente e corretto")
        return False
    except Exception as e:
        print(f"❌ [TEST] Errore generale: {e}")
        print(f"❌ [TEST] Tipo errore: {type(e).__name__}")
        return False

def force_morning_recovery():
    """Forza il recovery del morning report anche se i flag dicono che è già inviato"""
    print("🚀 [FORCE] Avvio recovery forzato morning report")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("serverlite", "555-serverlite.py")
        serverlite = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serverlite)
        
        generate_morning_news = serverlite.generate_morning_news
        set_message_sent_flag = serverlite.set_message_sent_flag
        save_daily_flags = serverlite.save_daily_flags
        GLOBAL_FLAGS = serverlite.GLOBAL_FLAGS
        
        # Forza il flag a False temporaneamente per permettere l'invio
        original_state = GLOBAL_FLAGS.get("morning_news_sent", False)
        GLOBAL_FLAGS["morning_news_sent"] = False
        print(f"🔄 [FORCE] Flag temporaneamente impostato a False (era: {original_state})")
        
        # Esegui morning report
        print("🌅 [FORCE] Esecuzione generate_morning_news()...")
        generate_morning_news()
        
        # Ripristina e salva flag corretti
        set_message_sent_flag("morning_news")
        save_daily_flags()
        
        print("✅ [FORCE] Recovery forzato completato!")
        return True
        
    except Exception as e:
        print(f"❌ [FORCE] Errore recovery forzato: {e}")
        return False

if __name__ == "__main__":
    # Test normale
    success = test_morning_recovery()
    
    # Se fallisce o se vuoi forzare comunque
    if not success:
        print("\n" + "="*60)
        print("🚨 [FALLBACK] Tentativo recovery forzato...")
        force_morning_recovery()