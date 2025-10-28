#!/usr/bin/env python3
"""
INVIO RAPIDO - Rassegna Stampa + Morning Report
Per l'app deployata su Render
"""
import requests
import time

def invio_rapido():
    # URL dell'app su Render
    BASE_URL = "https://five55a.onrender.com"
    
    print("🚀 INVIO RAPIDO - Rassegna Stampa + Morning Report")
    print("=" * 50)
    
    # Test se l'app è attiva
    try:
        print("🔍 Test connessione app...")
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ App attiva e funzionante!")
        else:
            print(f"⚠️ App risponde con status {response.status_code}")
    except Exception as e:
        print(f"❌ App non raggiungibile: {e}")
        return False
    
    # Reset flag prima dell'invio
    try:
        print("\n🔄 Reset flag...")
        requests.get(f"{BASE_URL}/api/reset-flags", timeout=10)
        print("✅ Flag resettati")
    except Exception as e:
        print(f"⚠️ Reset flag fallito: {e}")
    
    time.sleep(1)
    
    # 1. RASSEGNA STAMPA
    print("\n📰 INVIO RASSEGNA STAMPA...")
    try:
        response = requests.get(f"{BASE_URL}/api/force-rassegna", timeout=120)
        result = response.json()
        if result.get("status") == "success":
            print("✅ RASSEGNA STAMPA INVIATA!")
        else:
            print(f"❌ Errore rassegna: {result.get('message', 'Sconosciuto')}")
    except Exception as e:
        print(f"❌ Errore connessione rassegna: {e}")
    
    time.sleep(2)
    
    # 2. MORNING REPORT
    print("\n🌅 INVIO MORNING REPORT...")
    try:
        response = requests.get(f"{BASE_URL}/api/force-morning", timeout=120)
        result = response.json()
        if result.get("status") == "success":
            print("✅ MORNING REPORT INVIATO!")
        else:
            print(f"❌ Errore morning: {result.get('message', 'Sconosciuto')}")
    except Exception as e:
        print(f"❌ Errore connessione morning: {e}")
    
    print("\n🎉 INVIO COMPLETATO!")
    print("📱 Controlla i messaggi Telegram!")
    return True

if __name__ == "__main__":
    invio_rapido()