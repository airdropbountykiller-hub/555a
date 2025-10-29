#!/usr/bin/env python3
"""
Narrative Continuity System - 555 Lite
Sistema per mantenere coerenza narrativa tra tutti i messaggi della giornata

Flow temporale:
18:00 (giorno precedente) → 08:00 Rassegna → 09:00 Morning → 13:00 Lunch → 18:00 Daily Summary
"""

import json
import os
import datetime
import pytz
from typing import Dict, List, Any, Optional

# File per persistenza dati
CONTINUITY_FILE = os.path.join('salvataggi', 'narrative_continuity.json')

class NarrativeContinuity:
    def __init__(self):
        self.data = self.load_continuity_data()
    
    def load_continuity_data(self) -> Dict[str, Any]:
        """Carica dati di continuità dal file JSON"""
        try:
            if os.path.exists(CONTINUITY_FILE):
                with open(CONTINUITY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Verifica se i dati sono di oggi
                    today = datetime.datetime.now(pytz.timezone('Europe/Rome')).strftime('%Y-%m-%d')
                    if data.get('date') == today:
                        return data
            
            # Crea nuova struttura per oggi
            return self._create_new_day_structure()
        except Exception as e:
            print(f"⚠️ [CONTINUITY] Error loading data: {e}")
            return self._create_new_day_structure()
    
    def _create_new_day_structure(self) -> Dict[str, Any]:
        """Crea nuova struttura dati per la giornata"""
        italy_tz = pytz.timezone('Europe/Rome')
        now = datetime.datetime.now(italy_tz)
        
        return {
            'date': now.strftime('%Y-%m-%d'),
            'day_name': now.strftime('%A'),
            'yesterday_summary': self._get_yesterday_summary(),
            'predictions': {
                'morning_predictions': [],
                'lunch_updates': [],
                'verified_at_lunch': [],
                'verified_at_summary': []
            },
            'session_data': {
                'morning_regime': None,
                'morning_sentiment': None,
                'morning_key_focus': [],
                'lunch_sentiment_shift': None,
                'lunch_regime_confirmation': None,
                'daily_performance': {}
            },
            'narrative_threads': {
                'main_story': None,
                'sector_focus': None,
                'risk_theme': None,
                'crypto_narrative': None
            },
            'cross_references': {
                'rassegna_to_morning': [],
                'morning_to_lunch': [],
                'lunch_to_summary': []
            }
        }
    
    def _get_yesterday_summary(self) -> Dict[str, Any]:
        """Recupera il summary del giorno precedente se disponibile"""
        try:
            yesterday = datetime.datetime.now(pytz.timezone('Europe/Rome')) - datetime.timedelta(days=1)
            yesterday_file = CONTINUITY_FILE.replace('narrative_continuity.json', 
                                                   f'narrative_continuity_{yesterday.strftime("%Y%m%d")}.json')
            
            if os.path.exists(yesterday_file):
                with open(yesterday_file, 'r', encoding='utf-8') as f:
                    yesterday_data = json.load(f)
                    return yesterday_data.get('daily_summary', {})
        except:
            pass
        
        return {
            'final_sentiment': 'NEUTRAL',
            'key_themes': ['Market consolidation', 'Earnings season', 'Fed policy watch'],
            'outlook_delivered': 'Mixed market conditions, tech leadership',
            'unresolved_issues': ['Inflation concerns', 'Geopolitical tensions']
        }
    
    def save_continuity_data(self):
        """Salva dati di continuità su file"""
        try:
            os.makedirs(os.path.dirname(CONTINUITY_FILE), exist_ok=True)
            with open(CONTINUITY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            # Backup giornaliero
            today = self.data.get('date', '').replace('-', '')
            backup_file = CONTINUITY_FILE.replace('narrative_continuity.json', 
                                                f'narrative_continuity_{today}.json')
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ [CONTINUITY] Error saving data: {e}")
    
    # === RASSEGNA STAMPA 08:00 ===
    def get_rassegna_opening_context(self) -> Dict[str, str]:
        """Genera contesto di apertura per rassegna basato sul giorno precedente"""
        yesterday = self.data['yesterday_summary']
        
        return {
            'opening_line': f"📰 Continuando dall'outlook di ieri: {yesterday.get('outlook_delivered', 'Market analysis')}",
            'overnight_check': f"🌏 Overnight Asia: Verifica di {yesterday.get('final_sentiment', 'NEUTRAL')} sentiment",
            'key_themes_followup': f"🔍 Focus continuato: {', '.join(yesterday.get('key_themes', [])[:2])}",
            'unresolved_watch': f"⚠️ Monitor: {', '.join(yesterday.get('unresolved_issues', [])[:2])}"
        }
    
    def set_rassegna_themes(self, main_themes: List[str], sector_focus: str, risk_assessment: str):
        """Imposta i temi principali dalla rassegna"""
        self.data['narrative_threads']['main_story'] = main_themes[0] if main_themes else None
        self.data['narrative_threads']['sector_focus'] = sector_focus
        self.data['narrative_threads']['risk_theme'] = risk_assessment
        self.save_continuity_data()
    
    # === MORNING REPORT 09:00 ===
    def get_morning_rassegna_connection(self) -> Dict[str, str]:
        """Genera collegamenti dalla rassegna al morning report"""
        threads = self.data['narrative_threads']
        
        return {
            'rassegna_followup': f"📰 Dalla rassegna 08:00: {threads.get('main_story', 'Analisi mercati')} - Live update",
            'sector_continuation': f"🎯 Settore focus: {threads.get('sector_focus', 'Multi-sector')} momentum tracking",
            'risk_update': f"🛡️ Risk theme: {threads.get('risk_theme', 'Balanced')} - ML confirmation"
        }
    
    def set_morning_predictions(self, predictions: List[Dict[str, Any]]):
        """Salva previsioni del morning report"""
        self.data['predictions']['morning_predictions'] = predictions
        self.data['cross_references']['rassegna_to_morning'] = [
            f"Sentiment evolution: {self.data['narrative_threads'].get('main_story', 'TBD')}",
            f"Sector focus maintained: {self.data['narrative_threads'].get('sector_focus', 'TBD')}",
            f"Risk assessment updated: {self.data['narrative_threads'].get('risk_theme', 'TBD')}"
        ]
        self.save_continuity_data()
    
    def set_morning_regime_data(self, regime: str, sentiment: str, key_focus: List[str]):
        """Salva dati del regime di mercato dal morning"""
        self.data['session_data']['morning_regime'] = regime
        self.data['session_data']['morning_sentiment'] = sentiment
        self.data['session_data']['morning_key_focus'] = key_focus
        self.save_continuity_data()
    
    # === LUNCH REPORT 13:00 ===
    def get_lunch_morning_connection(self) -> Dict[str, str]:
        """Genera collegamenti dal morning al lunch report"""
        session_data = self.data['session_data']
        predictions = self.data['predictions']['morning_predictions']
        
        connection = {
            'morning_followup': f"🌅 Dal morning 09:00: {session_data.get('morning_regime', 'TBD')} regime - Intraday check",
            'sentiment_tracking': f"📊 Sentiment: {session_data.get('morning_sentiment', 'TBD')} - Evolution analysis",
            'focus_areas_update': f"🎯 Focus areas: {', '.join(session_data.get('morning_key_focus', [])[:2])} - Progress check"
        }
        
        if predictions:
            connection['predictions_check'] = f"🔍 Prediction check: {len(predictions)} morning forecasts under review"
        
        return connection
    
    def verify_morning_predictions(self, verifications: List[Dict[str, Any]]):
        """Verifica le previsioni del morning al lunch"""
        self.data['predictions']['verified_at_lunch'] = verifications
        self.data['cross_references']['morning_to_lunch'] = [
            f"Regime confirmation: {self.data['session_data'].get('morning_regime', 'TBD')}",
            f"Intraday evolution tracked from morning focus",
            f"Predictions: {len(verifications)} verified at midday"
        ]
        self.save_continuity_data()
    
    def set_lunch_sentiment_shift(self, sentiment_shift: str, regime_confirmation: bool):
        """Registra cambiamenti di sentiment al lunch"""
        self.data['session_data']['lunch_sentiment_shift'] = sentiment_shift
        self.data['session_data']['lunch_regime_confirmation'] = regime_confirmation
        self.save_continuity_data()
    
    # === EVENING REPORT 17:00 ===
    def set_evening_data(self, evening_sentiment: str, evening_performance: Dict[str, str], tomorrow_setup: Dict[str, str]):
        """Salva dati dell'evening report per il daily summary"""
        self.data['session_data']['evening_sentiment'] = evening_sentiment
        self.data['session_data']['evening_performance'] = evening_performance
        self.data['session_data']['tomorrow_setup'] = tomorrow_setup
        
        # Aggiorna cross references per daily summary
        self.data['cross_references']['evening_to_summary'] = [
            f"Evening sentiment: {evening_sentiment}",
            f"Wall Street close: {evening_performance.get('wall_street_close', 'TBD')}",
            f"Tomorrow outlook: {tomorrow_setup.get('asia_handoff', 'TBD')}"
        ]
        self.save_continuity_data()
    
    def get_evening_lunch_connection(self) -> Dict[str, str]:
        """Genera collegamenti dal lunch all'evening report per il messaggio 2"""
        session_data = self.data['session_data']
        verified_predictions = self.data['predictions'].get('verified_predictions', [])
        
        connection = {
            'lunch_followup': f"🍽️ Dal lunch 13:00: {session_data.get('lunch_sentiment_shift', 'Unknown')} sentiment shift tracked",
            'regime_status': f"🎯 Regime check: {'CONFIRMED' if session_data.get('lunch_regime_confirmation') else 'EVOLVING'}",
            'predictions_summary': f"✅ Morning forecasts: {len([v for v in verified_predictions if v.get('status') == 'CORRECT'])}/{len(verified_predictions)} verified"
        }
        
        # Accuracy calculation
        if verified_predictions:
            accuracy = len([v for v in verified_predictions if v.get('status') == 'CORRECT']) / len(verified_predictions) * 100
            session_data['lunch_predictions_accuracy'] = accuracy
            connection['accuracy_note'] = f"📊 Prediction accuracy: {accuracy:.0f}%"
        
        return connection
    
    # === DAILY SUMMARY 18:00 ===
    def get_summary_evening_connection(self) -> Dict[str, str]:
        """Genera collegamenti dall'evening al daily summary"""
        session_data = self.data['session_data']
        evening_perf = session_data.get('evening_performance', {})
        
        return {
            'evening_followup': f"🌆 Dall'evening 17:00: {session_data.get('evening_sentiment', 'Neutral')} close sentiment",
            'wall_street_summary': f"🏦 Wall Street: {evening_perf.get('wall_street_close', 'Performance data loading')}",
            'tomorrow_preparation': f"🔎 Tomorrow setup: {session_data.get('tomorrow_setup', {}).get('asia_handoff', 'Analysis in progress')}"
        }
    
    def get_summary_lunch_connection(self) -> Dict[str, str]:
        """Genera collegamenti dal lunch al daily summary (mantenuto per compatibilità)"""
        session_data = self.data['session_data']
        
        return {
            'lunch_followup': f"🍽️ Dal lunch 13:00: {session_data.get('lunch_sentiment_shift', 'Stable')} sentiment evolution",
            'regime_final_check': f"📊 Regime finale: {session_data.get('morning_regime', 'TBD')} {'confirmed' if session_data.get('lunch_regime_confirmation') else 'evolved'}",
            'session_continuity': f"🔄 Sessione: Morning → Lunch → Evening narrative complete"
        }
    
    def create_daily_summary_data(self, final_sentiment: str, key_achievements: List[str], 
                                tomorrow_outlook: str, unresolved_issues: List[str]) -> Dict[str, Any]:
        """Crea dati riassuntivi per il daily summary"""
        summary_data = {
            'final_sentiment': final_sentiment,
            'key_themes': key_achievements,
            'outlook_delivered': tomorrow_outlook,
            'unresolved_issues': unresolved_issues,
            'session_performance': {
                'morning_regime': self.data['session_data'].get('morning_regime'),
                'lunch_shift': self.data['session_data'].get('lunch_sentiment_shift'),
                'narrative_consistency': self._calculate_consistency_score()
            }
        }
        
        # Salva per domani
        self.data['daily_summary'] = summary_data
        self.data['cross_references']['lunch_to_summary'] = [
            f"Complete session tracked: {self.data['session_data'].get('morning_regime', 'TBD')} to {final_sentiment}",
            f"Narrative threads resolved: {len(key_achievements)} key achievements",
            f"Tomorrow setup: {tomorrow_outlook[:50]}..."
        ]
        self.save_continuity_data()
        return summary_data
    
    def _calculate_consistency_score(self) -> float:
        """Calcola score di coerenza narrativa della giornata"""
        score = 0.0
        
        # Verifica continuità temi
        if self.data['narrative_threads']['main_story']:
            score += 0.3
        
        # Verifica prediction tracking
        predictions = len(self.data['predictions']['morning_predictions'])
        verified = len(self.data['predictions']['verified_at_lunch'])
        if predictions > 0:
            score += 0.3 * (verified / predictions)
        
        # Verifica regime consistency
        if self.data['session_data']['morning_regime'] and self.data['session_data']['lunch_regime_confirmation']:
            score += 0.4
        
        return min(score, 1.0)
    
    # === UTILITY FUNCTIONS ===
    def get_current_narrative_state(self) -> Dict[str, Any]:
        """Ritorna lo stato attuale della narrativa"""
        return {
            'date': self.data['date'],
            'main_story': self.data['narrative_threads']['main_story'],
            'sector_focus': self.data['narrative_threads']['sector_focus'],
            'current_regime': self.data['session_data']['morning_regime'],
            'predictions_count': len(self.data['predictions']['morning_predictions']),
            'consistency_score': self._calculate_consistency_score()
        }
    
    def generate_cross_reference_summary(self, message_type: str) -> List[str]:
        """Genera riassunto riferimenti incrociati per un tipo di messaggio"""
        if message_type == 'morning':
            return self.data['cross_references']['rassegna_to_morning']
        elif message_type == 'lunch':
            return self.data['cross_references']['morning_to_lunch']
        elif message_type == 'summary':
            return self.data['cross_references']['lunch_to_summary']
        return []


# Singleton instance
narrative_continuity = NarrativeContinuity()

def get_narrative_continuity():
    """Ritorna l'istanza singleton del sistema di continuità"""
    return narrative_continuity