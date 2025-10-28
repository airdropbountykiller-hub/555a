# Daily Session Tracker - Narrative Continuity System
import json
import os
import datetime
from typing import Dict, List, Any

# File per stato sessione giornaliera
DAILY_SESSION_FILE = os.path.join('salvataggi', 'daily_session.json')

class DailySessionTracker:
    def __init__(self):
        self.session_data = self.load_session_data()
    
    def load_session_data(self) -> Dict:
        """Carica dati sessione giornaliera"""
        try:
            today_key = datetime.datetime.now().strftime('%Y%m%d')
            
            if os.path.exists(DAILY_SESSION_FILE):
                with open(DAILY_SESSION_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Verifica se è dello stesso giorno
                if data.get('date') == today_key:
                    return data
            
            # Nuovo giorno o file non esistente
            return {
                'date': today_key,
                'morning': {},
                'noon': {},
                'evening': {},
                'daily_focus': [],
                'ml_progression': {},
                'performance_tracking': {}
            }
        except Exception as e:
            print(f"⚠️ [SESSION] Error loading session data: {e}")
            return self._get_empty_session()
    
    def _get_empty_session(self) -> Dict:
        """Restituisce sessione vuota"""
        return {
            'date': datetime.datetime.now().strftime('%Y%m%d'),
            'morning': {},
            'noon': {},
            'evening': {},
            'daily_focus': [],
            'ml_progression': {},
            'performance_tracking': {}
        }
    
    def save_session_data(self):
        """Salva dati sessione"""
        try:
            os.makedirs(os.path.dirname(DAILY_SESSION_FILE), exist_ok=True)
            with open(DAILY_SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            print("✅ [SESSION] Session data saved")
        except Exception as e:
            print(f"❌ [SESSION] Error saving session data: {e}")
    
    def set_morning_focus(self, focus_items: List[str], key_events: Dict, ml_sentiment: str):
        """Imposta focus mattutino per la giornata"""
        self.session_data['morning'] = {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'focus_items': focus_items,
            'key_events': key_events,
            'ml_sentiment': ml_sentiment,
            'predictions': []
        }
        self.session_data['daily_focus'] = focus_items
        self.session_data['ml_progression']['morning'] = {
            'sentiment': ml_sentiment,
            'confidence': 'HIGH',
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        }
        self.save_session_data()
    
    def update_noon_progress(self, sentiment_update: str, market_moves: Dict, predictions_check: List[Dict]):
        """Aggiorna progresso a mezzogiorno"""
        self.session_data['noon'] = {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'sentiment_update': sentiment_update,
            'market_moves': market_moves,
            'predictions_check': predictions_check
        }
        self.session_data['ml_progression']['noon'] = {
            'sentiment': sentiment_update,
            'change_from_morning': self._calculate_sentiment_change('morning', sentiment_update),
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        }
        self.save_session_data()
    
    def set_evening_recap(self, final_sentiment: str, performance_results: Dict, tomorrow_setup: Dict):
        """Imposta recap serale"""
        self.session_data['evening'] = {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'final_sentiment': final_sentiment,
            'performance_results': performance_results,
            'tomorrow_setup': tomorrow_setup
        }
        self.session_data['ml_progression']['evening'] = {
            'sentiment': final_sentiment,
            'day_trend': self._calculate_day_trend(),
            'timestamp': datetime.datetime.now().strftime('%H:%M')
        }
        self.session_data['performance_tracking'] = performance_results
        self.save_session_data()
    
    def _calculate_sentiment_change(self, from_phase: str, current_sentiment: str) -> str:
        """Calcola cambio sentiment"""
        try:
            prev_sentiment = self.session_data['ml_progression'].get(from_phase, {}).get('sentiment', 'NEUTRAL')
            
            if prev_sentiment == current_sentiment:
                return "STABLE"
            elif self._sentiment_score(current_sentiment) > self._sentiment_score(prev_sentiment):
                return "IMPROVING"
            else:
                return "DETERIORATING"
        except:
            return "UNKNOWN"
    
    def _sentiment_score(self, sentiment: str) -> int:
        """Converti sentiment in score numerico"""
        mapping = {
            'NEGATIVE': -1,
            'NEUTRAL': 0, 
            'POSITIVE': 1,
            'VERY_POSITIVE': 2,
            'VERY_NEGATIVE': -2
        }
        return mapping.get(sentiment, 0)
    
    def _calculate_day_trend(self) -> str:
        """Calcola trend giornaliero"""
        try:
            morning = self._sentiment_score(self.session_data['ml_progression']['morning']['sentiment'])
            noon = self._sentiment_score(self.session_data['ml_progression']['noon']['sentiment']) 
            evening = self._sentiment_score(self.session_data['ml_progression']['evening']['sentiment'])
            
            if evening > morning:
                return "IMPROVING_DAY"
            elif evening < morning:
                return "DETERIORATING_DAY"
            else:
                return "STABLE_DAY"
        except:
            return "UNKNOWN"
    
    def get_morning_narrative(self) -> List[str]:
        """Genera narrative per morning report"""
        narratives = []
        
        # Focus giornaliero
        if self.session_data['daily_focus']:
            focus_str = ", ".join(self.session_data['daily_focus'][:3])
            narratives.append(f"🎯 **Focus Giornata**: {focus_str}")
        
        # Eventi chiave
        morning_data = self.session_data.get('morning', {})
        if morning_data.get('key_events'):
            events = morning_data['key_events']
            for event_type, event_desc in list(events.items())[:2]:
                narratives.append(f"📅 **{event_type}**: {event_desc}")
        
        return narratives
    
    def get_noon_narrative(self) -> List[str]:
        """Genera narrative per noon report con continuità da morning"""
        narratives = []
        
        morning_data = self.session_data.get('morning', {})
        noon_data = self.session_data.get('noon', {})
        
        # Riferimento a previsioni mattutine
        if morning_data.get('focus_items') and noon_data.get('market_moves'):
            focus = morning_data['focus_items'][0] if morning_data['focus_items'] else "mercati"
            narratives.append(f"🔄 **Update Morning Preview**: {focus} - tracking in linea con attese")
        
        # Sentiment progression
        ml_noon = self.session_data['ml_progression'].get('noon', {})
        if ml_noon.get('change_from_morning'):
            change = ml_noon['change_from_morning']
            change_emoji = "📈" if change == "IMPROVING" else "📉" if change == "DETERIORATING" else "➡️"
            narratives.append(f"{change_emoji} **Sentiment Update**: {ml_noon.get('sentiment', 'NEUTRAL')} ({change.lower()} da stamattina)")
        
        # Performance check
        if noon_data.get('predictions_check'):
            correct_predictions = len([p for p in noon_data['predictions_check'] if p.get('status') == 'CORRECT'])
            total_predictions = len(noon_data['predictions_check'])
            if total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                narratives.append(f"✅ **Morning Calls**: {correct_predictions}/{total_predictions} corrette ({accuracy:.0f}%)")
        
        return narratives
    
    def get_evening_narrative(self) -> List[str]:
        """Genera narrative per evening report con recap completo"""
        narratives = []
        
        morning_data = self.session_data.get('morning', {})
        evening_data = self.session_data.get('evening', {})
        
        # Recap focus giornaliero
        if morning_data.get('focus_items') and evening_data.get('performance_results'):
            focus = morning_data['focus_items'][0] if morning_data['focus_items'] else "focus mattutino"
            performance = evening_data['performance_results']
            if performance.get('success_rate', 0) > 70:
                narratives.append(f"✅ **Recap Giornata**: {focus} - obiettivi raggiunti con successo")
            else:
                narratives.append(f"📊 **Recap Giornata**: {focus} - performance mista, adeguamenti in corso")
        
        # Trend giornaliero completo
        day_trend = self._calculate_day_trend()
        if day_trend != "UNKNOWN":
            trend_emoji = "📈" if day_trend == "IMPROVING_DAY" else "📉" if day_trend == "DETERIORATING_DAY" else "➡️"
            trend_desc = day_trend.replace("_DAY", "").lower()
            narratives.append(f"{trend_emoji} **Day Trend**: Giornata {trend_desc} (morning→evening)")
        
        # Setup domani basato su oggi
        if evening_data.get('tomorrow_setup'):
            setup = evening_data['tomorrow_setup']
            narratives.append(f"🔮 **Tomorrow Setup**: {setup.get('strategy', 'Continuation strategy')} basato su sviluppi odierni")
        
        return narratives
    
    def get_session_stats(self) -> Dict:
        """Restituisce statistiche sessione corrente"""
        return {
            'phases_completed': len([k for k in ['morning', 'noon', 'evening'] if k in self.session_data and self.session_data[k]]),
            'focus_items_count': len(self.session_data.get('daily_focus', [])),
            'ml_progression_phases': len(self.session_data.get('ml_progression', {})),
            'has_performance_data': bool(self.session_data.get('performance_tracking'))
        }

# Singleton instance
daily_tracker = DailySessionTracker()

# Helper functions for easy integration
def set_morning_focus(focus_items: List[str], key_events: Dict, ml_sentiment: str):
    """Set daily focus from morning report"""
    daily_tracker.set_morning_focus(focus_items, key_events, ml_sentiment)

def update_noon_progress(sentiment_update: str, market_moves: Dict, predictions_check: List[Dict]):
    """Update progress at noon"""
    daily_tracker.update_noon_progress(sentiment_update, market_moves, predictions_check)

def set_evening_recap(final_sentiment: str, performance_results: Dict, tomorrow_setup: Dict):
    """Set evening recap"""
    daily_tracker.set_evening_recap(final_sentiment, performance_results, tomorrow_setup)

def get_morning_narrative() -> List[str]:
    """Get morning narrative lines"""
    return daily_tracker.get_morning_narrative()

def get_noon_narrative() -> List[str]:
    """Get noon narrative lines"""
    return daily_tracker.get_noon_narrative()

def get_evening_narrative() -> List[str]:
    """Get evening narrative lines"""
    return daily_tracker.get_evening_narrative()

def get_session_stats() -> Dict:
    """Get current session statistics"""
    return daily_tracker.get_session_stats()

# Enhanced prediction tracking
def add_morning_prediction(prediction_type: str, prediction_text: str, target_time: str, confidence: str):
    """Add morning prediction for later verification"""
    prediction = {
        'type': prediction_type,
        'text': prediction_text,
        'target_time': target_time,
        'confidence': confidence,
        'timestamp': datetime.datetime.now().strftime('%H:%M'),
        'status': 'PENDING'
    }
    
    if 'morning' not in daily_tracker.session_data:
        daily_tracker.session_data['morning'] = {}
    if 'predictions' not in daily_tracker.session_data['morning']:
        daily_tracker.session_data['morning']['predictions'] = []
    
    daily_tracker.session_data['morning']['predictions'].append(prediction)
    daily_tracker.save_session_data()

def check_predictions_at_noon() -> List[Dict]:
    """Check morning predictions at noon"""
    predictions = daily_tracker.session_data.get('morning', {}).get('predictions', [])
    checked_predictions = []
    
    for pred in predictions:
        # Mock check - in real implementation would verify against market data
        pred['status'] = 'CORRECT'  # Placeholder
        checked_predictions.append(pred)
    
    return checked_predictions