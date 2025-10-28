# ML Session Continuity System
# Sistema per mantenere coerenza ML tra Morning → Noon → Evening reports
import datetime
import json
from typing import Dict, Any, Optional
import os

class MLSessionContinuity:
    """
    Gestisce la continuità delle analisi ML durante la giornata di trading
    per assicurare coerenza tra Morning, Noon e Evening reports
    """
    
    def __init__(self, data_file="ml_session_data.json"):
        self.data_file = data_file
        self.session_data = self._load_session_data()
    
    def _load_session_data(self) -> Dict:
        """Carica i dati di sessione ML dal file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def _save_session_data(self):
        """Salva i dati di sessione ML su file"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error saving data: {e}")
    
    def store_morning_analysis(self, news_analysis: Dict[str, Any]):
        """Memorizza l'analisi ML del morning report per riutilizzo"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Store key ML data from morning analysis
            morning_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'sentiment': news_analysis.get('sentiment', 'NEUTRAL'),
                'market_impact': news_analysis.get('market_impact', 'MEDIUM'),
                'market_regime': news_analysis.get('market_regime', {}),
                'momentum': news_analysis.get('momentum', {}),
                'catalysts': news_analysis.get('catalysts', {}),
                'trading_signals': news_analysis.get('trading_signals', []),
                'risk_metrics': news_analysis.get('risk_metrics', {}),
                'category_weights': news_analysis.get('category_weights', {}),
                'analyzed_news_count': len(news_analysis.get('analyzed_news', []))
            }
            
            if today not in self.session_data:
                self.session_data[today] = {}
            
            self.session_data[today]['morning'] = morning_data
            self._save_session_data()
            
            print(f"✅ [ML-SESSION] Morning analysis stored: {morning_data['sentiment']} sentiment, {morning_data['market_regime'].get('name', 'UNKNOWN')} regime")
            
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error storing morning data: {e}")
    
    def get_morning_analysis(self) -> Optional[Dict]:
        """Recupera l'analisi ML del morning per riutilizzo in noon/evening"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            return self.session_data.get(today, {}).get('morning')
        except Exception:
            return None
    
    def store_noon_update(self, intraday_changes: Dict[str, Any]):
        """Memorizza gli aggiornamenti intraday per l'evening report"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            noon_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'market_moves': intraday_changes.get('market_moves', {}),
                'sentiment_shift': intraday_changes.get('sentiment_shift', False),
                'new_catalysts': intraday_changes.get('new_catalysts', []),
                'risk_update': intraday_changes.get('risk_update', {}),
                'regime_confirmation': intraday_changes.get('regime_confirmation', True)
            }
            
            if today not in self.session_data:
                self.session_data[today] = {}
            
            self.session_data[today]['noon'] = noon_data
            self._save_session_data()
            
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error storing noon data: {e}")
    
    def get_session_evolution(self) -> Dict[str, Any]:
        """Ritorna l'evoluzione della sessione ML per evening report"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            daily_data = self.session_data.get(today, {})
            
            morning = daily_data.get('morning', {})
            noon = daily_data.get('noon', {})
            
            if not morning:
                return {'status': 'no_morning_data'}
            
            # Calculate evolution metrics
            evolution = {
                'morning_regime': morning.get('market_regime', {}).get('name', 'UNKNOWN'),
                'morning_sentiment': morning.get('sentiment', 'NEUTRAL'),
                'morning_risk': morning.get('risk_metrics', {}).get('risk_level', 'MEDIUM'),
                'intraday_changes': noon.get('sentiment_shift', False),
                'regime_stable': noon.get('regime_confirmation', True),
                'session_consistency': self._calculate_consistency_score(morning, noon),
                'key_signals_today': morning.get('trading_signals', [])[:3],
                'dominant_category': self._get_dominant_category(morning.get('category_weights', {}))
            }
            
            return evolution
            
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error getting evolution: {e}")
            return {'status': 'error'}
    
    def _calculate_consistency_score(self, morning: Dict, noon: Dict) -> float:
        """Calcola score di consistenza ML tra morning e noon"""
        try:
            if not noon:
                return 0.8  # Default se noon non disponibile
            
            # Factors per consistency scoring
            regime_stable = 0.4 if noon.get('regime_confirmation', True) else 0.0
            sentiment_stable = 0.3 if not noon.get('sentiment_shift', False) else 0.1
            risk_stable = 0.3  # Default stable se non disponibile
            
            return min(1.0, regime_stable + sentiment_stable + risk_stable)
            
        except Exception:
            return 0.7  # Default moderate consistency
    
    def _get_dominant_category(self, category_weights: Dict) -> str:
        """Identifica la categoria dominante della giornata"""
        try:
            if not category_weights:
                return "Mixed"
            
            dominant = max(category_weights.items(), key=lambda x: x[1])
            return dominant[0] if dominant[1] > 1.2 else "Balanced"
            
        except Exception:
            return "Unknown"
    
    def generate_continuity_narrative(self) -> list:
        """Genera narrative di continuità per evening report"""
        try:
            evolution = self.get_session_evolution()
            
            if evolution.get('status') == 'no_morning_data':
                return ["• Session tracking: Morning baseline not available"]
            
            narratives = []
            
            # Regime continuity
            regime = evolution.get('morning_regime', 'UNKNOWN')
            if regime != 'UNKNOWN':
                status = "maintained" if evolution.get('regime_stable', True) else "shifted"
                narratives.append(f"• 📊 **Regime Continuity**: {regime} {status} throughout session")
            
            # Sentiment evolution
            sentiment = evolution.get('morning_sentiment', 'NEUTRAL')
            if evolution.get('intraday_changes', False):
                narratives.append(f"• 💭 **Sentiment Evolution**: Started {sentiment}, intraday shifts detected")
            else:
                narratives.append(f"• 💭 **Sentiment Stability**: {sentiment} maintained consistently")
            
            # Consistency score
            consistency = evolution.get('session_consistency', 0.7)
            consistency_emoji = "🟢" if consistency > 0.8 else "🟡" if consistency > 0.6 else "🔴"
            narratives.append(f"• {consistency_emoji} **Session Consistency**: {consistency*100:.0f}% - ML predictions alignment")
            
            # Dominant theme
            dominant = evolution.get('dominant_category', 'Mixed')
            if dominant not in ['Mixed', 'Balanced', 'Unknown']:
                narratives.append(f"• 🔥 **Daily Theme**: {dominant} sector dominated news flow")
            
            return narratives[:4]  # Max 4 narratives
            
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error generating narrative: {e}")
            return ["• Session continuity: Analysis in progress"]
    
    def cleanup_old_sessions(self, days_to_keep: int = 7):
        """Pulisce i dati di sessioni vecchie"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            keys_to_remove = [k for k in self.session_data.keys() if k < cutoff_str]
            
            for key in keys_to_remove:
                del self.session_data[key]
            
            if keys_to_remove:
                self._save_session_data()
                print(f"✅ [ML-SESSION] Cleaned {len(keys_to_remove)} old sessions")
                
        except Exception as e:
            print(f"⚠️ [ML-SESSION] Error cleaning old sessions: {e}")

# Global instance for session continuity
ml_session = MLSessionContinuity()

# Helper functions for integration with reports
def store_morning_ml_analysis(news_analysis):
    """Store morning analysis for session continuity"""
    ml_session.store_morning_analysis(news_analysis)

def get_stored_morning_analysis():
    """Get stored morning analysis for noon/evening reuse"""
    return ml_session.get_morning_analysis()

def update_noon_session(intraday_changes):
    """Update session with noon developments"""
    ml_session.store_noon_update(intraday_changes)

def get_evening_continuity_narrative():
    """Get continuity narrative for evening report"""
    return ml_session.generate_continuity_narrative()

def get_session_evolution_summary():
    """Get complete session evolution for evening"""
    return ml_session.get_session_evolution()