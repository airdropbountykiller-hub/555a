# Momentum indicators e segnali avanzati per analisi ML
import datetime
from typing import Dict, List, Tuple

def calculate_news_momentum(analyzed_news: List[Dict]) -> Dict:
    """
    Calcola momentum delle notizie nel tempo per rilevare accelerazioni di sentiment
    """
    try:
        # Raggruppa news per ora
        hourly_sentiment = {}
        hourly_impact = {}
        
        for news in analyzed_news:
            # Estrai ora dalla notizia (assumendo formato time in titolo/link)
            try:
                now = datetime.datetime.now()
                hour = now.hour  # Default ora corrente
                
                # Inizializza bucket orario
                if hour not in hourly_sentiment:
                    hourly_sentiment[hour] = []
                    hourly_impact[hour] = []
                
                # Converti sentiment in score numerico
                sentiment_score = 1 if news['sentiment'] == 'POSITIVE' else -1 if news['sentiment'] == 'NEGATIVE' else 0
                impact_score = 3 if news['impact'] == 'HIGH' else 2 if news['impact'] == 'MEDIUM' else 1
                
                hourly_sentiment[hour].append(sentiment_score)
                hourly_impact[hour].append(impact_score)
                
            except Exception:
                continue
        
        # Calcola momentum (derivata del sentiment)
        momentum_data = {}
        hours = sorted(hourly_sentiment.keys())
        
        for i, hour in enumerate(hours):
            avg_sentiment = sum(hourly_sentiment[hour]) / len(hourly_sentiment[hour])
            avg_impact = sum(hourly_impact[hour]) / len(hourly_impact[hour])
            
            if i > 0:
                prev_sentiment = sum(hourly_sentiment[hours[i-1]]) / len(hourly_sentiment[hours[i-1]])
                momentum = avg_sentiment - prev_sentiment
            else:
                momentum = 0
            
            momentum_data[hour] = {
                'sentiment': avg_sentiment,
                'impact': avg_impact,
                'momentum': momentum,
                'news_count': len(hourly_sentiment[hour])
            }
        
        # Calcola overall momentum trend
        recent_momentum = []
        for hour in sorted(hours)[-3:]:  # Ultime 3 ore
            recent_momentum.append(momentum_data[hour]['momentum'])
        
        avg_momentum = sum(recent_momentum) / len(recent_momentum) if recent_momentum else 0
        
        # Determina direzione momentum
        if avg_momentum > 0.3:
            momentum_direction = "ACCELERATING POSITIVE"
            momentum_emoji = "📈🚀"
        elif avg_momentum < -0.3:
            momentum_direction = "ACCELERATING NEGATIVE"  
            momentum_emoji = "📉⚡"
        elif avg_momentum > 0.1:
            momentum_direction = "MILD POSITIVE"
            momentum_emoji = "📈"
        elif avg_momentum < -0.1:
            momentum_direction = "MILD NEGATIVE"
            momentum_emoji = "📉"
        else:
            momentum_direction = "SIDEWAYS"
            momentum_emoji = "➡️"
        
        return {
            'momentum_direction': momentum_direction,
            'momentum_emoji': momentum_emoji,
            'momentum_score': avg_momentum,
            'hourly_data': momentum_data
        }
        
    except Exception as e:
        print(f"⚠️ [MOMENTUM] Error: {e}")
        return {
            'momentum_direction': "UNKNOWN",
            'momentum_emoji': "❓",
            'momentum_score': 0,
            'hourly_data': {}
        }

def detect_news_catalysts(analyzed_news: List[Dict], categoria_weights: Dict) -> Dict:
    """
    Rileva potenziali catalysts per movimenti di mercato
    """
    try:
        catalysts = []
        
        # Keywords per catalysts di alto impatto
        catalyst_keywords = {
            'fed_meeting': ['Powell', 'Fed', 'FOMC', 'rates', 'interest', 'monetary policy'],
            'earnings_surprise': ['earnings', 'beat', 'miss', 'guidance', 'revenue'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'bid', 'deal'],
            'geopolitical_shock': ['war', 'conflict', 'sanctions', 'trade war', 'nuclear'],
            'regulatory_change': ['regulation', 'SEC', 'antitrust', 'compliance', 'ban'],
            'tech_breakthrough': ['AI', 'breakthrough', 'innovation', 'patent', 'quantum'],
            'crypto_institutional': ['ETF', 'institutional', 'adoption', 'MicroStrategy', 'Tesla']
        }
        
        for news in analyzed_news:
            title_lower = news['title'].lower()
            
            for catalyst_type, keywords in catalyst_keywords.items():
                if any(keyword.lower() in title_lower for keyword in keywords):
                    # Calcola catalyst strength
                    cat_weight = categoria_weights.get(news['categoria'], 1.0)
                    impact_multiplier = 3 if news['impact'] == 'HIGH' else 2 if news['impact'] == 'MEDIUM' else 1
                    
                    catalyst_strength = cat_weight * impact_multiplier
                    
                    catalysts.append({
                        'type': catalyst_type.replace('_', ' ').title(),
                        'title': news['title'][:60] + "..." if len(news['title']) > 60 else news['title'],
                        'strength': catalyst_strength,
                        'categoria': news['categoria'],
                        'sentiment': news['sentiment']
                    })
        
        # Ordina per strength e prendi top 3
        top_catalysts = sorted(catalysts, key=lambda x: x['strength'], reverse=True)[:3]
        
        return {
            'has_major_catalyst': len([c for c in catalysts if c['strength'] > 4]) > 0,
            'top_catalysts': top_catalysts,
            'total_catalysts': len(catalysts)
        }
        
    except Exception as e:
        print(f"⚠️ [CATALYSTS] Error: {e}")
        return {
            'has_major_catalyst': False,
            'top_catalysts': [],
            'total_catalysts': 0
        }

def generate_trading_signals(market_regime: Dict, momentum: Dict, catalysts: Dict) -> List[str]:
    """
    Genera segnali di trading basati su regime + momentum + catalysts
    """
    try:
        signals = []
        
        # Combina informazioni per segnali
        regime_name = market_regime['name']
        momentum_direction = momentum['momentum_direction']
        has_catalyst = catalysts['has_major_catalyst']
        
        # Segnali regime-based
        if regime_name == 'BULL MARKET':
            if momentum_direction == 'ACCELERATING POSITIVE':
                signals.append("🚀 **STRONG BUY SIGNAL**: Bull regime + accelerating momentum")
            elif has_catalyst and any(c['sentiment'] == 'POSITIVE' for c in catalysts['top_catalysts']):
                signals.append("📈 **BUY SIGNAL**: Bull + positive catalyst alignment")
            else:
                signals.append("✅ **HOLD/BUY**: Bull regime supports longs")
        
        elif regime_name == 'BEAR MARKET':
            if momentum_direction == 'ACCELERATING NEGATIVE':
                signals.append("🩸 **STRONG SELL SIGNAL**: Bear regime + accelerating decline")
            elif has_catalyst and any(c['sentiment'] == 'NEGATIVE' for c in catalysts['top_catalysts']):
                signals.append("📉 **SELL/SHORT SIGNAL**: Bear + negative catalyst")
            else:
                signals.append("⚠️ **DEFENSIVE**: Bear regime - reduce risk")
        
        elif regime_name == 'HIGH VOLATILITY':
            if has_catalyst:
                signals.append("⚡ **VOLATILITY PLAY**: Major catalyst in volatile regime")
            if momentum_direction == 'SIDEWAYS':
                signals.append("🔄 **RANGE TRADING**: Mean reversion opportunities")
            else:
                signals.append("🛡️ **HEDGE PORTFOLIO**: High vol environment")
        
        # Momentum-specific signals
        if momentum_direction == 'ACCELERATING POSITIVE' and regime_name != 'BEAR MARKET':
            signals.append("🚀 **MOMENTUM BUY**: News sentiment accelerating up")
        elif momentum_direction == 'ACCELERATING NEGATIVE' and regime_name != 'BULL MARKET':
            signals.append("⚡ **MOMENTUM SELL**: News sentiment deteriorating fast")
        
        # Catalyst-specific signals
        if has_catalyst:
            for catalyst in catalysts['top_catalysts'][:2]:  # Top 2
                if catalyst['strength'] > 5:
                    direction = "📈 LONG" if catalyst['sentiment'] == 'POSITIVE' else "📉 SHORT" if catalyst['sentiment'] == 'NEGATIVE' else "📊 WATCH"
                    signals.append(f"🎯 **{catalyst['type'].upper()}**: {direction} {catalyst['categoria']}")
        
        # Failsafe - sempre almeno un segnale
        if not signals:
            signals.append(f"📊 **STANDARD ALLOCATION**: {regime_name} regime - normal risk management")
        
        return signals[:4]  # Max 4 segnali per non sovraccaricare
        
    except Exception as e:
        print(f"⚠️ [TRADING-SIGNALS] Error: {e}")
        return ["📊 **STANDARD**: Monitor market conditions"]

def calculate_risk_metrics(analyzed_news: List[Dict], market_regime: Dict) -> Dict:
    """
    Calcola metriche di rischio basate su news e regime
    """
    try:
        # Conta news per tipo di rischio
        geopolitical_risk = len([n for n in analyzed_news if n['categoria'] == 'Geopolitica' and n['impact'] != 'LOW'])
        financial_stress = len([n for n in analyzed_news if n['categoria'] == 'Finanza' and n['sentiment'] == 'NEGATIVE'])
        regulatory_risk = len([n for n in analyzed_news if 'regulation' in n['title'].lower() or 'ban' in n['title'].lower()])
        
        # Calcola VIX proxy basato su notizie
        total_high_impact = len([n for n in analyzed_news if n['impact'] == 'HIGH'])
        total_negative = len([n for n in analyzed_news if n['sentiment'] == 'NEGATIVE'])
        
        volatility_proxy = (total_high_impact * 2 + total_negative) / max(1, len(analyzed_news))
        
        # Risk score complessivo
        base_risk = market_regime.get('position_sizing', 1.0)  # Già incorpora regime risk
        news_risk_multiplier = 1 + (volatility_proxy * 0.3)
        
        total_risk_score = (2 - base_risk) * news_risk_multiplier  # Inverte position sizing
        
        # Risk level
        if total_risk_score > 1.5:
            risk_level = "HIGH"
            risk_emoji = "🚨"
        elif total_risk_score > 1.0:
            risk_level = "MEDIUM" 
            risk_emoji = "⚠️"
        else:
            risk_level = "LOW"
            risk_emoji = "✅"
        
        return {
            'risk_level': risk_level,
            'risk_emoji': risk_emoji,
            'risk_score': total_risk_score,
            'geopolitical_events': geopolitical_risk,
            'financial_stress_events': financial_stress,
            'regulatory_events': regulatory_risk,
            'volatility_proxy': volatility_proxy
        }
        
    except Exception as e:
        print(f"⚠️ [RISK-METRICS] Error: {e}")
        return {
            'risk_level': "UNKNOWN",
            'risk_emoji': "❓",
            'risk_score': 1.0,
            'geopolitical_events': 0,
            'financial_stress_events': 0,
            'regulatory_events': 0,
            'volatility_proxy': 0.5
        }