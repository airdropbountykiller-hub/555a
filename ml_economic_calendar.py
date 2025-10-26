# Enhanced Economic Calendar with ML Integration
import datetime
from typing import Dict, List, Tuple

def generate_dynamic_calendar_events(days_ahead: int = 7) -> List[Dict]:
    """
    Genera calendario eventi economici dinamico con ML context
    """
    try:
        events = []
        base_date = datetime.datetime.now()
        
        # Template eventi ricorrenti con ML implications
        event_templates = {
            'monday': [
                {'time': '15:30', 'event': 'US Factory Orders', 'importance': 'MEDIUM', 'ml_implication': 'Manufacturing sentiment gauge'},
                {'time': '09:00', 'event': 'EU Construction PMI', 'importance': 'LOW', 'ml_implication': 'Real estate cycle indicator'}
            ],
            'tuesday': [
                {'time': '14:30', 'event': 'US Trade Balance', 'importance': 'MEDIUM', 'ml_implication': 'Dollar strength proxy'},
                {'time': '16:00', 'event': 'US Consumer Credit', 'importance': 'LOW', 'ml_implication': 'Consumer leverage trends'}
            ],
            'wednesday': [
                {'time': '20:00', 'event': 'Fed FOMC Minutes', 'importance': 'HIGH', 'ml_implication': 'Policy pivot probability'},
                {'time': '18:00', 'event': 'EIA Crude Oil Inventories', 'importance': 'MEDIUM', 'ml_implication': 'Energy sector momentum'}
            ],
            'thursday': [
                {'time': '14:30', 'event': 'US Initial Jobless Claims', 'importance': 'HIGH', 'ml_implication': 'Labor market resilience'},
                {'time': '10:00', 'event': 'ECB Economic Bulletin', 'importance': 'MEDIUM', 'ml_implication': 'Euro monetary policy signal'}
            ],
            'friday': [
                {'time': '14:30', 'event': 'US Non-Farm Payrolls', 'importance': 'HIGH', 'ml_implication': 'Employment cycle position'},
                {'time': '16:00', 'event': 'US Consumer Sentiment', 'importance': 'MEDIUM', 'ml_implication': 'Spending outlook proxy'}
            ],
            'saturday': [
                {'time': '00:00', 'event': 'Weekend - Markets Closed', 'importance': 'INFO', 'ml_implication': 'Crypto-only trading active'}
            ],
            'sunday': [
                {'time': '00:00', 'event': 'Asia Pre-Open', 'importance': 'LOW', 'ml_implication': 'Gap risk for Monday open'}
            ]
        }
        
        # Genera eventi per i prossimi giorni
        for i in range(days_ahead):
            event_date = base_date + datetime.timedelta(days=i)
            weekday_name = event_date.strftime('%A').lower()
            
            if weekday_name in event_templates:
                for template in event_templates[weekday_name]:
                    event = {
                        'date': event_date.strftime('%Y-%m-%d'),
                        'day_name': event_date.strftime('%A'),
                        'time': template['time'],
                        'event': template['event'],
                        'importance': template['importance'],
                        'ml_implication': template['ml_implication'],
                        'impact_emoji': get_impact_emoji(template['importance']),
                        'day_emoji': get_day_emoji(weekday_name)
                    }
                    events.append(event)
        
        return events
        
    except Exception as e:
        print(f"⚠️ [CALENDAR] Error generating events: {e}")
        return []

def get_impact_emoji(importance: str) -> str:
    """Emoji per livello importanza"""
    emoji_map = {
        'HIGH': '🔥',
        'MEDIUM': '⚡', 
        'LOW': '📊',
        'INFO': 'ℹ️'
    }
    return emoji_map.get(importance, '📊')

def get_day_emoji(weekday: str) -> str:
    """Emoji per giorno settimana"""
    day_emojis = {
        'monday': '🌅',
        'tuesday': '📈', 
        'wednesday': '🎯',
        'thursday': '⚡',
        'friday': '🎭',
        'saturday': '🛌',
        'sunday': '🌏'
    }
    return day_emojis.get(weekday, '📅')

def analyze_calendar_risk_events(events: List[Dict]) -> Dict:
    """
    Analizza eventi calendario per identificare window di alto rischio
    """
    try:
        high_risk_events = [e for e in events if e['importance'] == 'HIGH']
        medium_risk_events = [e for e in events if e['importance'] == 'MEDIUM']
        
        # Calcola risk density per giorno
        daily_risk = {}
        for event in events:
            date = event['date']
            if date not in daily_risk:
                daily_risk[date] = {'high': 0, 'medium': 0, 'low': 0}
            
            daily_risk[date][event['importance'].lower()] += 1
        
        # Identifica giorni high-risk (2+ eventi HIGH o 3+ eventi MEDIUM+HIGH)
        high_risk_days = []
        for date, risk_counts in daily_risk.items():
            risk_score = risk_counts['high'] * 3 + risk_counts['medium'] * 1.5 + risk_counts['low'] * 0.5
            if risk_score >= 4.0:
                day_name = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                high_risk_days.append({
                    'date': date,
                    'day': day_name,
                    'risk_score': risk_score,
                    'events': risk_counts
                })
        
        return {
            'total_high_events': len(high_risk_events),
            'total_medium_events': len(medium_risk_events), 
            'high_risk_days': high_risk_days,
            'peak_risk_day': max(high_risk_days, key=lambda x: x['risk_score']) if high_risk_days else None
        }
        
    except Exception as e:
        print(f"⚠️ [CALENDAR-RISK] Error: {e}")
        return {
            'total_high_events': 0,
            'total_medium_events': 0,
            'high_risk_days': [],
            'peak_risk_day': None
        }

def generate_calendar_trading_strategies(events: List[Dict], risk_analysis: Dict) -> List[str]:
    """
    Genera strategie di trading basate su eventi calendario
    """
    try:
        strategies = []
        
        # Strategie basate su eventi HIGH importance
        high_events = [e for e in events if e['importance'] == 'HIGH']
        
        for event in high_events[:3]:  # Top 3 high impact
            day = event['day_name']
            time = event['time']
            
            if 'NFP' in event['event'] or 'Payrolls' in event['event']:
                strategies.append(f"💼 **NFP Strategy**: {day} {time} - Straddle USD pairs 30min before, volatility play")
                
            elif 'Fed' in event['event'] or 'FOMC' in event['event']:
                strategies.append(f"🏦 **Fed Strategy**: {day} {time} - Watch 2Y/10Y yield curve, prepare rate-sensitive trades")
                
            elif 'ECB' in event['event']:
                strategies.append(f"🇪🇺 **ECB Strategy**: {day} {time} - EUR momentum play, monitor peripheral spreads")
                
            elif 'Claims' in event['event']:
                strategies.append(f"📊 **Claims Strategy**: {day} {time} - Labor market health proxy, USD reaction play")
        
        # Strategie per giorni high-risk
        if risk_analysis['peak_risk_day']:
            peak_day = risk_analysis['peak_risk_day']
            strategies.append(f"⚠️ **Peak Risk Day**: {peak_day['day']} - Reduce position sizes by 30%, hedge portfolio")
        
        # Strategie per cluster di eventi
        if risk_analysis['total_high_events'] > 3:
            strategies.append(f"🔥 **High Event Week**: {risk_analysis['total_high_events']} major events - Consider VIX plays, range breakouts")
        
        # Weekend crypto strategy
        weekend_events = [e for e in events if e['day_name'] in ['Saturday', 'Sunday']]
        if weekend_events:
            strategies.append("₿ **Weekend Strategy**: Crypto markets active - Watch for BTC weekend gaps, thin liquidity")
        
        return strategies[:4]  # Max 4 strategie
        
    except Exception as e:
        print(f"⚠️ [CALENDAR-STRATEGY] Error: {e}")
        return ["📊 **Standard**: Monitor calendar events for volatility"]

def format_calendar_for_telegram(events: List[Dict], strategies: List[str]) -> List[str]:
    """
    Formatta calendario per messaggio Telegram ottimizzato
    """
    try:
        lines = []
        
        # Header calendario
        lines.append("📅 **CALENDARIO EVENTI CHIAVE (7 giorni)**")
        lines.append("")
        
        # Raggruppa eventi per giorno
        daily_events = {}
        for event in events:
            date = event['date']
            if date not in daily_events:
                daily_events[date] = []
            daily_events[date].append(event)
        
        # Formatta per ogni giorno
        for date in sorted(daily_events.keys())[:5]:  # Max 5 giorni
            day_events = daily_events[date]
            if not day_events:
                continue
                
            day_name = day_events[0]['day_name']
            day_emoji = day_events[0]['day_emoji']
            
            lines.append(f"{day_emoji} **{day_name} {date[5:]}**")
            
            for event in day_events[:3]:  # Max 3 eventi per giorno
                lines.append(f"  {event['impact_emoji']} {event['time']} - {event['event']}")
                lines.append(f"    💡 *{event['ml_implication']}*")
            
            lines.append("")
        
        # Aggiungi strategie
        if strategies:
            lines.append("🎯 **STRATEGIE CALENDARIO**")
            lines.append("")
            for strategy in strategies:
                lines.append(f"• {strategy}")
            lines.append("")
        
        return lines
        
    except Exception as e:
        print(f"⚠️ [CALENDAR-FORMAT] Error: {e}")
        return ["📅 Calendario eventi non disponibile"]

def get_market_hours_status() -> Dict:
    """
    Determina status mercati per contest calendario eventi
    """
    try:
        now = datetime.datetime.now()
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour
        
        # Weekend
        if weekday >= 5:  # Saturday or Sunday
            return {
                'status': 'WEEKEND',
                'message': 'Mercati tradizionali chiusi - Solo crypto attivo',
                'emoji': '🛌',
                'next_open': 'Lunedì 09:00 CET'
            }
        
        # Weekday - orari Europa
        if 9 <= hour <= 17:
            return {
                'status': 'MARKET_OPEN',
                'message': 'Mercati europei aperti',
                'emoji': '🔔',
                'next_event': 'Chiusura 17:30 CET'
            }
        elif hour < 9:
            return {
                'status': 'PRE_MARKET',
                'message': 'Pre-market - Mercati chiusi',
                'emoji': '🌅',
                'next_event': 'Apertura 09:00 CET'
            }
        else:
            return {
                'status': 'AFTER_MARKET',
                'message': 'After-hours - Mercati europei chiusi',
                'emoji': '🌙',
                'next_event': 'USA ancora aperto fino 22:00 CET'
            }
            
    except Exception as e:
        print(f"⚠️ [MARKET-HOURS] Error: {e}")
        return {
            'status': 'UNKNOWN',
            'message': 'Status mercati non determinabile',
            'emoji': '❓',
            'next_event': 'Controlla orari standard'
        }