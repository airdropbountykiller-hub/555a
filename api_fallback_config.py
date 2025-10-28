# API Fallback Configuration System
# Sistema di failover per garantire sempre dati live

import os
import requests
import time
from typing import Dict, List, Optional, Any

class APIFallbackManager:
    """
    Gestisce multiple API keys e providers per garantire sempre dati disponibili
    """
    
    def __init__(self):
        # === CRYPTO DATA PROVIDERS ===
        self.crypto_providers = {
            'cryptocompare': {
                'url': 'https://min-api.cryptocompare.com/data/pricemultifull',
                'keys': [
                    os.getenv('CRYPTOCOMPARE_API_KEY_1'),
                    os.getenv('CRYPTOCOMPARE_API_KEY_2'),
                    os.getenv('CRYPTOCOMPARE_API_KEY_3'),
                    None  # Free tier fallback
                ],
                'params_template': {'fsyms': '{symbols}', 'tsyms': 'USD'}
            },
            'coinapi': {
                'url': 'https://rest.coinapi.io/v1/exchangerate/{symbol}/USD',
                'keys': [
                    os.getenv('COINAPI_KEY_1'),
                    os.getenv('COINAPI_KEY_2')
                ],
                'headers_template': {'X-CoinAPI-Key': '{key}'}
            },
            'coingecko': {
                'url': 'https://api.coingecko.com/api/v3/simple/price',
                'keys': [
                    os.getenv('COINGECKO_API_KEY_1'),
                    os.getenv('COINGECKO_API_KEY_2'),
                    None  # Free tier
                ],
                'params_template': {'ids': '{ids}', 'vs_currencies': 'usd', 'include_24hr_change': 'true'}
            }
        }
        
        # === FINANCIAL DATA PROVIDERS ===
        self.financial_providers = {
            'alphavantage': {
                'url': 'https://www.alphavantage.co/query',
                'keys': [
                    os.getenv('ALPHAVANTAGE_API_KEY_1'),
                    os.getenv('ALPHAVANTAGE_API_KEY_2'),
                    os.getenv('ALPHAVANTAGE_API_KEY_3')
                ],
                'params_template': {'function': 'GLOBAL_QUOTE', 'symbol': '{symbol}', 'apikey': '{key}'}
            },
            'finnhub': {
                'url': 'https://finnhub.io/api/v1/quote',
                'keys': [
                    os.getenv('FINNHUB_API_KEY_1'),
                    os.getenv('FINNHUB_API_KEY_2')
                ],
                'params_template': {'symbol': '{symbol}', 'token': '{key}'}
            },
            'twelvedata': {
                'url': 'https://api.twelvedata.com/price',
                'keys': [
                    os.getenv('TWELVEDATA_API_KEY_1'),
                    os.getenv('TWELVEDATA_API_KEY_2')
                ],
                'params_template': {'symbol': '{symbol}', 'apikey': '{key}'}
            }
        }
        
        # === NEWS DATA PROVIDERS ===
        self.news_providers = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'keys': [
                    os.getenv('NEWSAPI_KEY_1'),
                    os.getenv('NEWSAPI_KEY_2')
                ],
                'params_template': {'q': 'bitcoin OR crypto OR market', 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 20, 'apiKey': '{key}'}
            },
            'marketaux': {
                'url': 'https://api.marketaux.com/v1/news/all',
                'keys': [
                    os.getenv('MARKETAUX_API_KEY_1'),
                    os.getenv('MARKETAUX_API_KEY_2')
                ],
                'params_template': {'symbols': 'BTC,ETH', 'filter_entities': 'true', 'language': 'en', 'api_token': '{key}'}
            }
        }
        
        # Rate limiting tracking
        self.rate_limits = {}
        self.failed_keys = {}
    
    def get_crypto_data_with_fallback(self, symbols: str = "BTC,ETH,BNB,SOL") -> Optional[Dict]:
        """
        Ottiene dati crypto con sistema di fallback automatico
        """
        print(f"🔄 [FALLBACK] Tentativo recupero dati crypto per: {symbols}")
        
        # Try CryptoCompare first (primary)
        result = self._try_cryptocompare(symbols)
        if result:
            return result
            
        # Try CoinGecko (secondary)
        result = self._try_coingecko(symbols)
        if result:
            return result
            
        # Try CoinAPI (tertiary)
        result = self._try_coinapi(symbols.split(','))
        if result:
            return result
            
        print("❌ [FALLBACK] Tutti i provider crypto falliti")
        return None
    
    def get_financial_data_with_fallback(self, symbol: str) -> Optional[Dict]:
        """
        Ottiene dati finanziari con fallback
        """
        print(f"🔄 [FALLBACK] Tentativo recupero dati finanziari per: {symbol}")
        
        # Try Alpha Vantage first
        result = self._try_alphavantage(symbol)
        if result:
            return result
            
        # Try Finnhub
        result = self._try_finnhub(symbol)  
        if result:
            return result
            
        # Try Twelve Data
        result = self._try_twelvedata(symbol)
        if result:
            return result
            
        print(f"❌ [FALLBACK] Tutti i provider finanziari falliti per {symbol}")
        return None
    
    def get_news_with_fallback(self, query: str = "bitcoin crypto market") -> Optional[List[Dict]]:
        """
        Ottiene notizie con fallback
        """
        print(f"🔄 [FALLBACK] Tentativo recupero notizie per: {query}")
        
        # Try NewsAPI first
        result = self._try_newsapi(query)
        if result:
            return result
            
        # Try MarketAux
        result = self._try_marketaux()
        if result:
            return result
            
        print("❌ [FALLBACK] Tutti i provider news falliti")
        return None
    
    def _try_cryptocompare(self, symbols: str) -> Optional[Dict]:
        """Tenta CryptoCompare con multiple keys"""
        provider = self.crypto_providers['cryptocompare']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('cryptocompare', i):
                continue
                
            try:
                params = provider['params_template'].copy()
                params['fsyms'] = symbols
                
                if api_key:
                    params['api_key'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'RAW' in data:
                        print(f"✅ [FALLBACK] CryptoCompare OK (key {i+1})")
                        return self._format_cryptocompare_data(data)
                elif response.status_code == 429:  # Rate limited
                    self._mark_key_failed('cryptocompare', i, 300)  # 5 min cooldown
                    continue
                elif response.status_code == 401:  # Invalid key
                    self._mark_key_failed('cryptocompare', i, 3600)  # 1 hour cooldown
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] CryptoCompare key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_coingecko(self, symbols: str) -> Optional[Dict]:
        """Tenta CoinGecko con mapping simboli"""
        provider = self.crypto_providers['coingecko']
        
        # Map symbols to CoinGecko IDs
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'ADA': 'cardano',
            'XRP': 'ripple',
            'DOT': 'polkadot',
            'LINK': 'chainlink'
        }
        
        symbol_list = symbols.split(',')
        gecko_ids = ','.join([symbol_map.get(s.strip(), s.lower()) for s in symbol_list])
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('coingecko', i):
                continue
                
            try:
                params = provider['params_template'].copy()
                params['ids'] = gecko_ids
                
                headers = {}
                if api_key:
                    headers['x-cg-pro-api-key'] = api_key
                
                response = requests.get(provider['url'], params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ [FALLBACK] CoinGecko OK (key {i+1 if api_key else 'free'})")
                    return self._format_coingecko_data(data, symbol_map)
                elif response.status_code == 429:
                    self._mark_key_failed('coingecko', i, 60)  # 1 min cooldown
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] CoinGecko key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_coinapi(self, symbols: List[str]) -> Optional[Dict]:
        """Tenta CoinAPI per singoli simboli"""
        provider = self.crypto_providers['coinapi']
        result_data = {}
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('coinapi', i) or not api_key:
                continue
                
            try:
                headers = provider['headers_template'].copy()
                headers['X-CoinAPI-Key'] = api_key
                
                for symbol in symbols[:4]:  # Limit to 4 symbols to avoid rate limits
                    url = provider['url'].format(symbol=symbol)
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        result_data[symbol] = {
                            'price': data.get('rate', 0),
                            'change_pct': 0  # CoinAPI doesn't provide 24h change in this endpoint
                        }
                    elif response.status_code == 429:
                        self._mark_key_failed('coinapi', i, 300)
                        break
                    
                    time.sleep(0.1)  # Avoid rate limits
                
                if result_data:
                    print(f"✅ [FALLBACK] CoinAPI OK (key {i+1})")
                    return result_data
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] CoinAPI key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_alphavantage(self, symbol: str) -> Optional[Dict]:
        """Tenta Alpha Vantage"""
        provider = self.financial_providers['alphavantage']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('alphavantage', i) or not api_key:
                continue
                
            try:
                params = provider['params_template'].copy()
                params['symbol'] = symbol
                params['apikey'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        result = {
                            'price': float(quote.get('05. price', 0)),
                            'change_pct': float(quote.get('10. change percent', '0%').replace('%', ''))
                        }
                        print(f"✅ [FALLBACK] Alpha Vantage OK (key {i+1})")
                        return result
                elif response.status_code == 429:
                    self._mark_key_failed('alphavantage', i, 60)
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] Alpha Vantage key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_finnhub(self, symbol: str) -> Optional[Dict]:
        """Tenta Finnhub"""
        provider = self.financial_providers['finnhub']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('finnhub', i) or not api_key:
                continue
                
            try:
                params = provider['params_template'].copy()
                params['symbol'] = symbol
                params['token'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'c' in data:  # current price
                        result = {
                            'price': float(data['c']),
                            'change_pct': float(data.get('dp', 0))  # percent change
                        }
                        print(f"✅ [FALLBACK] Finnhub OK (key {i+1})")
                        return result
                elif response.status_code == 429:
                    self._mark_key_failed('finnhub', i, 60)
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] Finnhub key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_twelvedata(self, symbol: str) -> Optional[Dict]:
        """Tenta Twelve Data"""
        provider = self.financial_providers['twelvedata']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('twelvedata', i) or not api_key:
                continue
                
            try:
                params = provider['params_template'].copy()
                params['symbol'] = symbol
                params['apikey'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'price' in data:
                        result = {
                            'price': float(data['price']),
                            'change_pct': 0  # Would need separate endpoint for change
                        }
                        print(f"✅ [FALLBACK] Twelve Data OK (key {i+1})")
                        return result
                elif response.status_code == 429:
                    self._mark_key_failed('twelvedata', i, 300)
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] Twelve Data key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_newsapi(self, query: str) -> Optional[List[Dict]]:
        """Tenta NewsAPI"""
        provider = self.news_providers['newsapi']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('newsapi', i) or not api_key:
                continue
                
            try:
                params = provider['params_template'].copy()
                params['q'] = query
                params['apiKey'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok' and 'articles' in data:
                        articles = []
                        for article in data['articles'][:10]:
                            articles.append({
                                'titolo': article.get('title', ''),
                                'fonte': article.get('source', {}).get('name', 'NewsAPI'),
                                'link': article.get('url', ''),
                                'categoria': 'Financial News'
                            })
                        print(f"✅ [FALLBACK] NewsAPI OK (key {i+1}) - {len(articles)} articles")
                        return articles
                elif response.status_code == 429:
                    self._mark_key_failed('newsapi', i, 3600)  # 1 hour cooldown
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] NewsAPI key {i+1} error: {e}")
                continue
        
        return None
    
    def _try_marketaux(self) -> Optional[List[Dict]]:
        """Tenta MarketAux"""
        provider = self.news_providers['marketaux']
        
        for i, api_key in enumerate(provider['keys']):
            if self._is_key_failed('marketaux', i) or not api_key:
                continue
                
            try:
                params = provider['params_template'].copy()
                params['api_token'] = api_key
                
                response = requests.get(provider['url'], params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        articles = []
                        for article in data['data'][:10]:
                            articles.append({
                                'titolo': article.get('title', ''),
                                'fonte': article.get('source', 'MarketAux'),
                                'link': article.get('url', ''),
                                'categoria': 'Market News'
                            })
                        print(f"✅ [FALLBACK] MarketAux OK (key {i+1}) - {len(articles)} articles")
                        return articles
                elif response.status_code == 429:
                    self._mark_key_failed('marketaux', i, 3600)
                    continue
                    
            except Exception as e:
                print(f"⚠️ [FALLBACK] MarketAux key {i+1} error: {e}")
                continue
        
        return None
    
    def _format_cryptocompare_data(self, data: Dict) -> Dict:
        """Formatta dati CryptoCompare nel formato standard"""
        formatted = {}
        
        for symbol, info in data['RAW'].items():
            if 'USD' in info:
                usd_data = info['USD']
                formatted[symbol] = {
                    'price': usd_data.get('PRICE', 0),
                    'change_pct': usd_data.get('CHANGEPCT24HOUR', 0),
                    'high_24h': usd_data.get('HIGH24HOUR', 0),
                    'low_24h': usd_data.get('LOW24HOUR', 0),
                    'volume_24h': usd_data.get('VOLUME24HOUR', 0),
                    'market_cap': usd_data.get('MKTCAP', 0)
                }
        
        # Calculate total market cap
        total_cap = sum(coin.get('market_cap', 0) for coin in formatted.values())
        formatted['TOTAL_MARKET_CAP'] = total_cap
        
        return formatted
    
    def _format_coingecko_data(self, data: Dict, symbol_map: Dict) -> Dict:
        """Formatta dati CoinGecko nel formato standard"""
        formatted = {}
        reverse_map = {v: k for k, v in symbol_map.items()}
        
        for gecko_id, info in data.items():
            symbol = reverse_map.get(gecko_id, gecko_id.upper())
            formatted[symbol] = {
                'price': info.get('usd', 0),
                'change_pct': info.get('usd_24h_change', 0),
                'high_24h': 0,  # Not available in simple price endpoint
                'low_24h': 0,
                'volume_24h': 0,
                'market_cap': 0
            }
        
        return formatted
    
    def _is_key_failed(self, provider: str, key_index: int) -> bool:
        """Controlla se una chiave è in cooldown"""
        key_id = f"{provider}_{key_index}"
        if key_id in self.failed_keys:
            return time.time() < self.failed_keys[key_id]
        return False
    
    def _mark_key_failed(self, provider: str, key_index: int, cooldown_seconds: int):
        """Marca una chiave come fallita con cooldown"""
        key_id = f"{provider}_{key_index}"
        self.failed_keys[key_id] = time.time() + cooldown_seconds
        print(f"⏱️ [FALLBACK] Key {provider} #{key_index+1} in cooldown for {cooldown_seconds}s")
    
    def get_status_report(self) -> Dict:
        """Genera report dello stato delle API"""
        report = {
            'crypto_providers': len(self.crypto_providers),
            'financial_providers': len(self.financial_providers), 
            'news_providers': len(self.news_providers),
            'total_keys': 0,
            'failed_keys': len(self.failed_keys),
            'active_keys': 0
        }
        
        # Count total keys
        for provider_group in [self.crypto_providers, self.financial_providers, self.news_providers]:
            for provider_name, config in provider_group.items():
                report['total_keys'] += len([k for k in config['keys'] if k is not None])
        
        report['active_keys'] = report['total_keys'] - report['failed_keys']
        
        return report

# Global instance
api_fallback = APIFallbackManager()