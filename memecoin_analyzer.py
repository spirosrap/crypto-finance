import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
import tweepy
from config import BEARER_TOKEN, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

class MemecoinAnalyzer:
    def __init__(self, use_twitter: bool = False):
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.twitter_trending = "https://api.twitter.com/2/trending"
        self.known_memecoins = {
            "dogecoin": "DOGE",
            "shiba-inu": "SHIB",
            "pepe": "PEPE",
            "floki": "FLOKI",            
            "bonk": "BONK"
        }
        self.logger = self._setup_logger()
        self.use_twitter = use_twitter
        self.twitter_api = self._setup_twitter_api() if use_twitter else None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MemecoinAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_twitter_api(self) -> tweepy.Client:
        """Setup Twitter API v2 authentication"""
        try:
            client = tweepy.Client(
                bearer_token=BEARER_TOKEN,
                consumer_key=CONSUMER_KEY,
                consumer_secret=CONSUMER_SECRET,
                access_token=ACCESS_TOKEN,
                access_token_secret=ACCESS_TOKEN_SECRET
            )
            return client
        except Exception as e:
            self.logger.error(f"Error setting up Twitter API: {str(e)}")
            return None

    def get_memecoin_data(self, coin_id: str) -> Dict:
        """Fetch detailed data for a specific memecoin"""
        try:
            url = f"{self.coingecko_api}/coins/{coin_id}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            self.logger.error(f"Error fetching data for {coin_id}: {str(e)}")
            return {}

    def analyze_social_metrics(self, coin_data: Dict) -> Dict:
        """Analyze social media metrics for a coin"""
        try:
            community_data = coin_data.get('community_data', {})
            if not community_data:
                return {
                    'twitter_followers': 0,
                    'reddit_subscribers': 0,
                    'telegram_members': 0,
                    'social_score': 0
                }

            metrics = {
                'twitter_followers': community_data.get('twitter_followers', 0) or 0,
                'reddit_subscribers': community_data.get('reddit_subscribers', 0) or 0,
                'telegram_members': community_data.get('telegram_channel_user_count', 0) or 0,
                'social_score': 0
            }
            
            # Calculate social score based on various metrics
            metrics['social_score'] = (
                float(metrics['twitter_followers']) * 0.4 +
                float(metrics['reddit_subscribers']) * 0.3 +
                float(metrics['telegram_members']) * 0.3
            ) / 1000  # Normalize score
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error in analyze_social_metrics: {str(e)}")
            return {
                'twitter_followers': 0,
                'reddit_subscribers': 0,
                'telegram_members': 0,
                'social_score': 0
            }

    def calculate_momentum_score(self, price_data: List[float]) -> float:
        """Calculate momentum score based on recent price action"""
        if not price_data or len(price_data) < 2:
            return 0
        
        returns = np.diff(price_data) / price_data[:-1]
        weights = np.linspace(0.5, 1, len(returns))  # More recent price changes have higher weights
        momentum_score = np.sum(returns * weights) * 100
        return round(momentum_score, 2)

    def detect_pump_patterns(self, volume_data: List[float], price_data: List[float]) -> bool:
        """Detect potential pump and dump patterns"""
        if len(volume_data) < 24 or len(price_data) < 24:
            return False

        # Calculate volume and price changes
        vol_change = (volume_data[-1] / np.mean(volume_data[-24:-1])) - 1
        price_change = (price_data[-1] / np.mean(price_data[-24:-1])) - 1

        # Suspicious patterns
        return vol_change > 3 and price_change > 0.3  # 300% volume increase and 30% price increase

    def get_twitter_trending(self) -> Dict[str, int]:
        """Analyze Twitter mentions for memecoins using search"""
        try:
            if not self.twitter_api:
                return {}

            coin_mentions = {symbol.lower(): 0 for symbol in self.known_memecoins.values()}
            
            # Search for each memecoin in the last 24 hours
            for coin_id, symbol in self.known_memecoins.items():
                try:
                    # Search using both coin ID and symbol
                    search_terms = [
                        f"#{symbol.lower()}", 
                        f"#{coin_id.replace('-', '')}", 
                        symbol.lower(),
                        coin_id
                    ]
                    
                    total_mentions = 0
                    for term in search_terms:
                        # Use Twitter API v2 search endpoint
                        response = self.twitter_api.search_recent_tweets(
                            query=term,
                            max_results=5,
                            tweet_fields=['public_metrics']
                        )
                        
                        if response and response.data:
                            total_mentions += len(response.data)
                    
                    coin_mentions[symbol.lower()] = total_mentions
                    time.sleep(2)  # Rate limiting precaution
                    
                except Exception as e:
                    self.logger.warning(f"Error searching for {symbol}: {str(e)}")
                    continue
            
            return coin_mentions
            
        except Exception as e:
            self.logger.error(f"Error fetching Twitter data: {str(e)}")
            return {}

    def find_opportunities(self) -> List[Dict]:
        """Find potential memecoin opportunities"""
        opportunities = []
        twitter_mentions = self.get_twitter_trending() if self.use_twitter else {}
        
        for coin_id, symbol in self.known_memecoins.items():
            try:
                coin_data = self.get_memecoin_data(coin_id)
                if not coin_data:
                    self.logger.warning(f"No data found for {coin_id}")
                    continue

                # Get price data with safety checks
                price_data = coin_data.get('market_data', {})
                if not price_data:
                    self.logger.warning(f"No market data found for {coin_id}")
                    continue

                current_price = price_data.get('current_price', {}).get('usd', 0) or 0
                price_change_24h = price_data.get('price_change_percentage_24h', 0) or 0
                volume_24h = price_data.get('total_volume', {}).get('usd', 0) or 0

                # Analyze metrics
                social_metrics = self.analyze_social_metrics(coin_data)
                
                # Add Twitter trending data to opportunity calculation only if enabled
                mentions = twitter_mentions.get(symbol.lower(), 0) if self.use_twitter else 0
                
                # Calculate scores with safety checks
                opportunity = {
                    'symbol': symbol,
                    'name': coin_data.get('name', symbol),
                    'current_price': float(current_price),
                    'price_change_24h': float(price_change_24h),
                    'volume_24h': float(volume_24h),
                    'social_score': float(social_metrics['social_score']),
                    'risk_level': self._calculate_risk_level(
                        float(price_change_24h),
                        float(volume_24h),
                        float(social_metrics['social_score'])
                    ),
                    'opportunity_score': self._calculate_opportunity_score(
                        float(price_change_24h),
                        float(volume_24h),
                        float(social_metrics['social_score']),
                        mentions if self.use_twitter else 0
                    )
                }
                
                # Only add Twitter mentions if Twitter is enabled
                if self.use_twitter:
                    opportunity['twitter_mentions'] = mentions

                opportunities.append(opportunity)
                
            except Exception as e:
                self.logger.error(f"Error processing {coin_id}: {str(e)}")
                continue

        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)

    def _calculate_risk_level(self, price_change: float, volume: float, social_score: float) -> str:
        """Calculate risk level based on various metrics"""
        try:
            # Base risk - all memecoins start with inherent risk
            base_risk = 0.4  # 40% base risk for being a memecoin
            
            # Price volatility risk (higher volatility = higher risk)
            price_risk = min(abs(float(price_change)) / 10, 1)  # 10% change = 1.0
            
            # Volume risk - higher volume doesn't necessarily mean lower risk for memecoins
            # Instead, look for abnormal volume patterns
            avg_memecoin_volume = 100e6  # $100M as baseline
            volume_ratio = float(volume) / avg_memecoin_volume
            volume_risk = 0.5
            if volume_ratio > 3:  # Unusually high volume
                volume_risk = 0.7
            elif volume_ratio < 0.3:  # Very low volume
                volume_risk = 0.8
            
            # Social engagement risk
            social_risk = max(1 - (float(social_score) / 1000), 0.3)  # Minimum 0.3 risk
            
            # Calculate weighted risk score with base risk included
            risk_score = (
                base_risk * 0.3 +          # Base memecoin risk
                price_risk * 0.3 +         # Price volatility impact
                volume_risk * 0.2 +        # Volume pattern risk
                social_risk * 0.2          # Social engagement risk
            ) * 100
            
            # Adjusted thresholds - higher baseline due to memecoin nature
            if risk_score > 75:
                return "VERY HIGH"
            elif risk_score > 60:
                return "HIGH"
            elif risk_score > 45:
                return "MEDIUM-HIGH"
            elif risk_score > 35:
                return "MEDIUM"
            return "MEDIUM-LOW"  # No memecoin should be considered "very low" risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return "UNKNOWN"

    def _calculate_opportunity_score(self, price_change: float, volume: float, 
                                  social_score: float, twitter_mentions: int = 0) -> float:
        """Calculate overall opportunity score including Twitter trends if enabled"""
        try:
            volume_score = min(float(volume) / 1e6, 100)  # Cap at 100
            price_score = min(max(float(price_change), -100), 100)  # Cap between -100 and 100
            
            if self.use_twitter:
                twitter_score = min(twitter_mentions / 1000, 100)  # Normalize Twitter mentions
                # Weighted average with Twitter
                score = (
                    price_score * 0.35 +
                    volume_score * 0.25 +
                    float(social_score) * 0.25 +
                    twitter_score * 0.15
                )
            else:
                # Weighted average without Twitter
                score = (
                    price_score * 0.4 +
                    volume_score * 0.3 +
                    float(social_score) * 0.3
                )
            
            return round(max(0, score), 2)  # Ensure non-negative score
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return 0.0

    def monitor_opportunities(self, interval: int = 300):
        """Continuously monitor for memecoin opportunities"""
        self.logger.info("Starting memecoin opportunity monitoring...")
        
        while True:
            try:
                opportunities = self.find_opportunities()
                if opportunities:
                    self._print_opportunities(opportunities)
                else:
                    self.logger.warning("No opportunities found in this iteration")
                time.sleep(interval)  # Wait for specified interval
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error during monitoring: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

    def _print_opportunities(self, opportunities: List[Dict]):
        """Print formatted opportunities"""
        print("\n" + "="*50)
        print(f"Memecoin Opportunities Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        for opp in opportunities:
            print(f"\nCoin: {opp['symbol']} ({opp['name']})")
            print(f"Price: ${opp['current_price']:.8f}")
            print(f"24h Change: {opp['price_change_24h']:.2f}%")
            print(f"24h Volume: ${opp['volume_24h']:,.2f}")
            print(f"Social Score: {opp['social_score']:.2f}")
            print(f"Risk Level: {opp['risk_level']}")
            if self.use_twitter and 'twitter_mentions' in opp:
                print(f"Twitter Mentions: {opp['twitter_mentions']:,}")
            print(f"Opportunity Score: {opp['opportunity_score']}")
            print("-"*30)

def main():
    # Create analyzer with Twitter disabled by default
    analyzer = MemecoinAnalyzer(use_twitter=False)
    analyzer.monitor_opportunities()

if __name__ == "__main__":
    main() 