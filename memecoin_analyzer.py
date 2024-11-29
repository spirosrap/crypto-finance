import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

class MemecoinAnalyzer:
    def __init__(self):
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.twitter_trending = "https://api.twitter.com/2/trending"  # You'll need Twitter API credentials
        self.known_memecoins = {
            "dogecoin": "DOGE",
            "shiba-inu": "SHIB",
            "pepe": "PEPE",
            "floki": "FLOKI",            
            "bonk": "BONK"
        }
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MemecoinAnalyzer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

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

    def find_opportunities(self) -> List[Dict]:
        """Find potential memecoin opportunities"""
        opportunities = []
        
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
                        float(social_metrics['social_score'])
                    )
                }
                
                opportunities.append(opportunity)
                
            except Exception as e:
                self.logger.error(f"Error processing {coin_id}: {str(e)}")
                continue

        # Sort by opportunity score
        return sorted(opportunities, key=lambda x: x['opportunity_score'], reverse=True)

    def _calculate_risk_level(self, price_change: float, volume: float, social_score: float) -> str:
        """Calculate risk level based on various metrics"""
        try:
            risk_score = abs(float(price_change)) * 0.4 + (float(volume) / 1e6) * 0.3 + float(social_score) * 0.3
            
            if risk_score > 80:
                return "VERY HIGH"
            elif risk_score > 60:
                return "HIGH"
            elif risk_score > 40:
                return "MEDIUM"
            elif risk_score > 20:
                return "LOW"
            return "VERY LOW"
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {str(e)}")
            return "UNKNOWN"

    def _calculate_opportunity_score(self, price_change: float, volume: float, social_score: float) -> float:
        """Calculate overall opportunity score"""
        try:
            volume_score = min(float(volume) / 1e6, 100)  # Cap at 100
            price_score = min(max(float(price_change), -100), 100)  # Cap between -100 and 100
            
            # Weighted average of different factors
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
            print(f"Opportunity Score: {opp['opportunity_score']}")
            print("-"*30)

def main():
    analyzer = MemecoinAnalyzer()
    analyzer.monitor_opportunities()

if __name__ == "__main__":
    main() 