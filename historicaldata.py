from coinbase.rest import RESTClient
from coinbase.rest import market_data
from datetime import datetime, timedelta
import requests
import logging
import time
import os
import json
import hashlib
from typing import List, Tuple, Dict, Optional

CHUNK_SIZE_CANDLES = {
    "ONE_MINUTE": 5,  # 300 minutes (5 hours)
    "FIVE_MINUTE": 25,  # 120 candles (10 hours)
    "TEN_MINUTE": 50,  # 60 candles (10 hours)
    "FIFTEEN_MINUTE": 75,  # 40 candles (10 hours)
    "THIRTY_MINUTE": 150,  # 20 candles (10 hours)
    "ONE_HOUR": 300,  # 300 candles (300 hours)
    "SIX_HOUR": 1800,  # 48 candles (12 days)
    "ONE_DAY": 7200,  # 24 candles (12 days)
}

CACHE_DIR = "candle_data"
CACHE_TTL = 3600  # Cache time-to-live in seconds (1 hour)

class HistoricalData:
    
    def __init__(self, client: RESTClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self._init_cache()

    def _init_cache(self):
        """Initialize the cache directory if it doesn't exist."""
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            self.logger.info(f"Created cache directory at {CACHE_DIR}")

    def _get_cache_key(self, product_id: str, start: int, end: int, granularity: str) -> str:
        """Generate a unique cache key for the request."""
        # Round timestamps to the nearest hour to improve cache hits, except for the last hour
        start_hour = start - (start % 3600)
        # Don't round the end time if it's within the last hour
        current_time = int(time.time())
        if current_time - end < 3600:
            end_hour = end  # Use exact timestamp for recent data
        else:
            end_hour = end - (end % 3600)
        key_parts = f"{product_id}_{start_hour}_{end_hour}_{granularity}"
        cache_key = hashlib.md5(key_parts.encode()).hexdigest()
        self.logger.debug(f"Generated cache key {cache_key} for {product_id} from {datetime.fromtimestamp(start)} to {datetime.fromtimestamp(end)}")
        return cache_key

    def _get_cached_data(self, cache_key: str, end_timestamp: int) -> Optional[List[dict]]:
        """Retrieve cached data if it exists and is not expired."""
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            self.logger.debug(f"Cache miss: {cache_key}")
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is expired
            if time.time() - cached_data['timestamp'] > CACHE_TTL:
                self.logger.debug(f"Cache expired: {cache_key}")
                os.remove(cache_file)
                return None

            # If this request includes current time period, don't use cache
            current_time = int(time.time())
            if current_time - end_timestamp < 3600:  # Within the last hour
                self.logger.debug(f"Skipping cache for recent data: {cache_key}")
                return None

            self.logger.debug(f"Cache hit: {cache_key}")
            return cached_data['candles']
        except Exception as e:
            self.logger.error(f"Error reading cache file: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            return None

    def _convert_candle_to_dict(self, candle) -> dict:
        """Convert a Candle object to a dictionary."""
        candle_dict = {
            'start': candle.start,
            'time': candle.start,  # Add time field for compatibility
            'low': candle.low,
            'high': candle.high,
            'open': candle.open,
            'close': candle.close,
            'volume': candle.volume
        }
        return candle_dict

    def _cache_data(self, cache_key: str, candles: List[dict]):
        """Cache the candle data to a file."""
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
        try:
            # Convert Candle objects to dictionaries if needed
            serializable_candles = [
                self._convert_candle_to_dict(candle) if not isinstance(candle, dict) else candle
                for candle in candles
            ]
            cache_data = {
                'timestamp': time.time(),
                'candles': serializable_candles
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            self.logger.error(f"Error writing to cache file: {e}")

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime, granularity: str = "ONE_HOUR") -> List[dict]:
        all_candles = []
        current_start = start_date
        chunk_size_hours = CHUNK_SIZE_CANDLES.get(granularity, 300)  # Default to 300 hours for ONE_HOUR
        chunk_size = timedelta(hours=chunk_size_hours)

        # Track unique candles to avoid duplicates
        seen_candles = set()
        
        # Track data sources for summary
        cached_ranges = []
        fetched_ranges = []

        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            start = int(current_start.timestamp())
            end = int(current_end.timestamp())

            # Format timestamps for logging
            start_str = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')
            end_str = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
            time_range = (start_str, end_str)

            # Try to get data from cache first
            cache_key = self._get_cache_key(product_id, start, end, granularity)
            cached_candles = self._get_cached_data(cache_key, end)

            if cached_candles is not None:
                # Add only unseen candles
                new_count = 0
                for candle in cached_candles:
                    candle_key = (candle['start'], candle['close'])
                    if candle_key not in seen_candles:
                        all_candles.append(candle)
                        seen_candles.add(candle_key)
                        new_count += 1
                cached_ranges.append((time_range, new_count))
                self.logger.debug(f"Retrieved {new_count} unique candles from cache for {product_id}")
            else:
                try:
                    candles = market_data.get_candles(
                        self.client,
                        product_id=product_id,
                        start=start,
                        end=end,
                        granularity=granularity
                    )
                    # Convert candles to dictionaries before caching
                    candle_dicts = [self._convert_candle_to_dict(candle) for candle in candles['candles']]
                    
                    # Add only unseen candles
                    new_candles = []
                    for candle in candle_dicts:
                        candle_key = (candle['start'], candle['close'])
                        if candle_key not in seen_candles:
                            new_candles.append(candle)
                            seen_candles.add(candle_key)
                    
                    # Only cache if not in the current time period
                    current_time = int(time.time())
                    if current_time - end >= 3600:  # Not within the last hour
                        self._cache_data(cache_key, candle_dicts)
                    
                    all_candles.extend(new_candles)
                    fetched_ranges.append((time_range, len(new_candles)))
                    self.logger.debug(f"Fetched {len(new_candles)} new candles from API for {product_id}")
                    time.sleep(0.5)  # Add a small delay to avoid rate limiting
                except requests.exceptions.HTTPError as e:
                    self.logger.error(f"Error fetching candle data: {e}", exc_info=True)

            current_start = current_end

        # Sort the candles by their start time to ensure they are in chronological order
        all_candles.sort(key=lambda x: x['start'])
        
        # Log summary of data sources with grouping
        def log_grouped_ranges(ranges, source_type):
            if not ranges:
                return
            
            self.logger.info(f"Data {source_type}:")
            current_group = []
            current_count = ranges[0][1]
            
            for (start_time, end_time), count in ranges:
                if count != current_count and current_group:
                    start_range = current_group[0][0]
                    end_range = current_group[-1][1]
                    self.logger.info(f"  - {start_range} to {end_range}: {current_count} candles per chunk ({len(current_group)} chunks)")
                    current_group = []
                current_group.append((start_time, end_time))
                current_count = count
            
            if current_group:
                start_range = current_group[0][0]
                end_range = current_group[-1][1]
                self.logger.info(f"  - {start_range} to {end_range}: {current_count} candles per chunk ({len(current_group)} chunks)")

        log_grouped_ranges(cached_ranges, "retrieved from cache")
        log_grouped_ranges(fetched_ranges, "fetched from API")
        
        total_cached = sum(count for _, count in cached_ranges)
        total_fetched = sum(count for _, count in fetched_ranges)
        self.logger.info(f"Summary for {product_id} ({granularity}):")
        self.logger.info(f"  - Total cached candles: {total_cached}")
        self.logger.info(f"  - Total fetched candles: {total_fetched}")
        self.logger.info(f"  - Total unique candles: {len(all_candles)}")
        
        return all_candles

    def clear_cache(self):
        """Clear all cached candle data."""
        try:
            for cache_file in os.listdir(CACHE_DIR):
                if cache_file.endswith('.json'):
                    os.remove(os.path.join(CACHE_DIR, cache_file))
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    # Additional methods related to historical data can be added here