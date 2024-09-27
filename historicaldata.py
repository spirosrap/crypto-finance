from coinbase.rest import RESTClient
from coinbase.rest import market_data
from datetime import datetime, timedelta
import requests
import logging
import time
from typing import List, Tuple
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

class HistoricalData:
    
    def __init__(self, client: RESTClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime, granularity: str = "ONE_HOUR") -> List[dict]:
        all_candles = []
        current_start = start_date
        chunk_size_hours = CHUNK_SIZE_CANDLES.get(granularity, 300)  # Default to 300 hours for ONE_HOUR
        chunk_size = timedelta(hours=chunk_size_hours)

        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            start = int(current_start.timestamp())
            end = int(current_end.timestamp())

            try:
                candles = market_data.get_candles(
                    self.client,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity=granularity
                )
                all_candles.extend(candles['candles'])
                current_start = current_end

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"Error fetching candle data: {e}", exc_info=True)

        # Sort the candles by their start time to ensure they are in chronological order
        all_candles.sort(key=lambda x: x['start'])
        self.logger.info(f"Fetched {len(all_candles)} candles for {product_id} with granularity {granularity}.")
        return all_candles

    # Additional methods related to historical data can be added here