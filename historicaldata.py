from coinbase.rest import RESTClient
from coinbase.rest import market_data
from datetime import datetime, timedelta
import requests
import logging
import time
from typing import List, Tuple
CHUNK_SIZE_HOURS = 300

class HistoricalData:
    def __init__(self, client: RESTClient):
        self.client = client
        self.logger = logging.getLogger(__name__)

    def get_hourly_data(self, product_id, days=60):
        end = int(datetime.utcnow().timestamp())
        all_candles = []
        
        for i in range(0, days, 14):
            start = end - 86400 * min(14, days - i)
            try:
                candles = market_data.get_candles(
                    self.client,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity="ONE_HOUR"
                )
                all_candles = candles['candles'] + all_candles
                end = start - 1  # Set end to 1 second before start for next iteration
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"Error fetching hourly candle data: {e}")
                break
        self.logger.info(f"Fetched {len(all_candles)} hourly candles for {product_id}.")
        return all_candles

    def get_6h_data(self, product_id):
        end = int(datetime.utcnow().timestamp())
        start = end - 86400 * 30  # 30 days in seconds
        try:
            candles = market_data.get_candles(
                self.client,
                product_id=product_id,
                start=start,
                end=end,
                granularity="SIX_HOUR"
            )
            self.logger.info(f"Fetched {len(candles['candles'])} 6-hour candles for {product_id}.")
            return candles['candles']
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Error fetching 6-hour candle data: {e}")
            return []

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime) -> List[dict]:
        all_candles = []
        current_start = start_date
        chunk_size = timedelta(hours=CHUNK_SIZE_HOURS)

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
                    granularity="ONE_HOUR"
                )
                all_candles.extend(candles['candles'])
                current_start = current_end

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"Error fetching candle data: {e}", exc_info=True)

        # Sort the candles by their start time to ensure they are in chronological order
        all_candles.sort(key=lambda x: x['start'])
        self.logger.info(f"Fetched {len(all_candles)} candles for {product_id}.")
        return all_candles

    # Additional methods related to historical data can be added here