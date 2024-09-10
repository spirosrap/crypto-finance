import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import yfinance as yf

class ExternalDataFetcher:
    def __init__(self):
        self.blockchain_info_url = "https://api.blockchain.info/charts/{chart}?timespan=all&sampled=false&metadata=false&cors=true&format=json"
        self.coingecko_url = "https://api.coingecko.com/api/v3"

    def get_data(self, start_date, end_date):
        # Ensure start_date and end_date are date objects
        start_date = start_date.date() if isinstance(start_date, datetime) else start_date
        end_date = end_date.date() if isinstance(end_date, datetime) else end_date

        btc_data = self._fetch_btc_network_data(start_date, end_date)
        time.sleep(1)  # Add a 1-second delay between API calls
        market_data = self._fetch_market_data(start_date, end_date)
        time.sleep(1)  # Add a 1-second delay between API calls
        sp500_data = self._fetch_sp500_data(start_date, end_date)

        # Create a date range from start_date to end_date
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        combined_data = pd.DataFrame({'date': date_range})

        # Convert all date columns to datetime
        combined_data['date'] = pd.to_datetime(combined_data['date'])
        btc_data['date'] = pd.to_datetime(btc_data['date'])
        market_data['date'] = pd.to_datetime(market_data['date'])
        sp500_data['date'] = pd.to_datetime(sp500_data['date'])

        # Merge all data
        combined_data = pd.merge(combined_data, btc_data, on='date', how='left')
        combined_data = pd.merge(combined_data, market_data, on='date', how='left')
        combined_data = pd.merge(combined_data, sp500_data, on='date', how='left')

        # Forward fill missing values
        combined_data = combined_data.ffill()

        return combined_data.sort_values('date').reset_index(drop=True)

    def _fetch_btc_network_data(self, start_date, end_date):
        hash_rate_data = self._get_blockchain_info_data("hash-rate", start_date, end_date)
        return pd.DataFrame(hash_rate_data, columns=['date', 'hash_rate'])

    def _fetch_market_data(self, start_date, end_date):
        btc_market_cap = self._get_coingecko_historical_data("bitcoin", start_date, end_date)
        total_market_cap = self._get_coingecko_historical_data("total", start_date, end_date)

        if btc_market_cap.empty or total_market_cap.empty:
            print("Failed to fetch market data from CoinGecko")
            return pd.DataFrame(columns=['date', 'btc_market_cap', 'total_crypto_market_cap'])

        # Create a date range for the total market cap
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        total_market_cap_full = pd.DataFrame({'date': date_range})

        # Ensure 'date' columns are of the same type (datetime64[ns])
        total_market_cap_full['date'] = pd.to_datetime(total_market_cap_full['date'])
        total_market_cap['date'] = pd.to_datetime(total_market_cap['date'])

        total_market_cap_full = pd.merge(total_market_cap_full, total_market_cap, on='date', how='left')
        total_market_cap_full['total_crypto_market_cap'] = total_market_cap_full['total_crypto_market_cap'].ffill().bfill()

        # Ensure 'date' column in btc_market_cap is also datetime64[ns]
        btc_market_cap['date'] = pd.to_datetime(btc_market_cap['date'])

        market_data = pd.merge(btc_market_cap, total_market_cap_full, on='date', how='outer')
        return market_data

    def _fetch_sp500_data(self, start_date, end_date):
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(start=start_date, end=end_date + timedelta(days=1))
        sp500_data = [(date.date(), close) for date, close in zip(data.index, data['Close'])]
        return pd.DataFrame(sp500_data, columns=['date', 'sp500'])

    def _get_blockchain_info_data(self, chart, start_date, end_date):
        url = self.blockchain_info_url.format(chart=chart)
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()['values']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Blockchain.info: {e}")
            return []

        filtered_data = []
        for d in data:
            try:
                # Try parsing as seconds first
                date = datetime.fromtimestamp(d['x']).date()
            except ValueError:
                # If that fails, try parsing as milliseconds
                date = datetime.fromtimestamp(d['x'] / 1000).date()
            
            # Convert start_date and end_date to date objects if they're not already
            start_date = start_date.date() if isinstance(start_date, datetime) else start_date
            end_date = end_date.date() if isinstance(end_date, datetime) else end_date
            
            if start_date <= date <= end_date:
                filtered_data.append((pd.to_datetime(date), d['y']))
        return filtered_data

    def _get_coingecko_historical_data(self, coin, start_date, end_date):
        if coin == "total":
            url = f"{self.coingecko_url}/global"
            params = {}
        else:
            url = f"{self.coingecko_url}/coins/{coin}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': int(datetime.combine(start_date, datetime.min.time()).timestamp()),
                'to': int(datetime.combine(end_date, datetime.min.time()).timestamp())
            }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data from CoinGecko: {response.status_code}")
            return pd.DataFrame(columns=['date', 'market_cap'])

        data = response.json()
        
        if coin == "total":
            total_market_cap = data['data']['total_market_cap']['usd']
            df = pd.DataFrame([{'date': end_date, 'total_crypto_market_cap': total_market_cap}])
        else:
            if 'market_caps' not in data:
                print("Unexpected response structure from CoinGecko")
                return pd.DataFrame(columns=['date', 'market_cap'])

            df = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df = df.drop('timestamp', axis=1)
            df = df.rename(columns={'market_cap': 'btc_market_cap'})
        
        # Ensure 'date' column is datetime64[ns]
        df['date'] = pd.to_datetime(df['date'])
        
        return df

# Example usage:
if __name__ == "__main__":
    fetcher = ExternalDataFetcher()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    data = fetcher.get_data(start_date, end_date)
    print(data.head())