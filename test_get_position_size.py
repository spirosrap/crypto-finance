from config import API_KEY_PERPS, API_SECRET_PERPS
from coinbaseservice import CoinbaseService
from crypto_alert_monitor import get_btc_perp_position_size

def main():
    cb_service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    size = get_btc_perp_position_size(cb_service)
    print(f"BTC-PERP-INTX open position size: {size}")

if __name__ == "__main__":
    main() 