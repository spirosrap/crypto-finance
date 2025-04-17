import logging
import sys
import os
from datetime import datetime, timedelta
from coinbaseservice import CoinbaseService
from historicaldata import HistoricalData
from bitcoinpredictionmodel import BitcoinPredictionModel
from technicalanalysis import TechnicalAnalysis

# Try to import API keys from config or use environment variables
try:
    from config import API_KEY, API_SECRET
except ImportError:
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('API_SECRET')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    # Check API keys
    if not API_KEY or not API_SECRET:
        logger.error("API keys not found. Please set API_KEY and API_SECRET in config.py or as environment variables.")
        return
        
    # Initialize services
    logger.info("Initializing services...")
    coinbase_service = CoinbaseService(api_key=API_KEY, api_secret=API_SECRET)
    historical_data = HistoricalData(coinbase_service.client)
    
    # Test parameters
    product_id = "BTC-USDC"
    granularity = "FIVE_MINUTE"
    
    # Fetch recent candles
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=6)
    
    logger.info(f"Fetching candles for {product_id} from {start_date} to {end_date}")
    candles = historical_data.get_historical_data(product_id, start_date, end_date, granularity=granularity)
    
    if not candles:
        logger.error("No candles received")
        return
    
    logger.info(f"Fetched {len(candles)} candles")
    
    # First, check the structure of the candles
    first_candle = candles[0]
    logger.info(f"First candle structure: {first_candle}")
    
    # Initialize prediction model
    logger.info("Initializing prediction model...")
    model = BitcoinPredictionModel(coinbase_service, product_id, granularity)
    
    try:
        # Load the model
        logger.info("Loading model...")
        model.load_model()
        
        # Prepare data
        logger.info("Preparing data...")
        df, X, y = model.prepare_data(candles)
        
        if X is None or df is None:
            logger.error("Failed to prepare data")
            return
        
        logger.info(f"Data prepared successfully. X shape: {X.shape}")
        
        # Make a prediction
        logger.info("Making prediction...")
        prediction = model.predict(X.iloc[-1:])
        
        if prediction is None:
            logger.error("Failed to make prediction")
            return
        
        logger.info(f"Prediction: {prediction}")
        
        # Use TechnicalAnalysis to get a signal
        logger.info("Getting technical analysis signal...")
        ta = TechnicalAnalysis(coinbase_service, candle_interval=granularity, product_id=product_id)
        ta._ensure_models_loaded()
        
        signal = ta.get_bitcoin_prediction_signal(candles)
        logger.info(f"Signal: {signal}")
        
    except Exception as e:
        logger.error(f"Error testing candle processing: {e}", exc_info=True)

if __name__ == "__main__":
    main() 