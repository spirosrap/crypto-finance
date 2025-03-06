# CLAUDE.md - Crypto Finance Development Guide

## Common Run Commands
- Basic backtesting: `python base.py --start_date 2023-01-01 --end_date 2023-12-31`
- Run with specific product: `python base.py --product_id ETH-USDC`
- Market analysis: `python market_analyzer.py`
- Advanced analysis: `python advanced_market_analyzer.py`
- Run multiple backtests: `python run_all_commands.py`
- UI/GUI version: `python market_ui.py`

## Test Commands
- Run all tests: `python -m unittest discover tests`
- Run specific test: `python -m unittest tests.test_technicalanalysis`
- Run coinbase service test: `python tests/test_coinbase_service.py`
- Create test file: `touch tests/test_<module_name>.py`

## Environment Setup
- Python requirement: 3.11+
- Create virtual environment: `conda create -n crypto-env python=3.11`
- Activate environment: `conda activate crypto-env`
- Install dependencies: `pip install -r requirements.txt`
- Set up config: Create `config.py` with required API keys

## Code Style Guidelines
- **Imports**: Follow PEP8 order: stdlib > third-party > local. Group imports.
- **Types**: Use explicit typing (from typing import List, Dict, Optional, etc).
- **Error handling**: Use specific exception classes and contextual error messages.
- **Naming**: UpperCamelCase for classes, snake_case for functions/variables.
- **Functions**: Use type hints and detailed docstrings when possible.
- **Documentation**: Use docstrings for classes and non-trivial functions.
- **Validation**: Use validation decorators for critical data inputs.
- **Linting**: Run `flake8` manually - not currently in CI pipeline

## Project Structure
- `/technical_analysis/`: Core analysis modules and strategies
- `/tests/`: Unit and integration tests
- `/crypto_market_analyzer/`: Market analysis components
- `/trading/`: Trading models and visualization
- `/train/`: ML model training scripts
- Root-level Python files: Main executable modules

## Data Handling
- Historical data stored in `/candle_data/`
- Use `HistoricalData` class for data retrieval and persistence
- Default timeframe is `ONE_HOUR` for most analysis
- ML models cache in `/cachedir/` and `/catboost_info/`

## API Usage Notes
- Coinbase API has rate limits (~15 requests/sec)
- Use error handling with exponential backoff for API calls
- Cache responses when possible to reduce API usage
- Monitor API key permissions and rotate regularly
- Fallback mechanisms in place for API outages

## Common Debugging Tips
- `TechnicalAnalysisError`: Usually from insufficient price data
- Connection errors: Check API credentials in `config.py`
- Model errors: Ensure `cachedir` permissions are correct
- TA-Lib errors: Verify TA-Lib installation for your OS
- Candle data issues: Check `historicaldata.py` logs