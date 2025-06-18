ChainSight: The Ultimate Crypto Market Analysis Powerhouse
Unleash the Power of Data-Driven Crypto Decisions.

ChainSight is not just another crypto tool; it's your all-in-one, real-time, intelligent cryptocurrency market analysis solution. Designed for both novice enthusiasts and seasoned traders, ChainSight combines cutting-edge technical indicators with a proprietary Machine Learning model to deliver actionable insights and clear, confident recommendations, helping you navigate the volatile crypto landscape with unparalleled precision.

Stop guessing. Start knowing. ChainSight brings clarity to chaos.

✨ Why ChainSight is the BEST Analysis Tool on the Internet
Real-Time, Dynamic Insights: Automatically fetches and processes live market data, providing continuous updates to keep you ahead of the curve.

Comprehensive Technical Analysis: Integrates an exhaustive suite of industry-standard indicators including:

Simple Moving Averages (SMA): Identify trends and potential reversals.

Relative Strength Index (RSI): Pinpoint overbought and oversold conditions.

Moving Average Convergence Divergence (MACD): Gauge momentum and trend strength.

Bollinger Bands (BB): Measure volatility and identify potential price extremes.

Stochastic Oscillator: Confirm trend direction and anticipate reversals.

Average True Range (ATR): Quantify market volatility.

Ichimoku Cloud: A holistic, future-projecting indicator for comprehensive trend and support/resistance analysis.

Intelligent Machine Learning Predictions: A built-in Linear Regression model predicts future price movements, providing a crucial forward-looking perspective. Special handling for stablecoins ensures accurate predictions for pegged assets.

Actionable Recommendations: Translates complex data into clear, concise BUY, SELL, or HOLD recommendations, complete with a confidence score and suggested execution timeframe.

Prioritized Coin Monitoring: Automatically tracks top popular cryptocurrencies by market cap and intelligently prioritizes trending coins, including meme coin favorites, ensuring you're always focused on relevant assets.

Detailed & Exportable Reports: Displays all analyzed data and recommendations in an easy-to-read terminal output and allows for convenient export to CSV for further analysis or record-keeping.

User-Friendly Interface: A simple command-line interface makes it accessible for anyone to run and understand.

🚀 Getting Started
Follow these simple steps to set up and run ChainSight on your local machine.

Prerequisites
Python 3.8+

Internet connection to fetch data from CoinGecko API.

Installation
Clone the Repository (or download the files):

git clone https://github.com/your-username/chainsight.git
cd chainsight

Install Dependencies:
ChainSight relies on a few external Python libraries. It's highly recommended to use a virtual environment.

# (Optional) Create a virtual environment
python -m venv venv
# Activate the virtual environment
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt

⚙️ Usage
Once installed, running ChainSight is straightforward:

python chainsight.py

The script will prompt you to enter a currency to compare against (e.g., usd, eur). Press Enter to use the default (usd).

ChainSight will then continuously fetch data, perform analysis, and display real-time reports directly in your terminal, updating every 60 seconds (configurable).

To stop the application, simply press Ctrl+C.

Configuration (Optional)
You can adjust various parameters by editing the chainsight.py file directly:

DEFAULT_VS_CURRENCY: The default fiat currency for comparison.

CHART_DATA_DAYS: Number of days of historical data to fetch for analysis.

PREDICTION_TIMEFRAME_HOURS: How many hours into the future the ML model will predict.

UPDATE_INTERVAL_SECONDS: How often the market data is refreshed.

SMA_WINDOW, RSI_WINDOW, FAST_EMA_WINDOW, etc.: Adjust the look-back periods for various technical indicators to fine-tune the analysis to your trading style.

MEME_COIN_IDS: Customize the list of meme coin IDs for prioritized trending analysis.

STABLECOIN_IDS: Define known stablecoin IDs for special ML prediction handling.

📊 Output Example
ChainSight provides a clear and comprehensive report for each monitored cryptocurrency, including:

Current Market Price

Values for all calculated Technical Indicators (SMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, Ichimoku Cloud components)

Machine Learning Predicted Next Close Price

Overall Recommendation: A confident BUY, SELL, or HOLD signal, summarized from all indicators, along with a confidence percentage.

Detailed Analysis: A breakdown of insights from each indicator.

Example snippet (actual output is more detailed):

--- Live Crypto Signal Report (2024-05-15 10:30:00 UTC) ---
================================================================================
Analysing: Bitcoin (BTC)
  Latest Date/Time: 2024-05-15 10:00
  Latest Price (Close): 62500.0000 USD
  Short SMA (10 periods): 62000.0000 USD
  Long SMA (30 periods): 61500.0000 USD
  RSI (14 periods): 65.23
  MACD Line (12,26): 150.0000
  Signal Line (9): 120.0000
  MACD Histogram: 30.0000
  Bollinger Bands (20,2 StdDev):
    Middle Band: 61800.0000 USD
    Upper Band: 63000.0000 USD
    Lower Band: 60600.0000 USD
  Stochastic Oscillator (14,3): %K=75.50, %D=70.10
  Average True Range (14 periods): 500.0000 USD
  Ichimoku Cloud:
    Tenkan-sen (9): 62300.0000
    Kijun-sen (26): 61900.0000
    Senkou Span A: 62100.0000
    Senkou Span B (52): 61800.0000
    Chikou Span (Lag 26): 60000.0000
  ML Predicted Next Close (4 hours ahead): 62750.0000 USD
  Recommendation:
  [92mBUY for Bitcoin (As of 10:30 AM EST) with 85.0% Confidence:[0m
   SMA Trend: Bullish (Short SMA 62000.0000 > Long SMA 61500.0000).
   RSI (65.23): Neutral. No strong overbought/oversold condition.
   MACD: Bullish momentum building (MACD Line 150.0000 > Signal Line 120.0000, Histogram positive 30.0000).
   Bollinger Bands: Price (62500.0000 USD) is within bands (60600.0000-63000.0000 USD).
   Stochastic (75.50/70.10): Bullish crossover. Momentum shifting up.
   ATR (14 periods): 500.0000 USD. Current volatility measure.
   Ichimoku Cloud: Price is above Tenkan/Kijun (Bullish); Tenkan-sen > Kijun-sen (Bullish momentum); Price is within the Cloud (Neutral/Consolidation); Bullish Cloud (Senkou A > Senkou B); Chikou Span is above lagged price (Bullish).
   ML Prediction for 4 hours ahead: Next close predicted to rise by 0.40% to 62750.0000 USD.
   Consider executing a BUY signal within the next 4 hours ahead.
--------------------------------------------------------------------------------

⚠️ Disclaimer
ChainSight is for informational and educational purposes only and should NOT be considered financial or investment advice. The cryptocurrency market is highly volatile and unpredictable. Machine Learning predictions are experimental and should not be solely relied upon for making trading decisions. Always conduct your own research and consult with a qualified financial professional before making any investment decisions. The creators of ChainSight are not responsible for any financial losses incurred from using this tool.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
