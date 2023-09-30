# Originally written by Jackson Walker (https://github.com/jacksonrgwalker), modified by Travis Cable

# Import packages
import pandas as pd
from pathlib import Path
import yfinance as yf
from tqdm import tqdm
from dotenv import load_dotenv

# Check if a .env file exists
assert load_dotenv(), "Failed to load .env file"

# Load in the earnings data from file
earnings_save_path = Path("data/earnings.parquet")
earnings_df = pd.read_parquet(earnings_save_path)
earnings_groups = earnings_df.groupby('ticker')

# With the dataframe loaded in, sweep through the ticker symbols
earnings_tickers = list(earnings_groups.groups.keys()) + ['^GSPC']
earnings_tickers.remove('MNK')
earnings_tickers.remove('OTIS')

ohlc_list = {}
start_date = earnings_df.index.get_level_values(0).min()

progress_bar = tqdm(earnings_tickers, desc="Forming features and labels")
for ticker in progress_bar:

    # Show current symbol in progress bar
    progress_bar.set_postfix_str(ticker.ljust(5, " "))

    # API call to Yahoo Finance to get daily adjusted OHLC
    ohlc = yf.download(ticker, start=start_date, progress=False)
    ohlc.reset_index(inplace=True)
    ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    ohlc['ticker'] = [ticker] * ohlc.shape[0]
    ohlc.set_index(['Date', 'ticker'], inplace=True)

    ohlc_list[ticker] = ohlc

# Group the list into a single DataFrame
ohlc_df = pd.concat(ohlc_list, axis=0)

# Sort by symbol and date
ohlc_df.sort_index(inplace=True)

ohlc_dtypes = {
    "Open": pd.Float64Dtype(),
    "High": pd.Float64Dtype(),
    "Low": pd.Float64Dtype(),
    "Close": pd.Float64Dtype(),
    "Adj Close": pd.Float64Dtype(),
    "Volume": pd.Int64Dtype(),
}

ohlc_df = ohlc_df.astype(ohlc_dtypes)

# Save the table
ohlc_df_save_path = Path("data/ohlc.parquet")
ohlc_df.to_parquet(ohlc_df_save_path)