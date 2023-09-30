from pathlib import Path
import pandas as pd
import requests
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Check if a .env file exists
assert load_dotenv(), "Failed to load .env file"

# Load symbols table
symbols_save_path = Path("data/symbols.parquet")
symbols = pd.read_parquet(symbols_save_path)
symbols_to_gather = symbols["symbol"].tolist()

# Some of the symbols have trouble when pulling from Alpha Vantage
problem_symbols = {
    "FBHS",
    "BMC",
    "TLAB",
    "CBE",
    "DF",
    "EK",
    "LEH",
    "DJ",
    "SLR",
    "JOYG",
    "FRE",
    "ACE",
    "FNM",
    "ABK",
    "SHLD",
    "AV",
    "TRB",
    "MEE",
    "CFC",
    "CCE",
    "ESV",
    "NOVL",
    "WIN",
    "TE",
    "COG",
    "SMS",
    "GR",
    "BS",
    "FB",
    "KFT",
    "FRC",
    "RX",
    "HFC",
    "QTRN",
    "ADS",
    "RE",
    "PTV",
    "MOLX",
    "LDW",
    "NYX",
    "TYC",
    "SBL",
    "TSO",
    "GLK",
    "MI",
    "GENZ",
    "SGP",
    "WLTW",
    "JCP",
    "KSE",
    "ABS",
    "WFR",
    "CEPH",
    "HPH",
}

# Loop through the list of earnings_tickers, pulling quarterly earnings data from Alpha Vantage
loop_counter = 0
failed_symbols = 0
key = os.getenv("AV_key")
progress_bar = tqdm(symbols_to_gather, desc="Gathering earnings data")
earnings_list = {}
for ticker in progress_bar:

    # Show current symbol in progress bar
    progress_bar.set_postfix_str(ticker.ljust(5, " "))

    if ticker in problem_symbols:
        failed_symbols += 1
        continue

    # Create a URL for each of the earnings_tickers in the style specified by AV
    url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol={}&apikey={}'.format(ticker, key)

    r = requests.get(url)

    # Convert the API pull to .json and then to DataFrame
    data = r.json()

    # Try to pull the quarterly earnings data from the json file
    try:
        quarterly_earnings = data['quarterlyEarnings']
    except:
        failed_symbols += 1
        continue

    # Assemble the data into a temporary dataframe
    temporary_df = pd.DataFrame(quarterly_earnings)
    temporary_df['reportedDate'] = pd.to_datetime(temporary_df['reportedDate'])
    temporary_df['ticker'] = [ticker] * temporary_df.shape[0]
    temporary_df.set_index(['reportedDate', 'ticker'], inplace=True)

    # Join the existing ticker data and with the recently acquired
    earnings_list[ticker] = temporary_df

# Sort by symbol and date
earnings_df = pd.concat(earnings_list, axis=0)

# Sort by symbol and date
earnings_df.sort_index(inplace=True)

# earnings_dtypes = {
#     "fiscalDateEnding": pd.CategoricalDtype(),
#     "reportedEPS": pd.Float64Dtype(),
#     "estimatedEPS": pd.Float64Dtype(),
#     "surprise": pd.Float64Dtype(),
#     "surprisePercentage": pd.Float64Dtype(),
# }
# earnings_df = earnings_df.astype(earnings_dtypes)

# Save the table
earnings_df_save_path = Path("data/earnings_try.parquet")
earnings_df.to_parquet(earnings_df_save_path)