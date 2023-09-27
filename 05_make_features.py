# Import packages
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Check if a .env file exists
assert load_dotenv(), "Failed to load .env file"

# Load in the earnings data from file
earnings_save_path = Path("data/earnings.parquet")
earnings_df = pd.read_parquet(earnings_save_path)
earnings_df['reportedEPS'] = earnings_df['reportedEPS'].astype(float)
earnings_groups = earnings_df.groupby('ticker')

# Load in the technical indicators data from file
tech_indicators_path = Path("data/technical_indicators.parquet")
indicators_df = pd.read_parquet(tech_indicators_path).drop(['atr', 'dx', 'adxr', 'macds', 'macdh'], axis=1)
indicators_groups = indicators_df.groupby('ticker')

# With the dataframe loaded in, sweep through the ticker symbols
tickers = list(earnings_groups.groups.keys())

# Choose a gap setting and windows to label data and save returns
gap_threshold = 0.02
label_days = [1, 5, 10, 25]

event_dfs = {}

# Remove problem earnings_tickers from list
tickers.remove('MNK')
tickers.remove('OTIS')

# SPX features
SPX_ticker = '^GSPC'
SPX_features = indicators_groups.get_group(SPX_ticker).copy()
SPX_features.reset_index(level='ticker', inplace=True)
SPX_features.sort_index(inplace=True)

SPX_feature_names = [s + '_SPX' for s in list(indicators_df.columns)[6:]]


output_names = ['date', 'earnings_surprise', 'earnings_quarterly_growth', 'earnings_YoY_growth'] + \
               list(indicators_df.columns)[6:] +['gap_fraction', 'candle_param'] + SPX_feature_names + \
               [f'label_after_{day}_days' for day in label_days] + \
               [f'returns_after_{day}_days' for day in label_days]

progress_bar = tqdm(tickers, desc="Forming features and labels")
for ticker in progress_bar:

    # Show current symbol in progress bar
    progress_bar.set_postfix_str(ticker.ljust(5, " "))

    # Get the data for the current stock
    this_stock_earnings = earnings_groups.get_group(ticker).copy()

    # Create some other features for the earnings data
    this_stock_earnings['quarterly_growth'] = this_stock_earnings['reportedEPS'].pct_change(periods=1)
    this_stock_earnings['YoY_growth'] = this_stock_earnings['reportedEPS'].pct_change(periods=4)
    this_stock_earnings.dropna(inplace=True)

    # Bring in the technical indicators
    this_stock_indicators = indicators_groups.get_group(ticker).copy()
    this_stock_indicators.reset_index(level='ticker', inplace=True)
    this_stock_indicators.sort_index(inplace=True)

    # Sweep through all of the earnings events
    saved_events = 0
    for i in range(this_stock_earnings.shape[0] - 1):

        reported_date = this_stock_earnings.index[i][0]

        # If the earnings report was released on a non-market day
        if reported_date not in this_stock_indicators.index or reported_date not in SPX_features.index:
            continue

        # Grab the row location in the OHLC DataFrame from the earnings event
        date_row = this_stock_indicators.index.get_loc(reported_date)

        # Determine if the report was released before market open, during market or after market close by
        # analyzing the volume (highest volume = gap day)
        if this_stock_indicators.iloc[date_row + 1, 6] > this_stock_indicators.iloc[date_row, 6]:
            date_row += 1

        # Now save the data for the current earnings event if it caused a significant gap
        gap_day_open = this_stock_indicators.iloc[date_row, 1]
        gap_day_high = this_stock_indicators.iloc[date_row, 2]
        gap_day_low = this_stock_indicators.iloc[date_row, 3]
        gap_day_close = this_stock_indicators.iloc[date_row, 4]

        previous_day_close = this_stock_indicators.iloc[date_row - 1, 4]

        if gap_day_high == gap_day_low:
            continue

        candle_shape = (gap_day_close - gap_day_open) / (gap_day_high - gap_day_low)
        gap_fraction = gap_day_open/previous_day_close - 1.0

        # # Select only the gap up stocks for now
        # if np.abs(gap_fraction) < gap_threshold:
        #     continue

        # Create the labels and save features for the current earnings gap
        gap_label = 0
        if gap_fraction > 0:
            gap_label += 1

        label_vector = []
        returns_vector = []
        for j in range(len(label_days)):

            look_ahead_close = this_stock_indicators.iloc[date_row + label_days[j], 4]
            look_ahead_return = look_ahead_close / gap_day_close - 1
            #
            # # If the stock gaped up
            # if gap_label > 0:

            if look_ahead_close > gap_day_close:
                labeled_close = 1
            else:
                labeled_close = 0
            #
            # else:
            #
            #     if look_ahead_close < gap_day_close:
            #         labeled_close = 1
            #     else:
            #         labeled_close = 0

            # Add label and return to the Dataframe
            returns_vector.append(look_ahead_return)
            label_vector.append(labeled_close)


        # Add the label vector to the feature vector then save
        try:
            earnings_surpise = float(this_stock_earnings.iloc[i, 3])
        except:
            continue

        # Save the features
        current_event_vector = np.array([reported_date] + list(this_stock_earnings.iloc[i, [3, 5, 6]]) +
                                        list(this_stock_indicators.iloc[date_row, 7:]) +
                                        [gap_fraction, candle_shape] + list(SPX_features.iloc[date_row, 7:]) +
                                        label_vector + returns_vector)

        if saved_events == 0:
            saved_events_table = current_event_vector
        else:
            saved_events_table = np.vstack((saved_events_table, current_event_vector))

        saved_events += 1

    # Assemble a DataFrame with the current stock's event table
    if saved_events > 1:
        stock_specific_df = pd.DataFrame(saved_events_table, columns=output_names)
        stock_specific_df.set_index('date', inplace=True)
        event_dfs[ticker] = stock_specific_df
    elif saved_events == 1:
        stock_specific_df = pd.DataFrame(np.expand_dims(saved_events_table, axis=1).T, columns=output_names)
        stock_specific_df.set_index('date', inplace=True)
        event_dfs[ticker] = stock_specific_df

# Group the list into a single DataFrame
earnings_event_df = pd.concat(event_dfs, axis=0)

# Convert to correct data types
earnings_dtypes = {
    "earnings_surprise": pd.Float32Dtype(),
    "earnings_quarterly_growth": pd.Float32Dtype(),
    "earnings_YoY_growth": pd.Float32Dtype(),
    "adx": pd.Float32Dtype(),
    "trix": pd.Float32Dtype(),
    "cci": pd.Float32Dtype(),
    "macd": pd.Float32Dtype(),
    "rsi_14": pd.Float32Dtype(),
    "kdjk": pd.Float32Dtype(),
    "wr_14": pd.Float32Dtype(),
    "clv": pd.Float32Dtype(),
    "cmf": pd.Float32Dtype(),
    "gap_fraction": pd.Float32Dtype(),
    "candle_param": pd.Float32Dtype(),
    "atr_percent": pd.Float32Dtype(),
    "adx_SPX": pd.Float32Dtype(),
    "trix_SPX": pd.Float32Dtype(),
    "cci_SPX": pd.Float32Dtype(),
    "macd_SPX": pd.Float32Dtype(),
    "rsi_14_SPX": pd.Float32Dtype(),
    "kdjk_SPX": pd.Float32Dtype(),
    "wr_14_SPX": pd.Float32Dtype(),
    "clv_SPX": pd.Float32Dtype(),
    "cmf_SPX": pd.Float32Dtype(),
    "atr_percent_SPX": pd.Float32Dtype(),
    "label_after_1_days": pd.Int32Dtype(),
    "label_after_5_days": pd.Int32Dtype(),
    "label_after_10_days": pd.Int32Dtype(),
    "label_after_25_days": pd.Int32Dtype(),
    "returns_after_1_days": pd.Float32Dtype(),
    "returns_after_5_days": pd.Float32Dtype(),
    "returns_after_10_days": pd.Float32Dtype(),
    "returns_after_25_days": pd.Float32Dtype(),
}
earnings_event_df = earnings_event_df.astype(earnings_dtypes)

# Sort by symbol and date
earnings_event_df.sort_index(inplace=True)

# Save the table
earnings_df_save_path = Path("data/all_earnings_events.parquet")
earnings_event_df.to_parquet(earnings_df_save_path)