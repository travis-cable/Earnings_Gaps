# Import packages
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
from tqdm import tqdm

# Pull in data from the given time range for the current ticker
def Stock_data(tick, EDF_, GPL):

    # Columns in the current earnings Dataframe
    Earnings_features = EDF_.columns

    # Choose a 20-day window for ATR SMA calculation
    n = 20

    # Assume ~55 days before next earnings event
    NED = 50

    # Identify earliest and latest earnings events to use
    SD = EDF_.index[-1] - timedelta(int(np.round(4*n,0)))
    ED = EDF_.index[0]

    # Load in OHLCV for the current stock
    Stock = yf.Ticker(tick)
    data = yf.download(tick, start = SD, end = ED)

    # Add in Average True Range Percent, Modified Williams R and Volume Percentage
    data['ATRP'] = 100.0 * (data['High'] - data['Low'])/ ((data['High'] - data['Low']).rolling(n).mean())
    data['MWR'] = 100.0 * (data['Open'] - data['Close']) / (data['High'] - data['Low'])
    data['VP'] = 100.0 * data['Volume'] / data['Volume'].rolling(n).mean()
    data.dropna(inplace = True)

    # Check to ensure all of the data was loaded in. If not, truncate the earnings
    # info to match up with the stock data
    if data.index[0] > EDF_.index[-1]:
        EDF_ = EDF_.loc[EDF_.index >= data.index[0]]

    # Counters for the loop and instances saved
    LC = 1
    OI = 0
    OA = None

    # Loop through every earnings event to see if there's been a significant gap
    for index in tqdm(EDF_.index[1:]):

        # As long as earnings were not released on a holiday or over the weekend
        if index not in data.index:
            continue
        else:

            # Pick out the current row and next successive row in the stock data DataFrame
            IR = data.index.get_loc(index)
            IR_P1 = IR + 1

            # Find the day before the next earnings date in the stock dataframe (NER)
            NER = pd.DataFrame(data.index - (index + timedelta(NED))).abs().idxmin().iloc[0]

            # Determine if earnings were released before market open or after market close by assessing daily trading
            # volume
            if (data.iloc[IR_P1, 5] - data.iloc[IR, 5]) > 0:
                Gap_ind = IR_P1
            else:
                Gap_ind = IR

            # Now determine if the current earnings data point experienced a gap that's greater that the desired
            # Gap Percentage (GPL)
            GP = 100.0 * (data.iloc[Gap_ind, 0] - data.iloc[Gap_ind - 1, 3]) / \
                 data.iloc[Gap_ind - 1, 3]

            if np.abs(GP) > GPL:

                # Calculate dependent variables
                # End State: 1 if the closing price in 60 days in higher/lower than the gap day close for
                # gap up/down, respectively. 0 otherwise
                # Max simple return
                # Days until max return

                EC = data.iloc[NER, 3]
                # Gap up
                if GP > 0:

                    # For gap up, we're looking for the stock to run up after earnings, so determine the highest high
                    # until the next earnings event
                    Max_price = data.iloc[Gap_ind+1:NER, 1].max()
                    Max_day = data.iloc[Gap_ind+1:NER, 1].idxmax()

                    if EC > data.iloc[Gap_ind, 3]:
                        ES = 1
                    else:
                        ES = 0

                    # Generate a date range with business days
                    business_days = pd.date_range(start=data.index[Gap_ind], end=Max_day, freq='B')

                    # Calculate the number of trading days till max
                    num_trading_days = len(business_days)

                    # Calculate max simple return
                    Max_R = 100.0 * (Max_price / data.iloc[Gap_ind, 3] - 1.0)

                # Gap down
                else:

                    # For gap down, we're looking for the stock to run down after earnings, so determine the lowest
                    # low until the next earnings event
                    Min_price = data.iloc[Gap_ind+1:NER, 2].min()
                    Min_day = data.iloc[Gap_ind+1:NER, 2].idxmin()

                    if EC < data.iloc[Gap_ind, 3]:
                        ES = 1
                    else:
                        ES = 0

                    # Generate a date range with business days
                    business_days = pd.date_range(start=data.index[Gap_ind], end=Min_day, freq='B')

                    # Calculate the number of trading days till max
                    num_trading_days = len(business_days)

                    # Calculate max simple return
                    Max_R = 100.0 * (Min_price / data.iloc[Gap_ind, 3] - 1.0)


                # The output array (6 input features and 3 outputs)
                temp = np.array([tick, index, EDF_.loc[index, Earnings_features[0]], EDF_.loc[index, Earnings_features[1]],
                                 data.iloc[Gap_ind, 6], data.iloc[Gap_ind, 7], data.iloc[Gap_ind, 8], GP, ES, Max_R,
                                 num_trading_days])

                if OI == 0:
                    OA = temp
                else:
                    OA = np.vstack((OA,temp))

                # Increment the number of items instances output
                OI += 1

    return OA